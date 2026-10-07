import asyncio
from types import SimpleNamespace

import apriltag
import numpy as np
import cv2


from typing import (Any, ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple, cast)
from typing_extensions import Self

from viam.components.pose_tracker import PoseTracker
from viam.components.camera import Camera
from viam.media.video import CameraMimeType, NamedImage, ViamImage
from viam.proto.common import ResponseMetadata
from viam.module.module import Module
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Geometry, PointCloudObject, PoseInFrame, Pose, RectangularPrism, ResourceName, Transform, Vector3
from viam.proto.service.vision import Classification, Detection, Detection3D, GetPropertiesResponse
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.registry import Registry
from viam.resource.types import Model, ModelFamily, RESOURCE_TYPE_COMPONENT, RESOURCE_TYPE_SERVICE
from viam.errors import ResourceNotFoundError
from viam.logging import getLogger
from viam.services.vision import CaptureAllResult, Vision
from viam.spatialmath import RotationMatrix
from viam.utils import struct_to_dict, ValueTypes
from viam.media.utils.pil import viam_to_pil_image


# required attributes
cam_attr = "camera_name"
family_attr = "tag_family"
width_attr = "tag_width_mm"
confidence_threshold_attr = "confidence_threshold_pct"
bbox_padding_attr = "bbox_padding_px"

LOGGER = getLogger(__name__)


def _color_image_from_camera_images(
    cam_images: Sequence[NamedImage],
) -> NamedImage:
    """Pick the color frame from a camera GetImages response (Orbbec, crop-camera, etc.)."""
    if not cam_images:
        raise Exception("camera returned no images")

    def is_depth(img: NamedImage) -> bool:
        name = (img.name or "").lower()
        if "depth" in name:
            return True
        return img.mime_type in (CameraMimeType.VIAM_RAW_DEPTH, CameraMimeType.PCD)

    by_name = {(img.name or "").lower(): img for img in cam_images}
    if "color" in by_name and not is_depth(by_name["color"]):
        return by_name["color"]

    jpeg = next(
        (img for img in cam_images if img.mime_type == CameraMimeType.JPEG and not is_depth(img)),
        None,
    )
    if jpeg is not None:
        return jpeg

    color = next((img for img in cam_images if not is_depth(img)), None)
    if color is not None:
        return color

    raise Exception("camera returned no color images")


def _gray_from_viam_image(image: ViamImage) -> tuple[np.ndarray, int, int]:
    pil_img = viam_to_pil_image(image)
    rgb = np.array(pil_img)
    if rgb.ndim == 2:
        gray = rgb
    else:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    width = int(image.width or rgb.shape[1])
    height = int(image.height or rgb.shape[0])
    return gray, width, height


# decimate=2.0 matches the old dt_apriltags default: ~2.7x faster than 1.0; 1.0 finds smaller/farther tags.
# Build the native detector once per resource in reconfigure() and reuse it here.
def _detect_apriltags(
    detector: Any,
    gray_image: np.ndarray,
    *,
    estimate_tag_pose: bool = False,
    camera_params: Optional[Sequence[float]] = None,
    tag_size: Optional[float] = None,
) -> list[Any]:
    """Detect tags and adapt apriltag-python's dicts to attribute objects.

    Each result has tag_id, corners, center, decision_margin and, when
    estimate_tag_pose is set, pose_R (3x3) and pose_t (3x1, meters).
    """
    # The native detector requires a C-contiguous 2-D uint8 buffer.
    gray = np.ascontiguousarray(gray_image, dtype=np.uint8)
    tags = []
    for d in detector.detect(gray):
        tag = SimpleNamespace(
            tag_id=d["id"], corners=d["lb-rb-rt-lt"], center=d["center"], decision_margin=d["margin"]
        )
        if estimate_tag_pose:
            fx, fy, cx, cy = camera_params
            pose = detector.estimate_tag_pose(d, tag_size, fx, fy, cx, cy)
            tag.pose_R, tag.pose_t = pose["R"], pose["t"]
        tags.append(tag)
    return tags


def _tag_pose(tag: Any) -> Pose:
    """Tag pose in the camera frame: mm translation, orientation vector in degrees."""
    # The detector reports pose_t in meters; Viam poses are in mm.
    x, y, z = (float(v) * 1000 for v in tag.pose_t.flatten())
    return RotationMatrix(tag.pose_R.flatten()).to_quaternion().to_pose(x, y, z)


async def _camera_intrinsics(camera: Camera, timeout: Optional[float]) -> List[float]:
    """[fx, fy, cx, cy] for the detector pose estimator."""
    intr = (await camera.get_properties(timeout=timeout)).intrinsic_parameters
    return [intr.focal_x_px, intr.focal_y_px, intr.center_x_px, intr.center_y_px]


def _tag_confidence(tag: Any) -> float:
    # decision_margin is commonly 20-150; map so good tags clear segmenter defaults.
    return float(min(max(tag.decision_margin / 40.0, 0.0), 1.0))


def _parse_optional_int(attrs: Mapping[str, Any], key: str, default: int) -> int:
    value = attrs.get(key, default)
    if value is None:
        return default
    if not isinstance(value, (int, float)):
        raise Exception(f"{key} must be an integer")
    return int(value)


def _expand_bbox(
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    padding_px: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    if padding_px <= 0:
        return x_min, y_min, x_max, y_max
    return (
        max(0, x_min - padding_px),
        max(0, y_min - padding_px),
        min(width - 1, x_max + padding_px),
        min(height - 1, y_max + padding_px),
    )


def _parse_confidence_threshold(attrs: Mapping[str, Any]) -> float:
    threshold = attrs.get(confidence_threshold_attr, 0.0)
    if threshold is None:
        return 0.0
    if not isinstance(threshold, (int, float)):
        raise Exception(confidence_threshold_attr + " must be a number between 0.0 and 1.0")
    threshold = float(threshold)
    if threshold < 0.0 or threshold > 1.0:
        raise Exception(confidence_threshold_attr + " must be between 0.0 and 1.0")
    return threshold


def _parse_optional_tag_width(attrs: Mapping[str, Any]) -> Optional[float]:
    value = attrs.get(width_attr)
    if value is None:
        return None
    # bool is an int subclass; reject it explicitly.
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise Exception(width_attr + " must be a positive number")
    return float(value)


def _tags_to_detections(
    tags: Sequence[Any],
    width: int,
    height: int,
    *,
    confidence_threshold_pct: float = 0.0,
    bbox_padding_px: int = 0,
) -> List[Detection]:
    if width <= 0 or height <= 0:
        raise Exception("image width and height are required for detections")

    detections: List[Detection] = []
    for tag in tags:
        confidence = _tag_confidence(tag)
        if confidence < confidence_threshold_pct:
            continue
        xs = tag.corners[:, 0]
        ys = tag.corners[:, 1]
        # Detection bbox fields are int64 in the vision proto, not float.
        x_min = int(round(float(np.min(xs))))
        y_min = int(round(float(np.min(ys))))
        x_max = int(round(float(np.max(xs))))
        y_max = int(round(float(np.max(ys))))
        x_min, y_min, x_max, y_max = _expand_bbox(
            x_min, y_min, x_max, y_max, bbox_padding_px, width, height
        )
        detections.append(
            Detection(
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
                x_min_normalized=x_min / width,
                y_min_normalized=y_min / height,
                x_max_normalized=x_max / width,
                y_max_normalized=y_max / height,
                confidence=confidence,
                class_name=str(tag.tag_id),
            )
        )
    return detections


def _tags_to_detections_3d(
    tags: Sequence[Any],
    name_prefix: str,
    camera_name: str,
    tag_width_mm: float,
    confidence_threshold_pct: float,
) -> List[Detection3D]:
    """One Detection3D per tag: a tag-sized box posed in the camera frame.

    Tags must come from a pose-estimating detect (pose_R/pose_t set). Frame names
    are prefixed with the service name so several detectors can share a WorldState.
    """
    detections: List[Detection3D] = []
    for tag in tags:
        confidence = _tag_confidence(tag)
        if confidence < confidence_threshold_pct:
            continue
        label = f"tag-{tag.tag_id}"
        detections.append(
            Detection3D(
                transforms=[
                    Transform(
                        reference_frame=f"{name_prefix}/{label}",
                        pose_in_observer_frame=PoseInFrame(reference_frame=camera_name, pose=_tag_pose(tag)),
                        # center left unset: the box sits at the transform's origin.
                        physical_object=Geometry(
                            box=RectangularPrism(dims_mm=Vector3(x=tag_width_mm, y=tag_width_mm, z=1)),
                            label=label,
                        ),
                    )
                ],
                classifications=[Classification(class_name=str(tag.tag_id), confidence=confidence)],
            )
        )
    return detections


class ApriltagModule(Module):
    """Module wrapper that resolves dependencies without a full parent refresh.

    The default Module._get_resource calls parent.refresh(), which tries to
    remove every resource that was disabled in the machine config. On older
    viam-sdk versions that can KeyError for resources this module never cached
    (for example an unrelated disabled gripper), causing apriltag reconfigure to
    fail even though it only depends on its camera.
    """

    async def _get_resource(self, name: ResourceName) -> ResourceBase:
        await self._connect_to_parent()
        assert self.parent is not None

        if name.type == RESOURCE_TYPE_COMPONENT:
            getter = self.parent.get_component
        elif name.type == RESOURCE_TYPE_SERVICE:
            getter = self.parent.get_service
        else:
            raise ValueError("Dependency does not describe a component nor a service")

        try:
            return getter(name)
        except ResourceNotFoundError:
            await self.parent._create_or_reset_client(name)
            return getter(name)


class Apriltag(PoseTracker, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily("marcus-org", "apriltag"), "pose_tracker")

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        instance = super().new(config, dependencies)
        instance.reconfigure(config, dependencies)
        return instance

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Tuple[Sequence[str], Sequence[str]]:
        attrs = struct_to_dict(config.attributes)
        cam = attrs.get(cam_attr)
        if cam is None:
            raise Exception("Missing required " + cam_attr + " attribute.")
        if attrs.get(family_attr) is None:
            raise Exception("Missing requried " + family_attr + " attribute.")
        if attrs.get(width_attr) is None:
            raise Exception("Missing requried " + width_attr + " attribute.")
        return [str(cam)], []

    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]):
        attrs = struct_to_dict(config.attributes)
        cam_name = str(attrs.get(cam_attr))
        self.camera = cast(Camera, dependencies[Camera.get_resource_name(cam_name)])
        self.tag_family = attrs.get(family_attr)
        self.tag_width_mm = attrs.get(width_attr)
        self.detector = apriltag.apriltag(self.tag_family, decimate=2.0)

    async def get_poses(
        self,
        body_names: List[str],
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Dict[str, PoseInFrame]:
        """This method returns the poses of the requested Apriltag IDs. 
        If no body names are requested, all detected Apriltags are returned

        Args:
            body_names (List[str]): A list of Apriltag IDs to return

        Returns:
            Dict[str, PoseInFrame]: A dictionary mapping Apriltag ID strings to their detected PoseInFrame
        """
        intrinsics = await _camera_intrinsics(self.camera, timeout)
        cam_images, _ = await self.camera.get_images(timeout=timeout)
        gray_image, _, _ = _gray_from_viam_image(_color_image_from_camera_images(cam_images))

        tags = _detect_apriltags(
            self.detector,
            gray_image,
            estimate_tag_pose=True,
            camera_params=intrinsics,
            tag_size=0.001 * self.tag_width_mm,
        )

        return {
            str(tag.tag_id): PoseInFrame(reference_frame=self.camera.name, pose=_tag_pose(tag))
            for tag in tags
            if len(body_names) == 0 or str(tag.tag_id) in body_names
        }

    async def get_geometries(self, *, extra: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None) -> List[Geometry]:
        raise NotImplementedError()

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Mapping[str, ValueTypes]:
        cmd = dict(command)
        if "get_poses" in cmd:
            body_names = cmd["get_poses"] if isinstance(cmd["get_poses"], list) else []
            poses = await self.get_poses(body_names, timeout=timeout)
            return {
                tag_id: {
                    "x": p.pose.x,
                    "y": p.pose.y,
                    "z": p.pose.z,
                    "o_x": p.pose.o_x,
                    "o_y": p.pose.o_y,
                    "o_z": p.pose.o_z,
                    "theta": p.pose.theta,
                    "reference_frame": p.reference_frame,
                }
                for tag_id, p in poses.items()
            }
        raise NotImplementedError(f"unknown command: {list(cmd.keys())}")


class ApriltagCamera(Camera, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily("marcus-org", "apriltag"), "camera")

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        instance = super().new(config, dependencies)
        instance.reconfigure(config, dependencies)
        return instance

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Tuple[Sequence[str], Sequence[str]]:
        attrs = struct_to_dict(config.attributes)
        cam = attrs.get(cam_attr)
        if cam is None:
            raise Exception("Missing required " + cam_attr + " attribute.")
        if attrs.get(family_attr) is None:
            raise Exception("Missing required " + family_attr + " attribute.")
        if attrs.get(width_attr) is None:
            raise Exception("Missing required " + width_attr + " attribute.")
        return [str(cam)], []

    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]):
        attrs = struct_to_dict(config.attributes)
        cam_name = str(attrs.get(cam_attr))
        self.camera = cast(Camera, dependencies[Camera.get_resource_name(cam_name)])
        self.tag_family = attrs.get(family_attr)
        self.tag_width_mm = attrs.get(width_attr)
        self.detector = apriltag.apriltag(self.tag_family, decimate=2.0)

    async def get_images(
        self,
        *,
        filter_source_names: Optional[Sequence[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Tuple[Sequence[NamedImage], ResponseMetadata]:
        try:
            cam_images, metadata = await self.camera.get_images(timeout=timeout)
        except Exception as e:
            LOGGER.error("ApriltagCamera.get_images: failed to get images from source camera: %s", e)
            raise

        source = _color_image_from_camera_images(cam_images)

        pil_img = viam_to_pil_image(source)
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        tags = _detect_apriltags(self.detector, gray)

        for tag in tags:
            corners = np.ascontiguousarray(tag.corners, dtype=np.int32)
            cv2.polylines(bgr, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
            center = (int(tag.center[0]), int(tag.center[1]))
            cv2.circle(bgr, center, 5, (0, 0, 255), -1)
            cv2.putText(bgr, f"ID:{tag.tag_id}", (center[0] + 8, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        _, jpeg_bytes = cv2.imencode(".jpg", bgr)
        named = NamedImage(name=self.name, data=jpeg_bytes.tobytes(), mime_type=CameraMimeType.JPEG)
        return [named], metadata

    async def get_properties(self, *, extra=None, timeout=None, **kwargs):
        return await self.camera.get_properties()

    async def get_point_cloud(self, *, extra=None, timeout=None, **kwargs):
        raise NotImplementedError()

    async def get_geometries(self, *, extra=None, timeout=None):
        raise NotImplementedError()


class ApriltagVision(Vision, EasyResource):
    """2D AprilTag detector that also returns 3D tag detections.

    Implements the Vision service detection API. Each detected tag becomes a
    Detection whose class_name is the tag ID. Pair this with
    viam:vision:detections-to-segments (which calls get_detections) to produce
    3D point-cloud segments from a depth camera. When tag_width_mm is
    configured, get_detections_3d returns each tag's pose as a Detection3D.
    """

    MODEL: ClassVar[Model] = Model(ModelFamily("marcus-org", "apriltag"), "vision")

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        instance = super().new(config, dependencies)
        instance.reconfigure(config, dependencies)
        return instance

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Tuple[Sequence[str], Sequence[str]]:
        attrs = struct_to_dict(config.attributes)
        cam = attrs.get(cam_attr)
        if cam is None:
            raise Exception("Missing required " + cam_attr + " attribute.")
        if attrs.get(family_attr) is None:
            raise Exception("Missing required " + family_attr + " attribute.")
        _parse_confidence_threshold(attrs)
        _parse_optional_int(attrs, bbox_padding_attr, 0)
        _parse_optional_tag_width(attrs)
        return [str(cam)], []

    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]):
        attrs = struct_to_dict(config.attributes)
        cam_name = str(attrs.get(cam_attr))
        self.camera = cast(Camera, dependencies[Camera.get_resource_name(cam_name)])
        self.tag_family = attrs.get(family_attr)
        self.confidence_threshold_pct = _parse_confidence_threshold(attrs)
        self.bbox_padding_px = _parse_optional_int(attrs, bbox_padding_attr, 0)
        self.tag_width_mm = _parse_optional_tag_width(attrs)
        self.detector = apriltag.apriltag(self.tag_family, decimate=2.0)

    async def _detections_from_camera(self, timeout: Optional[float]) -> List[Detection]:
        cam_images, _ = await self.camera.get_images(timeout=timeout)
        source = _color_image_from_camera_images(cam_images)
        gray, width, height = _gray_from_viam_image(source)
        tags = _detect_apriltags(self.detector, gray)
        return _tags_to_detections(
            tags,
            width,
            height,
            confidence_threshold_pct=self.confidence_threshold_pct,
            bbox_padding_px=self.bbox_padding_px,
        )

    async def _detections_3d_from_image(self, source: NamedImage, timeout: Optional[float]) -> List[Detection3D]:
        # Callers check tag_width_mm is set.
        intrinsics = await _camera_intrinsics(self.camera, timeout)
        gray, _, _ = _gray_from_viam_image(source)
        tags = _detect_apriltags(
            self.detector,
            gray,
            estimate_tag_pose=True,
            camera_params=intrinsics,
            tag_size=self.tag_width_mm / 1000,
        )
        return _tags_to_detections_3d(
            tags,
            self.name,
            self.camera.name,
            self.tag_width_mm,
            confidence_threshold_pct=self.confidence_threshold_pct,
        )

    async def get_properties(
        self,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> GetPropertiesResponse:
        return GetPropertiesResponse(
            classifications_supported=False,
            detections_supported=True,
            object_point_clouds_supported=False,
            detections_3d_supported=self.tag_width_mm is not None,
            default_camera=self.camera.name,
        )

    async def get_detections_from_camera(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> List[Detection]:
        return await self._detections_from_camera(timeout)

    async def get_detections(
        self,
        image: ViamImage,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> List[Detection]:
        gray, width, height = _gray_from_viam_image(image)
        tags = _detect_apriltags(self.detector, gray)
        return _tags_to_detections(
            tags,
            width,
            height,
            confidence_threshold_pct=self.confidence_threshold_pct,
            bbox_padding_px=self.bbox_padding_px,
        )

    async def get_classifications_from_camera(
        self,
        camera_name: str,
        count: int,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> List[Classification]:
        raise NotImplementedError()

    async def get_classifications(
        self,
        image: ViamImage,
        count: int,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> List[Classification]:
        raise NotImplementedError()

    async def get_object_point_clouds(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> List[PointCloudObject]:
        raise NotImplementedError()

    async def get_detections_3d(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> List[Detection3D]:
        # Like the other *_from_camera methods, camera_name is ignored in favor of the configured camera.
        if self.tag_width_mm is None:
            raise Exception(width_attr + " must be set in the vision service config to use get_detections_3d")
        cam_images, _ = await self.camera.get_images(timeout=timeout)
        return await self._detections_3d_from_image(_color_image_from_camera_images(cam_images), timeout)

    async def capture_all_from_camera(
        self,
        camera_name: str,
        return_image: bool = False,
        return_classifications: bool = False,
        return_detections: bool = False,
        return_object_point_clouds: bool = False,
        return_detections_3d: bool = False,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> CaptureAllResult:
        image: Optional[ViamImage] = None
        detections: Optional[List[Detection]] = None
        detections_3d: Optional[List[Detection3D]] = None
        want_3d = return_detections_3d and self.tag_width_mm is not None

        # Only touch the camera once for the image, 2D and 3D detections.
        if return_image or return_detections or want_3d:
            cam_images, _ = await self.camera.get_images(timeout=timeout)
            source = _color_image_from_camera_images(cam_images)
            if return_image:
                image = ViamImage(source.data, source.mime_type)
            if return_detections:
                gray, width, height = _gray_from_viam_image(source)
                tags = _detect_apriltags(self.detector, gray)
                detections = _tags_to_detections(
                    tags,
                    width,
                    height,
                    confidence_threshold_pct=self.confidence_threshold_pct,
                    bbox_padding_px=self.bbox_padding_px,
                )
            if want_3d:
                detections_3d = await self._detections_3d_from_image(source, timeout)

        # Unsupported features return None rather than raising, so the combined
        # Control tab view (which requests everything) stays healthy.
        return CaptureAllResult(image=image, detections=detections, detections_3d=detections_3d)


async def run_module():
    module = ApriltagModule.from_args()
    for key in Registry.REGISTERED_RESOURCE_CREATORS().keys():
        module.add_model_from_registry(*key.split("/"))  # pyright: ignore [reportArgumentType]
    await module.start()


if __name__ == "__main__":
    asyncio.run(run_module())
