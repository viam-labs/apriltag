import asyncio
import dt_apriltags as apriltag
import numpy as np
import cv2
import math

from scipy.spatial.transform import Rotation
from .spatialmath import quaternion_to_orientation_vector

from typing import (Any, ClassVar, Dict, List, Mapping, Optional, Sequence, cast)
from typing_extensions import Self

from viam.components.pose_tracker import PoseTracker
from viam.components.camera import Camera
from viam.media.video import CameraMimeType
from viam.module.module import Module
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Geometry, PoseInFrame, Pose, ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.logging import getLogger
from viam.utils import struct_to_dict, ValueTypes
from viam.media.utils.pil import viam_to_pil_image


# required attributes
cam_attr = "camera_name"
family_attr = "tag_family"
width_attr = "tag_width_mm"

LOGGER = getLogger(__name__)

class Apriltag(PoseTracker, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily("viam", "apriltag"), "pose_tracker")

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        """This method creates a new instance of the Apriltag PoseTracker component.
        The default implementation sets the name from the `config` parameter and then calls `reconfigure`.

        Args:
            config (ComponentConfig): The configuration for this resource
            dependencies (Mapping[ResourceName, ResourceBase]): The dependencies (both implicit and explicit)

        Returns:
            Self: The resource
        """
        return super().new(config, dependencies)

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Sequence[str]:
        """This method allows you to validate the configuration object received from the machine,
        as well as to return any implicit dependencies based on that `config`.

        Args:
            config (ComponentConfig): The configuration for this resource

        Returns:
            Sequence[str]: A list of implicit dependencies
        """
        attrs = struct_to_dict(config.attributes)
        cam = attrs.get(cam_attr)
        if cam is None:
            raise Exception("Missing required " + cam_attr + " attribute.")
        if attrs.get(family_attr) is None:
            raise Exception("Missing requried " + family_attr + " attribute.")
        if attrs.get(width_attr) is None:
            raise Exception("Missing requried " + width_attr + " attribute.")
        return [str(cam)]

    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]):
        """This method allows you to dynamically update your service when it receives a new `config` object.

        Args:
            config (ComponentConfig): The new configuration
            dependencies (Mapping[ResourceName, ResourceBase]): Any dependencies (both implicit and explicit)
        """
        attrs = struct_to_dict(config.attributes)
        self.camera = cast(Camera, dependencies.get(Camera.get_resource_name(str(attrs.get(cam_attr)))))
        self.tag_family = attrs.get(family_attr)
        self.tag_width_mm = attrs.get(width_attr)
        return super().reconfigure(config, dependencies)

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
        try:
            # need to get the camera intrinsics
            properties = await self.camera.get_properties()
            intrinsics = [
                properties.intrinsic_parameters.focal_x_px,
                properties.intrinsic_parameters.focal_y_px,
                properties.intrinsic_parameters.center_x_px,
                properties.intrinsic_parameters.center_y_px
            ]

            # get an image from camera resource and convert it to OpenCV format
            cam_images = await self.camera.get_images()
            for image in cam_images[0]:
                if image.mime_type == CameraMimeType.JPEG:
                    color_image = cv2.cvtColor(np.array(viam_to_pil_image(cam_images[0][0])), cv2.COLOR_RGB2GRAY)  # convert to grayscale

            # initialize AprilTag detector - can include multiple families of tags in comma separated string
            detector = apriltag.Detector(families=self.tag_family)
            tags = detector.detect(color_image, estimate_tag_pose=True, camera_params=intrinsics, tag_size=0.001*self.tag_width_mm)

            poses = {}
            for tag in tags:
                #
                if len(body_names) == 0 or str(tag.tag_id) in body_names:
                    o = quaternion_to_orientation_vector(Rotation.from_matrix(tag.pose_R))
                    # need to convert the returned positions from mm to m and the orientation's theta to degrees.
                    poses[str(tag.tag_id)] = PoseInFrame(
                        reference_frame=self.camera.name, 
                        pose=Pose(
                            x=tag.pose_t[0][0] * 1000,
                            y=tag.pose_t[1][0] * 1000,
                            z=tag.pose_t[2][0] * 1000, 
                            o_x=o.o_x,
                            o_y=o.o_y,
                            o_z=o.o_z,
                            theta=o.theta * 180 / math.pi
                        )
                    )        
            return poses
                
        except Exception as e:
            raise e
        
    async def get_geometries(self, *, extra: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None) -> List[Geometry]:
        raise NotImplementedError()

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Mapping[str, ValueTypes]:
        raise NotImplementedError()


if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())
