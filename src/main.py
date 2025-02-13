import asyncio
from typing import (Any, ClassVar, Dict, List, Mapping, Optional, Sequence, cast)

from typing_extensions import Self
from viam.components.pose_tracker import PoseTracker
from viam.components.camera import Camera
from viam.module.module import Module
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Geometry, PoseInFrame, ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.utils import struct_to_dict, ValueTypes
from viam.media.utils.pil import viam_to_pil_image

from PIL import Image
import dt_apriltags as apriltag
import numpy as np
import cv2

# required attributes
cam_attr = "camera_name"

class Apriltag(PoseTracker, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily("luddite", "apriltag"), "tracker")

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        """This method creates a new instance of this PoseTracker component.
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
        cam = struct_to_dict(config.attributes).get(cam_attr)
        if cam is None:
            raise Exception("Missing required " + cam_attr + " attribute.")
        return [str(cam)]

    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]):
        """This method allows you to dynamically update your service when it receives a new `config` object.

        Args:
            config (ComponentConfig): The new configuration
            dependencies (Mapping[ResourceName, ResourceBase]): Any dependencies (both implicit and explicit)
        """
        cam = struct_to_dict(config.attributes).get(cam_attr)
        self.camera = cast(Camera, dependencies.get(Camera.get_resource_name(str(cam))))
        return super().reconfigure(config, dependencies)

    async def get_poses(
        self,
        body_names: List[str],
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Dict[str, PoseInFrame]:
        raise NotImplementedError()

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Mapping[str, ValueTypes]:
        try:
            cam_image = await self.camera.get_image(mime_type="image/jpeg")

            # convert ViamImage to OpenCV format
            image_pil = viam_to_pil_image(cam_image)
            image_cv = np.array(image_pil)
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)  # Convert to grayscale for apriltag

            # initialize AprilTag detector - can include multiple families of tags in comma separated string
            detector = apriltag.Detector(families="tag16h5")
            tags = detector.detect(image_cv)
            print(tags)
        except Exception as e:
            raise e

    async def get_geometries(self, *, extra: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None) -> List[Geometry]:
        raise NotImplementedError()


if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())

