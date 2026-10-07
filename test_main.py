"""Plain-assert checks for src/main.py. Run: .venv/bin/python -m test_main"""
import math
from types import SimpleNamespace

import numpy as np

from src.main import ApriltagVision, _camera_intrinsics, _detect_apriltags, _parse_optional_tag_width, _tag_pose, _tags_to_detections_3d


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, abs_tol=1e-6)


def fake_tag(tag_id=7, pose_R=None, pose_t=(0.1, 0.2, 0.3), decision_margin=80.0):
    return SimpleNamespace(
        tag_id=tag_id,
        pose_R=np.eye(3) if pose_R is None else np.array(pose_R, dtype=float),
        pose_t=np.array(pose_t, dtype=float).reshape(3, 1),
        decision_margin=decision_margin,
        corners=np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float),
    )


def test_tag_pose_identity():
    p = _tag_pose(fake_tag())
    # meters -> mm
    assert close(p.x, 100) and close(p.y, 200) and close(p.z, 300), p
    assert close(p.o_x, 0) and close(p.o_y, 0) and close(p.o_z, 1) and close(p.theta, 0), p


def test_tag_pose_rotated_90_about_z():
    p = _tag_pose(fake_tag(pose_R=[[0, -1, 0], [1, 0, 0], [0, 0, 1]]))
    assert close(p.o_x, 0) and close(p.o_y, 0) and close(p.o_z, 1), p
    assert close(p.theta, 90), p  # degrees


def test_tags_to_detections_3d():
    good = fake_tag(tag_id=7, decision_margin=80.0)   # confidence 1.0
    weak = fake_tag(tag_id=9, decision_margin=4.0)    # confidence 0.1
    dets = _tags_to_detections_3d(
        [good, weak], "my-vision", "my-cam", 50.0, confidence_threshold_pct=0.5
    )
    assert len(dets) == 1, dets

    d = dets[0]
    assert len(d.transforms) == 1
    t = d.transforms[0]
    assert t.reference_frame == "my-vision/tag-7"
    assert t.pose_in_observer_frame.reference_frame == "my-cam"
    pose = t.pose_in_observer_frame.pose
    assert close(pose.x, 100) and close(pose.y, 200) and close(pose.z, 300), pose
    assert close(pose.o_z, 1) and close(pose.theta, 0), pose
    dims = t.physical_object.box.dims_mm
    assert (dims.x, dims.y, dims.z) == (50.0, 50.0, 1.0), dims
    assert t.physical_object.label == "tag-7"

    assert len(d.classifications) == 1
    assert d.classifications[0].class_name == "7"
    assert close(d.classifications[0].confidence, 1.0)


def test_vision_is_concrete():
    # SDK 0.84 made get_detections_3d abstract; a missing override breaks instantiation.
    assert not ApriltagVision.__abstractmethods__, ApriltagVision.__abstractmethods__


def test_parse_optional_tag_width():
    assert _parse_optional_tag_width({}) is None
    assert _parse_optional_tag_width({"tag_width_mm": 50}) == 50.0
    try:
        _parse_optional_tag_width({"tag_width_mm": True})  # bool is an int subclass
    except Exception:
        return
    raise AssertionError("accepted True")


def test_real_detector_on_rendered_tag():
    import apriltag
    import cv2

    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), 7, 200)
    gray = np.full((720, 1280), 255, dtype=np.uint8)
    gray[260:460, 540:740] = marker  # 200 px tag centered on the principal point
    detector = apriltag.apriltag("tag36h11", decimate=2.0)

    (tag,) = _detect_apriltags(detector, gray)
    assert tag.tag_id == 7 and tag.decision_margin > 0 and tag.corners.shape == (4, 2)

    # 0.1 m tag spanning 200 px at f=1000 px -> 0.5 m away
    (tag,) = _detect_apriltags(
        detector, gray, estimate_tag_pose=True, camera_params=[1000.0, 1000.0, 640.0, 360.0], tag_size=0.1
    )
    p = _tag_pose(tag)
    assert math.isclose(p.z, 500, abs_tol=10) and abs(p.x) < 10 and abs(p.y) < 10, p


def test_camera_intrinsics_rejects_missing():
    import asyncio
    from viam.proto.component.camera import IntrinsicParameters

    class Cam:
        name = "cam"

        def __init__(self, intr):
            self.intr = intr

        async def get_properties(self, timeout=None):
            return SimpleNamespace(intrinsic_parameters=self.intr)

    ok = IntrinsicParameters(focal_x_px=1000, focal_y_px=1000, center_x_px=640, center_y_px=360)
    assert asyncio.run(_camera_intrinsics(Cam(ok), None)) == [1000, 1000, 640, 360]
    try:
        asyncio.run(_camera_intrinsics(Cam(IntrinsicParameters()), None))
    except Exception as e:
        assert "intrinsic" in str(e)
        return
    raise AssertionError("missing intrinsics accepted")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
