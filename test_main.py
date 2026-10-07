"""Plain-assert checks for src/main.py. Run: .venv/bin/python -m test_main"""
import math
import sys
from types import SimpleNamespace

import numpy as np

# dt-apriltags has no macOS wheel; stub it when absent. The helpers under test never call it.
try:
    import dt_apriltags  # noqa: F401
except ImportError:
    sys.modules["dt_apriltags"] = SimpleNamespace(Detector=None)

from src.main import _tag_pose


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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
