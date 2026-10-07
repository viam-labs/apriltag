# `apriltag` module

A Viam module that uses apriltags as an implementation for a PoseTracker component, an annotated camera, and a vision detector (2D detections, plus 3D detections when `tag_width_mm` is set).

Fork of [viam-labs/apriltag](https://github.com/viam-labs/apriltag) with fixes for module reconfigure when unrelated machine resources are disabled.

**Module:** [marcus-org/apriltag](https://app.viam.com/module/marcus-org/apriltag)

## Models

| Model | API | Purpose |
| ----- | --- | ------- |
| `marcus-org:apriltag:pose_tracker` | pose tracker | 6-DOF tag poses via PnP |
| `marcus-org:apriltag:camera` | camera | Annotated debug feed |
| `marcus-org:apriltag:vision` | vision | 2D tag detections, native 3D detections when `tag_width_mm` is set, or use with `detections-to-segments` |

## Configuration and Usage

Navigate to the [**CONFIGURE** tab](https://docs.viam.com/build/configure/) of your [machine](https://docs.viam.com/fleet/machines/) in [the Viam app](https://app.viam.com/).

### Pose tracker

[Add a pose tracker component](https://docs.viam.com/build/configure/#components) using model **`marcus-org:apriltag:pose_tracker`**.

```json
{
    "camera_name": "crop-camera",
  "tag_family": "tag36h11",
  "tag_width_mm": 29.5
}
```

### Vision detector + detections-to-segments

Use **`marcus-org:apriltag:vision`** for native 3D detections (set `tag_width_mm`), or wire it into **`viam:vision:detections-to-segments`** as an alternative point-cloud route that works without `tag_width_mm`.

`marcus-org:apriltag:vision` returns 2D detections and, when `tag_width_mm` is set, 3D detections via `GetDetections3D` (one tag-sized box per tag, posed in the camera frame, with frame name `<service name>/tag-<id>`, usable directly in a motion `WorldState`). It implements the Vision service detection API (`GetDetections`, `GetDetectionsFromCamera`, `GetDetections3D`, `CaptureAllFromCamera`, `GetProperties`) and reports `detections_supported = true`, `detections_3d_supported` (true when `tag_width_mm` is set), `classifications_supported = false`, `object_point_clouds_supported = false`. Alternatively, feed it into `viam:vision:detections-to-segments`, which calls `GetDetections` and projects the boxes onto the depth camera's point cloud. Because AprilTags are small, set `bbox_padding_px` to grow each tag bbox so the segmenter has enough depth points to work with (this padding applies consistently to every detection path).

**1. Detector** (`marcus-org:apriltag:vision`):

```json
{
  "name": "apriltag-detector",
  "api": "rdk:service:vision",
  "model": "marcus-org:apriltag:vision",
  "attributes": {
    "camera_name": "crop-camera",
    "tag_family": "tag36h11",
    "confidence_threshold_pct": 0.0,
    "bbox_padding_px": 40
  }
}
```

> [!IMPORTANT]  
> Set the detector's `camera_name` to the **same camera** the segmenter uses (`crop-camera` here). `detections-to-segments` calls `GetDetections` with an image from its own `camera_name`, so the detector and segmenter must agree on the source camera for the 2D boxes to line up with the point cloud.

**2. Segmenter** (`viam:vision:detections-to-segments`):

```json
{
  "name": "apriltag-segment",
  "api": "rdk:service:vision",
  "model": "viam:vision:detections-to-segments",
  "attributes": {
    "detector_name": "apriltag-detector",
    "camera_name": "crop-camera",
    "confidence_threshold_pct": 0.1,
    "mean_k": 1,
    "sigma": 2.5,
    "infer_minimum_depth": true
  }
}
```

```python
vision = VisionClient.from_robot(robot, "apriltag-segment")
objects = await vision.get_object_point_clouds("crop-camera")
```

Each object's `geometries.geometries[0].label` is the tag ID string; `center` is the 3D point in the camera frame.

> [!NOTE]  
> For more information, see [Configure a Machine](https://docs.viam.com/manage/configuration/).

### Attributes

| Name | Type | Inclusion | Description |
| ---- | ---- | --------- | ----------- |
| `camera_name` | string | **Required** | The name of the camera to depend on. |
| `tag_family` | string | **Required** | The Apriltag 'tag family' to detect. |
| `tag_width_mm` | float | Required for pose tracker / camera; optional for vision | Tag width in mm (corner to corner). For the vision model, optional and required for 3D detections. |
| `confidence_threshold_pct` | float | Optional (vision only) | Detections below this confidence are dropped. Range `0.0`–`1.0`. Default `0.0`. Confidence is `decision_margin / 40`, capped at `1.0`. |
| `bbox_padding_px` | int | Optional (vision only) | Pixels to expand each detection bbox on every side. Helps `detections-to-segments` capture enough depth points around small tags. Default `0`. |

> [!NOTE]
> `crop-camera` (or whichever depth camera you use) must support point clouds (`supports_pcd: true`).

### Generating Apriltags

To get quickly started tracking poses of Apriltags the example file `tag36h11_1-30.pdf` can be printed and used with the example configuration above. There exist a number of [online generators](https://shiqiliu-67.github.io/apriltag-generator/) that can be used to create similar files suitable to your specific needs.

For more information about the Apriltag specification and how they can be used see the [AprilRobotics repo](https://github.com/aprilrobotics/apriltag).

## Publishing a new module version

From the module root, with the [Viam CLI](https://docs.viam.com/cli/) authenticated:

```bash
# 1. Build the upload archive
make module.tar.gz

# 2. Upload for each platform your meta.json declares
viam module upload --version 0.3.0 --platform linux/amd64 --upload module.tar.gz
viam module upload --version 0.3.0 --platform linux/arm64 --upload module.tar.gz

# Alternative: push to a machine via cloud build. run.sh must NOT be executable on disk
# (the reload archives your working tree); `python -m test_main` checks this.
# viam module reload --part-id <your-part-id>

# 3. Optional: refresh module metadata on the registry
viam module update --module ./meta.json
```

Bump `--version` to a new semver for each release. After upload, update the module version on your machine's **CONFIGURE** tab to pick up the new release.
