# [`apriltag` module](https://app.viam.com/module/luddite/apriltag) 

A Viam module that uses apriltags as an implementation for a PoseTracker component.

## Configuration and Usage

Navigate to the [**CONFIGURE** tab](https://docs.viam.com/build/configure/) of your [machine](https://docs.viam.com/fleet/machines/) in [the Viam app](https://app.viam.com/).
[Add `pose_tracker / apriltag` to your machine](https://docs.viam.com/build/configure/#components).

On the new component panel, copy and paste the following attribute template into your camera’s attributes field, editing the attributes as applicable:

```json
{
  "camera_name": "camera-1",
  "tag_family": "tag36h11",
  "tag_width_mm": 29.5
}
```

> [!NOTE]  
> For more information, see [Configure a Machine](https://docs.viam.com/manage/configuration/).

### Attributes

The following attributes are available for `viam:universal-robots` arms:

| Name | Type | Inclusion | Description |
| ---- | ---- | --------- | ----------- |
| `camera_name` | string | **Required** | The name of the camera to depend on. |
| `tag_family` | string | **Required** | The Apriltag 'tag family' to detect. |
| `tag_width_mm` | float | **Required** | The width of the tags to be detected, specified in mm. |
