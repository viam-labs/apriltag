import asyncio

from viam.robot.client import RobotClient
from viam.components.pose_tracker import PoseTracker

async def connect():
    opts = RobotClient.Options.with_api_key( 
        api_key='p4q8k9ams7e7gjnve2y58gja62bo9yxv',
        api_key_id='4cc4a83d-69e3-4881-a0d3-6d6cb49b6b72'
    )
    return await RobotClient.at_address('apriltag-main.2d8xnaiovq.viam.cloud', opts)

async def main():
    machine = await connect()

    print('Resources:')
    print(machine.resource_names)
    
    # pose_tracker
    pose_tracker = PoseTracker.from_robot(machine, "pose_tracker")
    poses = await pose_tracker.get_poses(None)
    print(poses)

    # Don't forget to close the machine when you're done!
    await machine.close()

if __name__ == '__main__':
    asyncio.run(main())
