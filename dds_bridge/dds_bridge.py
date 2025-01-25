import sys
sys.path.append('..')

from Go2Py.utils.ros2 import PoindCloud2Bridge, ros2_init, ros2_close, ROS2ExecutorManager, ROS2CameraReader
from Go2Py.utils.ros2 import ROS2OdometryReader
from Go2Py.robot.interface import GO2Real
from dds_telemetry import Go2DDSServer
import time
import numpy as np
import time
import cv2

import pickle
RECORD_DATASET = False
DATASET_LENGTH = 120
ros2_init()
map_bridge = PoindCloud2Bridge('/Laser_map')
fast_lio_odom = ROS2OdometryReader('/Odometry', 'lidar_odometry_subscriber')
go2_odom = ROS2OdometryReader('/go2/odom', 'leg_odometry_subscriber')
rgb_camera_bridge = ROS2CameraReader('/camera/color/image_raw', 'rgb_reader')
depth_camera_bridge = ROS2CameraReader('/camera/depth/image_rect_raw', 'depth_reader')

ros2_exec_manager = ROS2ExecutorManager()
ros2_exec_manager.add_node(map_bridge)
ros2_exec_manager.add_node(fast_lio_odom)
ros2_exec_manager.add_node(rgb_camera_bridge)
ros2_exec_manager.add_node(depth_camera_bridge)
ros2_exec_manager.add_node(go2_odom)
ros2_exec_manager.start()

# The extrinsics of the MID360 LiDAR with respect to the robot body
body_T_lidar = np.array([
    [ 0.9743701,  0.0,       0.2249511,  0.1870 ],
    [ 0.0,        1.0,       0.0,        0.0    ],
    [-0.2249511,  0.0,       0.9743701,  0.0803 ],
    [ 0.0,        0.0,       0.0,        1.0    ]
])

robot = GO2Real(mode='highlevel')
dds_server = Go2DDSServer(robot_name='go2')

def GetLidarPose():
    lio_T_lidar = fast_lio_odom.get_pose()
    if lio_T_lidar is not None:
        lio_T_body = lio_T_lidar @ np.linalg.inv(body_T_lidar)
        return lio_T_body
    else:
        return None
    
def GetLegOdom():
    odom = go2_odom.get_pose()
    if odom is not None:
        return odom
    else:
        return None

def GetLidarMap():
    pcd = map_bridge._points
    if pcd is not None:
        return pcd
    else:
        return None
    
print('Waitng for the ROS2 bridges to start up ...')
time.sleep(5)
print('Running the bridge loop')
lio_pcd = None
map_update_counter = 0

rgb_imgs = []
depth_imgs = []
lio_Ts_robot = []
odom_Ts_robot = []
pcd_maps = []



start_time = time.time()

while True:
    tic = time.time()
    lio_T_robot = GetLidarPose()
    odom_T_robot = GetLegOdom()
    if lio_T_robot is not None:
        dds_server.sendLidarOdom(lio_T_robot)
        if RECORD_DATASET:
            lio_Ts_robot.append(lio_T_robot)
    if odom_T_robot is not None:
        dds_server.sendLegOdom(odom_T_robot)
        if RECORD_DATASET:
            odom_Ts_robot.append(odom_T_robot)

    if map_update_counter%10==0:
        lio_pcd = GetLidarMap()
        if lio_pcd is not None:
            dds_server.sendMap(lio_pcd)
            if RECORD_DATASET:
                    pcd_maps.append(lio_pcd.copy())
    
    map_update_counter += 1

    depth = depth_camera_bridge.get_image()
    rgb = rgb_camera_bridge.get_image()
    if depth is not None and rgb is not None:
        depth = cv2.resize(depth, (320, 240))
        rgb = cv2.resize(rgb, (320, 240))
        # print('sending image')
        dds_server.sendDepth(depth)
        dds_server.sendRGB(rgb)
    
        if RECORD_DATASET:
            rgb_imgs.append(rgb.copy())
            depth_imgs.append(depth.copy())
        

    if time.time()-start_time >= DATASET_LENGTH:
        break

    # Forward the rgb, depth, lio, lio_T_lidar to the computer through DDS
    toc = time.time()
    command = dds_server.getHighCmd()
    if command is not None:
        print('Recievied a command from the client. Forwarding to the robot...')
        robot.setCommandsHigh(command.vx, command.vy, command.omega)

    print(f'{(toc-tic):0.3f}')
    while(time.time()-tic) < 0.1:
        time.sleep(0.001)


