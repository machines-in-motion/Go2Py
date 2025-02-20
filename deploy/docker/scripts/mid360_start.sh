source /opt/ros/humble/setup.bash
source /unitree_ros2/cyclonedds_ws/install/setup.bash
# export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
# export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces>
#                             <NetworkInterface name="eth0" priority="default" multicast="default" />
#                         </Interfaces></General></Domain></CycloneDDS>'
source /root/ws_livox/install/setup.bash && ros2 launch livox_ros_driver2 msg_MID360_launch.py