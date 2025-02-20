from Go2Py.robot.interface import GO2Real
import time
import numpy as np

from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore
from rosbags.typesys.stores.ros2_foxy import (
    builtin_interfaces__msg__Time as Time,
    sensor_msgs__msg__Imu as Imu,
    std_msgs__msg__Header as Header,
    geometry_msgs__msg__Vector3 as Vector3,
    geometry_msgs__msg__Quaternion as Quaternion,
)


def main():
    robot = GO2Real(mode='lowlevel')
    time.sleep(1)
    soc = robot.getBatteryState()
    imu = robot.getIMU()
    accel = imu['accel']
    gyro = imu['gyro']
    """Iterate over IMAGES and save to output bag."""
    typestore = get_typestore(Stores.ROS2_FOXY)
    with Writer('imu_noise_identification_dataset') as writer:
        conn = writer.add_connection('/go2/imu', Imu.__msgtype__)

        while (soc > 10):
            soc = robot.getBatteryState()
            imu = robot.getIMU()
            a = imu['accel']
            g = imu['gyro']
            accel = Vector3(a[0], a[1], a[2])
            gyro = Vector3(g[0], g[1], g[2])
            q = Quaternion(0, 0, 0, 1)
            timestamp = time.time() * 10**9
            msg = Imu(
                Header(
                    stamp=Time(sec=int(timestamp // 10**9), nanosec=int(timestamp % 10**9)),
                    frame_id='base_link'),
                linear_acceleration=accel,
                angular_velocity=gyro,
                orientation=q,
                orientation_covariance=np.zeros(9),
                linear_acceleration_covariance=np.zeros(9),
                angular_velocity_covariance=np.zeros(9)
            )
            writer.write(conn, timestamp, typestore.serialize_cdr(msg, msg.__msgtype__))
            time.sleep(0.02)
            print(soc)
    robot.close()
    exit()


if __name__ == '__main__':
    main()
