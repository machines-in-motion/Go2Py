import struct
import threading
import time
import numpy as np
import rclpy
import tf2_ros
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
from Go2Py.utils.point_cloud2 import read_points_numpy
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

import numpy as np
from scipy.spatial.transform import Rotation as R

class ROS2OdometryReader(Node):
    """
    A class for subscribing to and retrieving Odometry messages.

    Args:
        odom_topic (str): The topic name for the odometry data.
        node_name (str): A unique name for this node.

    Attributes:
        odom (Odometry): The latest odometry message received from the topic.
    """

    def __init__(self, odom_topic, node_name):
        super().__init__(f'{node_name}_odom_reader')
        self.odom = None  # Stores the latest odometry message
        
        # Create a subscription to the specified odometry topic
        self.odom_subscriber = self.create_subscription(
            Odometry,
            odom_topic,
            self.odom_callback,
            10
        )

    def odom_callback(self, msg):
        """
        Callback function that stores the latest odometry message.
        
        Args:
            msg (Odometry): The odometry message received.
        """
        self.odom = msg

    def get_odometry(self):
        """
        Returns the latest odometry message received from the topic.

        Returns:
            Odometry: The latest odometry message.
        """
        return self.odom
    
    def get_pose(self):
        if self.odom is not None:
            position = self.odom.pose.pose.position
            orientation = self.odom.pose.pose.orientation
            t = np.array([position.x, position.y, position.z])
            q = np.array([orientation.x, orientation.y, orientation.z, orientation.w])
            Rot = R.from_quat(q).as_matrix()
            T = np.eye(4)
            T[:3, :3] = Rot
            T[:3, -1] = t
            return T
        else:
            return None

    def close(self):
        """
        Destroys the node, cleaning up resources.
        """
        self.destroy_node()

def ros2_init(args=None):
    rclpy.init(args=args)


def ros2_close():
    rclpy.shutdown()

class ROS2ExecutorManager:
    """A class to manage the ROS2 executor. It allows to add nodes and start the executor in a separate thread."""
    def __init__(self):
        self.executor = MultiThreadedExecutor()
        self.nodes = []
        self.executor_thread = None

    def add_node(self, node: Node):
        """Add a new node to the executor."""
        self.nodes.append(node)
        self.executor.add_node(node)

    def _run_executor(self):
        try:
            self.executor.spin()
        except KeyboardInterrupt:
            pass
        finally:
            self.terminate()

    def start(self):
        """Start spinning the nodes in a separate thread."""
        self.executor_thread = threading.Thread(target=self._run_executor)
        self.executor_thread.start()

    def terminate(self):
        """Terminate all nodes and shutdown rclpy."""
        for node in self.nodes:
            node.destroy_node()
        rclpy.shutdown()
        if self.executor_thread:
            self.executor_thread.join()

# class ROS2TFInterface(Node):

#     def __init__(self, parent_name, child_name, node_name):
#         super().__init__(f'{node_name}_tf2_listener')
#         self.parent_name = parent_name
#         self.child_name = child_name
#         self.tfBuffer = tf2_ros.Buffer()
#         self.listener = tf2_ros.TransformListener(self.tfBuffer, self)
#         self.T = None
#         self.stamp = None
#         self.running = True
#         self.thread = threading.Thread(target=self.update_loop)
#         self.thread.start()
#         self.trans = None

#     def update_loop(self):
#         while self.running:
#             try:
#                 self.trans = self.tfBuffer.lookup_transform(self.parent_name, self.child_name, rclpy.time.Time(), rclpy.time.Duration(seconds=0.1))
#             except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
#                 print(e)
#             time.sleep(0.01)    

#     def get_pose(self):
#         if self.trans is None:
#             return None
#         else:
#             translation = [self.trans.transform.translation.x, self.trans.transform.translation.y, self.trans.transform.translation.z]
#             rotation = [self.trans.transform.rotation.x, self.trans.transform.rotation.y, self.trans.transform.rotation.z, self.trans.transform.rotation.w]
#             self.T = np.eye(4)
#             self.T[0:3, 0:3] = R.from_quat(rotation).as_matrix()
#             self.T[:3, 3] = translation
#             self.stamp = self.trans.header.stamp.nanosec * 1e-9 + self.trans.header.stamp.sec
#             return self.T

#     def close(self):
#         self.running = False
#         self.thread.join()  
#         self.destroy_node()


class ROS2TFInterface(Node):
    def __init__(self, parent_name, child_name, node_name):
        super().__init__(f"{node_name}_tf2_listener")
        self.parent_name = parent_name
        self.child_name = child_name
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer, self)
        self.T = None
        self.stamp = None
        self.running = True
        self.trans = None

    def get_pose(self):
        try:
            self.trans = self.tfBuffer.lookup_transform(
                self.parent_name,
                self.child_name,
                rclpy.time.Time(),
                rclpy.time.Duration(seconds=0.1),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            return None

        translation = [
            self.trans.transform.translation.x,
            self.trans.transform.translation.y,
            self.trans.transform.translation.z,
        ]
        rotation = [
            self.trans.transform.rotation.x,
            self.trans.transform.rotation.y,
            self.trans.transform.rotation.z,
            self.trans.transform.rotation.w,
        ]
        self.T = np.eye(4)
        self.T[0:3, 0:3] = R.from_quat(rotation).as_matrix()
        self.T[:3, 3] = translation
        self.stamp = (
            self.trans.header.stamp.nanosec * 1e-9 + self.trans.header.stamp.sec
        )

        return self.T

    def close(self):
        self.running = False
        self.destroy_node()


class PoindCloud2Bridge(Node):
    def __init__(self, topic_name):
        super().__init__("point_cloud_listener")
        self.subscription = self.create_subscription(
            PointCloud2, topic_name, self.listener_callback, 1
        )
        self.data = None
        self.new_frame_flag = False
        self.points = None

        self._points = None
        self.points = None
        self.ready = False
        self.running = True

    def listener_callback(self, msg):
        # Read the x, y, z fields from the PointCloud2 message
        self._points = read_points_numpy(
            msg,
            field_names=("x", "y", "z"),
            skip_nans=True,
            reshape_organized_cloud=True,
        )
        # self._points = point_cloud2.read_points(
        #     msg,
        #     field_names=("x", "y", "z"),
        #     skip_nans=True,
        # )
        # _points = np.reshape(_points, (-1, 16, 3))
        # self._points = np.reshape(_points, (-1, 16, 3))
        self.new_frame_flag = True

    @property
    def state(self):
        return self._points

class ROS2CameraReader(Node):
    """
    A class for interacting with a ROS2 camera topics.

    Args:
        image_topic (str): The topic name for the image stream.
        camera_info_topic (str, optional): The topic name for the camera information.
        K (numpy.ndarray, optional): The intrinsic camera matrix for the raw (distorted) images.
        D (numpy.ndarray, optional): The distortion coefficients.

    Attributes:
        color_frame (numpy.ndarray): The latest image frame received from the topic.
        camera_info (CameraInfo): The latest camera information received from the topic.
        K (numpy.ndarray): The intrinsic camera matrix.
        D (numpy.ndarray): The distortion coefficients.
    """
    def __init__(self, image_topic, node_name, camera_info_topic=None, K=None, D=None):
        super().__init__(f'{node_name}_camera_reader')
        self.bridge = CvBridge()
        self.color_frame = None
        self.camera_info = None
        self.K = K
        self.D = D
        self.node_name = node_name

        self.image_subscriber = self.create_subscription(Image, image_topic, self.image_callback, 10)
        if camera_info_topic:
            self.camera_info_subscriber = self.create_subscription(CameraInfo, camera_info_topic, self.camera_info_callback, 10)

    def image_callback(self, msg):
        self.color_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def camera_info_callback(self, msg):
        self.camera_info = msg
        self.K = np.array(msg.k).reshape((3, 3))
        self.D = np.array(msg.d)

    def get_image(self):
        """
        Returns the latest image frame received from the topic.

        Returns:
            numpy.ndarray: The latest image frame received from the topic.
        """
        return self.color_frame 

    def get_intrinsics(self):
        """
        Returns the intrinsic camera matrix.

        Returns:
            numpy.ndarray: The intrinsic camera matrix.
        """
        return {'K': self.K, 'D': self.D}
    def close(self):
        self.destroy_node()