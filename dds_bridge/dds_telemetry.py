import socket
import struct
import numpy as np
import time
import threading
from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.sub import DataReader
from cyclonedds.pub import Publisher, DataWriter
from dataclasses import dataclass
from Go2DDSMsgs import PointCloud, RGBImage, DepthImage, Pose, HighCommand
from scipy.spatial.transform import Rotation as R


def get_last_msg(reader, topic_type):
    """ """
    last_msg = reader.take()

    if last_msg:
        while True:
            a = reader.take()
            if not a:
                break
            else:
                last_msg = a
    if last_msg:
        msg = last_msg[0]
        if type(msg) == topic_type:
            return msg
        else:
            return None

    else:
        return None
    
class Go2DDSServer:
    def __init__(self, robot_name):
        self.robot_name = robot_name
        self.domain = DomainParticipant()
        self.rgb_topic = Topic(self.domain, f'{self.robot_name}_rgb', RGBImage)
        self.depth_topic = Topic(self.domain, f'{self.robot_name}_depth', DepthImage)  
        self.map_topic = Topic(self.domain, f'{self.robot_name}_lio_map', PointCloud)
        self.lidar_odom_topic = Topic(self.domain, f'{self.robot_name}_lio_odom', Pose)
        self.high_cmd_topic = Topic(self.domain, f'{self.robot_name}_high_cmd', HighCommand)
        self.rgb_publisher = Publisher(self.domain)
        self.depth_publisher = Publisher(self.domain)
        self.map_publisher = Publisher(self.domain)
        self.lidar_odom_publisher = Publisher(self.domain)
        self.high_cmd_reader = DataReader(self.domain, self.high_cmd_topic)
        self.rgb_writer = DataWriter(self.rgb_publisher, self.rgb_topic)
        self.depth_writer = DataWriter(self.depth_publisher, self.depth_topic)
        self.map_writer = DataWriter(self.map_publisher, self.map_topic)
        self.lidar_odom_writer = DataWriter(self.lidar_odom_publisher, self.lidar_odom_topic)
        
        
        

    def sendRGB(self, rgb):
        self.rgb_writer.write(RGBImage(data=rgb.reshape(-1).tolist(), width=rgb.shape[1], height=rgb.shape[0], timestamp=''))
    
    def sendDepth(self, depth):
        self.depth_writer.write(DepthImage(data=depth.reshape(-1).tolist(), width=depth.shape[1], height=depth.shape[0], timestamp=''))

    def sendMap(self, pcd):
        map_pcd_mgs = PointCloud(x = pcd[:,0].tolist(), 
                                 y = pcd[:,1].tolist(), 
                                 z = pcd[:,2].tolist(), 
                                 timestamp=''
                                )
        self.map_writer.write(map_pcd_mgs)
    
    def sendLidarOdom(self, lio_T_body):
        q = R.from_matrix(lio_T_body[:3,:3]).as_quat()
        t = lio_T_body[:3,3]
        pose_msg = Pose(quat=q.tolist(), trans=t.tolist(), timestamp='')
        self.lidar_odom_writer.write(pose_msg)

    def getHighCmd(self):
        return get_last_msg(self.high_cmd_reader, HighCommand)    
    

class Go2DDSClient:
    def __init__(self, robot_name):
        self.robot_name = robot_name
        self.domain = DomainParticipant()
        self.rgb_topic = Topic(self.domain, f'{self.robot_name}_rgb', RGBImage)
        self.depth_topic = Topic(self.domain, f'{self.robot_name}_depth', DepthImage)  
        self.map_topic = Topic(self.domain, f'{self.robot_name}_lio_map', PointCloud)
        self.lidar_odom_topic = Topic(self.domain, f'{self.robot_name}_lio_odom', Pose)
        self.high_cmd_topic = Topic(self.domain, f'{self.robot_name}_high_cmd', HighCommand)
        self.rgb_reader = DataReader(self.domain, self.rgb_topic)
        self.depth_reader = DataReader(self.domain, self.depth_topic)
        self.map_reader = DataReader(self.domain, self.map_topic)
        self.lidar_odom_reader = DataReader(self.domain, self.lidar_odom_topic)
        self.high_cmd_writer = DataWriter(self.domain, self.high_cmd_topic)

    def getRGB(self):
        return get_last_msg(self.rgb_reader, RGBImage)
    
    def getDepth(self):
        return get_last_msg(self.depth_reader, DepthImage)
    
    def getMap(self):
        return get_last_msg(self.map_reader, PointCloud)
    
    def getLidarOdom(self):
        return get_last_msg(self.lidar_odom_reader, Pose)
    
    def sendHighCmd(self, vx, vy, omega):
        self.high_cmd_writer.write(HighCommand(vx=vx, vy=vy, omega=omega, timestamp=''))
        