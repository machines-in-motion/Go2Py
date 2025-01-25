from cyclonedds.topic import Topic
from cyclonedds.pub import DataWriter

from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.sub import DataReader
from cyclonedds.util import duration
import numpy as np

from Go2DDSMsgs import PointCloud, RGBImage, DepthImage, Pose, HighCommand
# from scipy.spatial.transform import Rotation as R

participant = DomainParticipant()
topic = Topic(participant, 'test_topic', RGBImage)
writer = DataWriter(participant, topic)

# domain = DomainParticipant()
# rgb_topic = Topic(domain, 'go2_rgb', RGBImage)
# rgb_publisher = Publisher(domain)
# rgb_writer = DataWriter(rgb_publisher, rgb_topic)


rgb = np.random.randint(0, 255, (10, 10, 3)).astype(np.uint8)

rgb_msg = RGBImage(data=[0, 1, 2], timestamp='')
writer.write(rgb_msg)