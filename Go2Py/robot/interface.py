import struct
import time
import numpy as np
import numpy.linalg as LA
from scipy.spatial.transform import Rotation as R

from cyclonedds.domain import DomainParticipant
from go2py_messages.msg.dds_ import Go2pyLowCmd_
from go2py_messages.msg.dds_ import Go2pyHighCmd_
from go2py_messages.msg.dds_ import Go2pyState_
from cyclonedds.topic import Topic
from cyclonedds.pub import DataWriter

from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.sub import DataReader
from cyclonedds.util import duration
from threading import Thread
from scipy.spatial.transform import Rotation
from Go2Py.joy import xKeySwitch, xRockerBtn
from Go2Py.utils import set_cyclonedds_config
from copy import deepcopy

class GO2Real():
    def __init__(
        self,
        interface_name=None,
        mode='lowlevel',  # 'highlevel' or 'lowlevel'
        vx_max=0.5,
        vy_max=0.4,
        ωz_max=0.5,
    ):
        assert mode in ['highlevel', 'lowlevel'], "mode should be either 'highlevel' or 'lowlevel'"
        if interface_name is not None:
            set_cyclonedds_config(interface_name)
            
        self.mode = mode
        self.simulated = False
        self.prestanding_q = np.array([0.0, 1.26186061, -2.5,
                                       0.0, 1.25883281, -2.5,
                                       0.0, 1.27193761, -2.6,
                                       0.0, 1.27148342, -2.6])

        self.sitting_q = np.array([-0.02495611, 1.26249647, -2.82826662,
                                   0.04563564, 1.2505368, -2.7933557,
                                   -0.30623949, 1.28283751, -2.82314873,
                                   0.26400229, 1.29355574, -2.84276843])

        self.standing_q = np.array([0.0, 0.77832842, -1.56065452,
                                    0.0, 0.76754963, -1.56634164,
                                    0.0, 0.76681757, -1.53601146,
                                    0.0, 0.75422204, -1.53229916])
        self.latest_command_stamp = time.time()
        self.highcmd_topic_name = "rt/go2/twist_cmd"
        self.lowcmd_topic_name = "rt/go2py/low_cmd"
        self.highcmd_topic_name = "rt/go2py/high_cmd"
        self.lowstate_topic_name = "rt/go2py/state"
        try:
            self.participant = DomainParticipant()
        except:
            raise Exception('Could not initialize the DDS communication. Is the interface name provided correctly?')
        
        self.lowstate_topic = Topic(self.participant, self.lowstate_topic_name, Go2pyState_)
        self.state_reader = DataReader(self.participant, self.lowstate_topic)

        self.lowcmd_topic = Topic(self.participant, self.lowcmd_topic_name, Go2pyLowCmd_)
        self.lowcmd_writer = DataWriter(self.participant, self.lowcmd_topic)

        self.highcmd_topic = Topic(self.participant, self.highcmd_topic_name, Go2pyHighCmd_)
        self.highcmd_writer = DataWriter(self.participant, self.highcmd_topic)
        self.vx_max = vx_max
        self.vy_max = vy_max
        self.P_v_max = np.diag([1 / self.vx_max**2, 1 / self.vy_max**2])
        self.ωz_max = ωz_max
        self.ωz_min = -ωz_max

        self.state = None
        self.setCommands = {'lowlevel': self.setCommandsLow,
                            'highlevel': self.setCommandsHigh}[self.mode]
        self.state = Go2pyState_
        self.state_thread = Thread(target=self.state_update)
        self.running = True
        self.latest_command = None
        self.state_thread.start()

    def state_update(self):
        """
        Retrieve the state of the robot
        """
        while self.running:
            for msg in self.state_reader.take_iter(timeout=duration(milliseconds=1.)):
                self.state = msg

    def getIMU(self):
        accel = self.state.accel
        gyro = self.state.gyro
        quat = self.state.quat
        temp = self.state.imu_temp
        return {'accel': np.array(accel), 'gyro': np.array(gyro), 'quat': np.array(quat), 'temp': temp}

    def getFootContacts(self):
        """Returns the raw foot contact forces"""
        footContacts = self.state.contact
        return footContacts

    def getJointStates(self):
        """Returns the joint angles (q) and velocities (dq) of the robot"""
        return {'q': self.state.q,
                'dq': self.state.dq,
                'tau_est': self.state.tau,
                'temperature': self.state.motor_temp}

    def getRemoteState(self):
        """A method to get the state of the wireless remote control.
        Returns a xRockerBtn object:
        - head: [head1, head2]
        - keySwitch: xKeySwitch object
        - lx: float
        - rx: float
        - ry: float
        - L2: float
        - ly: float
        """
        wirelessRemote = self.state.wireless_remote[:24]
        binary_data = bytes(wirelessRemote)

        format_str = "<2BH5f"
        data = struct.unpack(format_str, binary_data)

        head = list(data[:2])
        lx = data[3]
        rx = data[4]
        ry = data[5]
        L2 = data[6]
        ly = data[7]

        _btn = bin(data[2])[2:].zfill(16)
        btn = [int(char) for char in _btn]
        btn.reverse()

        keySwitch = xKeySwitch(*btn)
        rockerBtn = xRockerBtn(head, keySwitch, lx, rx, ry, L2, ly)
        return rockerBtn

    def getCommandFromRemote(self):
        """Do not use directly for control!!!"""
        rockerBtn = self.getRemoteState()

        lx = rockerBtn.lx
        ly = rockerBtn.ly
        rx = rockerBtn.rx

        v_x = ly * self.vx_max
        v_y = lx * self.vy_max
        ω = rx * self.ωz_max

        return v_x, v_y, ω

    def getBatteryState(self):
        """Returns the battery percentage of the robot"""
        return self.state.soc

    def setCommandsHigh(self, v_x, v_y, ω_z, bodyHeight=0.0, footRaiseHeight=0.0, mode=2):
        self.cmd_watchdog_timer = time.time()
        _v_x, _v_y, _ω_z = self.clip_velocity(v_x, v_y, ω_z)
        highcmd = Go2pyHighCmd_(
            _v_x,
            _v_y,
            _ω_z
        )
        self.highcmd_writer.write(highcmd)

    def setCommandsLow(self, q_des, dq_des, kp, kd, tau_ff):
        assert q_des.size == dq_des.size == kp.size == kd.size == tau_ff.size == 12, "q, dq, kp, kd, tau_ff should have size 12"
        self.latest_command = dict(
            kp=deepcopy(kp),
            kd=deepcopy(kd),
            q_des=deepcopy(q_des),
            dq_des=deepcopy(dq_des),
            tau_ff=deepcopy(tau_ff),
        )
        lowcmd = Go2pyLowCmd_(
            q_des,
            dq_des,
            kp,
            kd,
            tau_ff
        )
        self.lowcmd_writer.write(lowcmd)
        self.latest_command_stamp = time.time()

    def close(self):
        self.running = False

    def clip_velocity(self, v_x, v_y, ω_z):
        _v = np.array([[v_x], [v_y]])
        _scale = np.sqrt(_v.T @ self.P_v_max @ _v)[0, 0]

        if _scale > 1.0:
            scale = 1.0 / _scale
        else:
            scale = 1.0

        return scale * v_x, scale * v_y, np.clip(ω_z, self.ωz_min, self.ωz_max)

    def overheat(self):
        return False

    def getGravityInBody(self):
        q = self.getIMU()['quat']
        R = Rotation.from_quat(q).as_matrix()
        g_in_body = R.T @ np.array([0.0, 0.0, -1.0]).reshape(3, 1)
        return g_in_body
    
    def getForwardVecInBody(self):
        q = self.getIMU()['quat']
        R = Rotation.from_quat(q).as_matrix()
        g_in_body = R.T @ np.array([1.0, 0.0, 0.0]).reshape(3, 1)
        return g_in_body
