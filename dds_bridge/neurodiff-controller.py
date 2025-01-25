from Go2Py.robot.fsm import FSM
from Go2Py.robot.safety import SafetyHypervisor
from Go2Py.robot.interface import GO2Real
from Go2Py.control.neuro_diff import NeuroDiffAgent

import torch
import numpy as np
import time


class NeuroDiffController:
    def __init__(self, robot, remote, checkpoint):
        self.remote = remote
        self.robot = robot
        self.policy = Policy(checkpoint)
        self.command_profile = CommandInterface()
        self.agent = NeuroDiffSimAgent(self.command_profile, self.robot)
        self.hist_data = {}

    def init(self):
        self.obs = self.agent.reset()
        self.policy_info = {}
        self.command_profile.yaw_vel_cmd = 0.0
        self.command_profile.x_vel_cmd = 0.0
        self.command_profile.y_vel_cmd = 0.0

    def update(self, robot, remote):
        if not hasattr(self, "obs"):
            self.init()
        commands = getRemote(remote)
        self.command_profile.yaw_vel_cmd = 0.0
        self.command_profile.x_vel_cmd = 0.0
        self.command_profile.y_vel_cmd = 0.0

        self.obs = self.agent.get_obs()
        action = self.policy(self.obs, self.policy_info)
        _, self.ret, self.done, self.info = self.agent.step(action)
        for key, value in self.info.items():
            if key in self.hist_data:
                self.hist_data[key].append(value)
            else:
                self.hist_data[key] = [value]



robot = GO2Real(mode='lowlevel')
safety_hypervisor = SafetyHypervisor(robot)

time.sleep(3)

print(robot.getJointStates())

robot.close()