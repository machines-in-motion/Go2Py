import time
import numpy as np


class SafetyHypervisor():
    def __init__(self, robot, max_temperature=70, body_stability_idx=0.7):
        self.robot = robot
        self.max_temperature = max_temperature
        self.normalize = lambda v: v/np.linalg.norm(v)
        self.compute_body_stability_idx = lambda v: (self.normalize(v).T@np.array([0,0,-1]))
        self.body_stability_idx = body_stability_idx
    
    def unsafe(self):
        if self.robot.simulated:
            return False
        else:
            state = self.robot.getJointStates()
            temperature = state['temperature']
            if np.any(np.array(temperature)>self.max_temperature):
                print(f'One of the actuators reached max temperature of {self.max_temperature}')
                return True
            #if self.compute_body_stability_idx(self.robot.getGravityInBody()) < self.body_stability_idx:
            #    print('Body is tilted too much! deactivating the controller')
            #    return True
        return False

    def controlTimeout(self):
        if time.time() - self.robot.latest_command_stamp > 0.1:
            print('controller timeout')
            return True
        else:
            return False
