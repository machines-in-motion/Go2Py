import threading
import numpy as np
import time
from enum import Enum


class FSM:
    def __init__(self, robot, remote, safety_hypervisor, control_dT=1/50, user_controller_callback=None):
        self.robot = robot
        self.remote = remote
        self.remote.flushStates()
        self.safety = safety_hypervisor
        self.user_controller_callback = user_controller_callback

        self.state = "damping"
        self.tracking_kp = np.array(4 * [150, 150, 150.]).reshape(12)
        self.tracking_kv = np.array(12 * [3.])
        self.damping_kv = np.array(12 * [2.])

        self.tracking_complete = True
        self.robot.setCommands(
            np.zeros(12),
            np.zeros(12),
            np.zeros(12),
            self.damping_kv,
            np.zeros(12))
        self.fsm_dT = 1. / 50.
        
        self.control_dT = control_dT
        self.dT = self.fsm_dT

        self.modes = {"tracking": self.trackingControlUpdate,
                      "damping": self.dampingControlUpdate,
                      "user": self.userControlUpdate,
                      }
        self.setMode("damping")

        self.running = True
        self.fsm_thread = threading.Thread(target=self.update)
        self.fsm_thread.start()
        # if the robot is a simulation, create a thread for stepping it
        if self.robot.simulated:
            if not self.robot.async_mode:
                self.sim_thread = threading.Thread(target=self.simUpdate)
                self.sim_thread.start()

    def setMode(self, mode):
        assert mode in self.modes.keys(), 'the requested control update mode is not implemented'
        self.updateCommands = self.modes[mode]
        # print(f'setting mode to {mode}')

    def moveTo(self, target, duration=0.5):
        # assert self.tracking_complete, 'The previous moveTo command is not completed yet!'
        self.q_start = self.robot.getJointStates()['q']
        self.q_target = target
        self.time = 0.0
        self.move_duration = duration
        self.q_des = lambda t: [self.q_start +
                                np.clip((t) /
                                        self.move_duration, 0, 1) *
                                (self.q_target -
                                 self.q_start), True if np.clip((t) /
                                                                self.move_duration, 0, 1) == 1 else False]  # q_des(t), Movement finished
        self.tracking_complete = False
        self.setMode("tracking")

    def trackingControlUpdate(self):
        self.time += self.dT
        q_des, done = self.q_des(self.time)
        self.robot.setCommands(
            q_des,
            np.zeros(12),
            self.tracking_kp,
            self.tracking_kv,
            np.zeros(12))
        self.tracking_complete = done

    def dampingControlUpdate(self):
        self.robot.setCommands(
            np.zeros(12),
            np.zeros(12),
            np.zeros(12),
            self.damping_kv,
            np.zeros(12))

    def userControlUpdate(self):
        if self.user_controller_callback is not None:
            self.user_controller_callback(self.robot, self.remote)

    def simUpdate(self):
        while self.running:
            self.robot.step()
            time.sleep(self.robot.dt)

    def update(self):
        while self.running:
            tic = time.time()
            getattr(self, self.state)()
            self.updateCommands()
            while time.time()-tic < self.fsm_dT:
                time.sleep(0.0001)

    def close(self):
        self.running = False

    # The following functions each are the states of the FSM
    def damping(self):
        # print('damping')
        if self.remote.standUpDownSeq():
            self.moveTo(self.robot.prestanding_q, 1)
            self.state = "pre_standing"
        else:
            self.setMode("damping")
            self.state = "damping"

    def pre_standing(self):
        # print("pre_stance")
        if self.tracking_complete:
            self.moveTo(self.robot.standing_q, duration=1.5)
            self.state = 'standing'
        else:
            self.state = "pre_standing"

    def standing(self):
        # print("standing")
        if self.tracking_complete:
            # self.moveTo(robot.standing_q, duration=1)
            self.state = 'locked_stance'
        else:
            self.state = "standing"

    def locked_stance(self):
        # print("locked_stance")
        if self.remote.startSeq():
            self.setMode("user")
            self.dT = self.control_dT
            self.state = "user_loop"
            self.robot.setCommands(
                np.zeros(12),
                np.zeros(12),
                np.zeros(12),
                np.zeros(12),
                np.zeros(12))
        elif self.remote.standUpDownSeq() or self.robot.overheat():
            self.moveTo(self.robot.sitting_q, duration=1.5)
            self.state = "sitting"

    def user_loop(self):
        # print("user_loop")
        if self.safety.unsafe():
            self.dT = self.fsm_dT
            self.setMode("damping")
        elif self.remote.standUpDownSeq() or self.safety.controlTimeout():
            self.dT = self.fsm_dT
            self.moveTo(self.robot.standing_q, duration=1)
            self.timer = time.time()
            self.state = "switch_back_to_locked_stance"
        else:
            self.state = "user_loop"

    def sitting(self):
        # print('sitting')
        if self.tracking_complete:
            self.setMode("damping")
            self.state = 'damping'
        else:
            self.state = "sitting"

    def switch_back_to_locked_stance(self):
        if time.time() - self.timer > 0.5:
            # print("going back to locked stance")
            self.state = "locked_stance"
