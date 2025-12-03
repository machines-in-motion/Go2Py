import threading
# from pynput import keyboard
import pygame
from time import sleep
import threading
from pygame.locals import * 
import numpy as np
import time

class JoyManager():

    def __init__(self, user_callback = None):
        self.dt = 0.01 #Sampling frequence of the joystick
        self.runnign = False
        self.joy_thread_handle = threading.Thread(target=self.joy_thread)
        #an optional callback that can be used to send the reading values to a user provided function
        self.user_callback = user_callback
        pygame.init()
        self.offset = None

    def joy_thread(self):
        while self.running:   
            for event in [pygame.event.wait(200), ] + pygame.event.get():
                    # QUIT             none
                    # ACTIVEEVENT      gain, state
                    # KEYDOWN          unicode, key, mod
                    # KEYUP            key, mod
                    # MOUSEMOTION      pos, rel, buttons
                    # MOUSEBUTTONUP    pos, button
                    # MOUSEBUTTONDOWN  pos, button
                    # JOYAXISMOTION    joy, axis, value
                    # JOYBALLMOTION    joy, ball, rel
                    # JOYHATMOTION     joy, hat, value
                    # JOYBUTTONUP      joy, button
                    # JOYBUTTONDOWN    joy, button
                    # VIDEORESIZE      size, w, h
                    # VIDEOEXPOSE      none
                    # USEREVENT        code
                    if event.type == QUIT:
                        self.stop_daq()
                    elif event.type == KEYDOWN and event.key in [K_ESCAPE, K_q]:
                        self.stop_daq()
                    elif event.type == JOYAXISMOTION:
                        self.analog_cmd[event.axis] = event.value
                    elif event.type == JOYBUTTONUP:
                        self.digital_cmd[event.button] = 0
                    elif event.type == JOYBUTTONDOWN:
                        self.digital_cmd[event.button] = 1
                    if self.user_callback is not None and event.type!=NOEVENT:
                        if self.offset is None:
                            self.user_callback(self.analog_cmd, self.digital_cmd)
                        else:
                            self.user_callback(np.array(self.analog_cmd)-self.offset, self.digital_cmd)

                
    def start_daq(self, joy_idx):
        #Get the joy object
        assert pygame.joystick.get_count() != 0, 'No joysticks detected, you can not start the class'
        assert pygame.joystick.get_count() >= joy_idx, 'The requested joystick ID exceeds the number of availble devices'
        self.joy = pygame.joystick.Joystick(joy_idx)
        
        self.analog_cmd = [self.joy.get_axis(i) for i in range(self.joy.get_numaxes())]
        self.digital_cmd = [self.joy.get_button(i) for i in range(self.joy.get_numbuttons())]
        self.running = True
        self.joy_thread_handle.start()
        
    def stop_daq(self):
        self.running = False
        self.joy_thread_handle.join()
        
    def read_raw(self):
        return self.analog_cmd, self.digital_cmd

    def offset_calibration(self):
        analog , _ = self.read_raw()
        offset = np.array(analog)
        print('Put your stick at reset and do not touch it while calibrating')
        sleep(1)
        for i in range(2000):
            sleep(0.001)
            analog , _ = self.read_raw()
            offset += np.array(analog)
        self.offset = offset / 2000

    def read_values(self):
        analog, digital = self.read_raw()
        if self.offset is None:
            return analog, digital
        else:
            return analog-self.offset, digital
        
class XBoxController(JoyManager):
    def __init__(self, joystick_index):
        super(XBoxController, self).__init__()
        self.start_daq(joystick_index)
        time.sleep(0.5)
        try:
            self.offset_calibration()
        except:
            raise Exception(f'Could not communicate with the joystick with index: {joystick_index}')
        
    def getStates(self, deadzone=0.1):
        analog, digital = self.read_values()
        for i in [0,1,3,4]:
            if np.abs(analog[i])<=deadzone:
                analog[i]=0

        left_joy = analog[0:2]
        right_joy = analog[3:5]
        left_joy[1] =  -left_joy[1]
        right_joy[1] = -right_joy[1]
        left_trigger = analog[2]
        right_trigger = analog[5]
        A, B, X, Y = digital[0:4]
        left_bumper, right_bumper = digital[4:6]
        options_left, options_right = digital[6:8]
        left_joy_btn, right_joy_btn = digital[9:]

        return {
            'left_joy':left_joy,
            'right_joy':right_joy,
            'left_trigger':left_trigger,
            'right_trigger':right_trigger,
            'A':A,
            'B':B,
            'X':X,
            'Y':Y,
            'left_bumper':left_bumper,
            'right_bumper':right_bumper,
            'options_left':options_left,
            'options_right':options_right,
            'left_joy_btn':left_joy_btn,
            'right_joy_btn':right_joy_btn
        }
    def close(self):
        self.stop_daq()


class BaseRemote:
    def __init__(self):
        pass

    def startSeq(self):
        return False

    def standUpDownSeq(self):
        return False

    def flushStates(self):
        pass


remote = BaseRemote()


class KeyboardRemote(BaseRemote):
    def __init__(self):
        super().__init__()
        self.start_seq_flag = False
        self.stand_up_down_seq_flag = False
        self.listener_thread = threading.Thread(target=self._listen_to_keyboard)
        self.listener_thread.daemon = True
        self.listener_thread.start()
        self.yaw_vel_cmd = 0
        self.x_vel_cmd = 0
        self.y_vel_cmd = 0

    def _on_press(self, key):
        try:
            if key.char == 's':  # Start sequence
                self.start_seq_flag = True
            elif key.char == 'u':  # Stand up/down sequence
                self.stand_up_down_seq_flag = True
        except AttributeError:
            pass  # Special keys (like space) will be handled here

    def _on_release(self, key):
        try:
            if key.char == 's':  # Start sequence
                self.start_seq_flag = False
            elif key.char == 'u':  # Stand up/down sequence
                self.stand_up_down_seq_flag = False
        except AttributeError:
            pass  # Special keys (like space) will be handled here

    def _listen_to_keyboard(self):
        with keyboard.Listener(on_press=self._on_press, on_release=self._on_release) as listener:
            listener.join()

    def startSeq(self):
        if self.start_seq_flag:
            self.start_seq_flag = False
            return True
        return False

    def standUpDownSeq(self):
        if self.stand_up_down_seq_flag:
            self.stand_up_down_seq_flag = False
            return True
        return False

    def flushStates(self):
        self.stand_up_down_seq_flag = False
        self.start_seq_flag = False

    def getCommands(self):
        return np.array([self.y_vel_cmd, self.x_vel_cmd, self.yaw_vel_cmd])

class UnitreeRemote(BaseRemote):
    def __init__(self, robot):
        self.robot = robot

    def startSeq(self):
        remote = self.robot.getRemoteState()
        if remote.btn.start:
            return True
        else:
            return False

    def standUpDownSeq(self):
        remote = self.robot.getRemoteState()
        if remote.btn.L2 and remote.btn.A:
            return True
        else:
            return False

    def getEstop(self):
        remote = self.robot.getRemoteState()
        if remote.btn.L2 and remote.btn.R2:
            return True
        else:
            return False

class XBoxRemote(BaseRemote):
    def __init__(self):
        self.xbox_controller = XBoxController(0)

    def startSeq(self):
        remote = self.xbox_controller.getStates()
        if remote["left_bumper"] and remote["X"]:
            return True
        else:
            return False

    def standUpDownSeq(self):
        remote = self.xbox_controller.getStates()
        if remote["left_bumper"] and remote["A"]:
            return True
        else:
            return False

    def flushStates(self):
        self.stand_up_down_seq_flag = False
        self.start_seq_flag = False

    def getCommands(self):
        remote = self.xbox_controller.getStates()
        return np.concatenate([remote["left_joy"], remote["right_joy"][0:1]])