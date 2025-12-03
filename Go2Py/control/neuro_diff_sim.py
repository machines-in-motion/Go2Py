import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import onnxruntime
from models import model_utils

class ActorStochasticMLP(nn.Module):
    def __init__(self):
        super(ActorStochasticMLP, self).__init__()

        self.hidden_size = 256
        obs_dim = 49
        action_dim = 12
        self.gru = nn.GRU(obs_dim, self.hidden_size, batch_first=True)
        self.layer_dims = [obs_dim + self.hidden_size] + [256, 128, 64] + [action_dim]

        init_ = lambda m: model_utils.init(m, nn.init.orthogonal_, lambda x: nn.init.
                        constant_(x, 0), np.sqrt(2))

        modules = []
        for i in range(len(self.layer_dims) - 1):
            modules.append(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1]))
            if i < len(self.layer_dims) - 2:
                modules.append(model_utils.get_activation_func('elu'))
                modules.append(torch.nn.LayerNorm(self.layer_dims[i+1]))
            else:
                modules.append(model_utils.get_activation_func('identity'))

        self.mu_net = nn.Sequential(*modules)

        logstd = -1.0

        self.logstd = torch.nn.Parameter(torch.ones(action_dim, dtype=torch.float32) * logstd)

        self.action_dim = action_dim
        self.obs_dim = obs_dim

        self.min_logstd = -1.427

        print(self.mu_net)
        print(self.logstd)

    def forward(self, obs, l):
        out, time_latent = self.gru(obs, l)
        x = torch.cat([obs, out], dim=-1)

        mu = self.mu_net(x) # torch.tanh(self.mu_net(x)) * 2.0

        return mu, time_latent

class Policy:
    def __init__(self, checkpoint_path):
        self.agent = ActorStochasticMLP()
        agent_init, _, _, self.obs_rms, _, _ = torch.load(checkpoint_path, map_location="cpu")
        self.agent.load_state_dict(agent_init.state_dict())
        onnx_file_name = checkpoint_path.replace(".pt", ".onnx")
        self.actor_hidden_in = np.zeros((1, 256), dtype=np.float32)

        dummy_input = torch.randn(1, 49)
        with torch.no_grad():
            torch_out, torch_hidden_out = self.agent(dummy_input, l = torch.tensor(self.actor_hidden_in))

        torch.onnx.export(
            self.agent,                          # The model being converted
            (dummy_input, torch.tensor(self.actor_hidden_in)),                    # An example input for the model
            onnx_file_name,                 # Output file name
            export_params=True,             # Store trained parameter weights inside the model file
            opset_version=11,               # ONNX version (opset) to export to, adjust as needed
            do_constant_folding=True,       # Whether to perform constant folding optimization
            input_names=['input', 'hidden_in'],          # Name of the input in the ONNX graph (can be customized)
            output_names=['action', 'hidden_out'],  # Name of the output (assuming get_action and get_value are key outputs)
        )

        self.ort_session = onnxruntime.InferenceSession(onnx_file_name, providers=["CPUExecutionProvider"])
        ort_inputs = {'input': dummy_input.numpy(), 'hidden_in': self.actor_hidden_in}
        ort_outs = self.ort_session.run(None, ort_inputs)
        action, hidden_out = ort_outs[0], ort_outs[1]
        np.testing.assert_allclose(torch_out.numpy(), action, rtol=1e-03, atol=1e-05)
        np.testing.assert_allclose(torch_hidden_out.numpy(), hidden_out, rtol=1e-03, atol=1e-05)
        print("Exported model has been tested with ONNXRuntime, and the result looks good!")

    def __call__(self, obs, info):
        obs = torch.cat(
            [torch.zeros((1, 3)), torch.tensor(obs[np.newaxis]), torch.zeros((1, 9))],
            dim = -1
        )
        obs = self.obs_rms.normalize(obs)[:, 3:-9].numpy().astype(np.float32)
        ort_inputs = {'input': obs, 'hidden_in': self.actor_hidden_in}
        ort_outs = self.ort_session.run(None, ort_inputs)
        self.actor_hidden_in = ort_outs[1]
        return ort_outs[0]

class CommandInterface:
    def __init__(self, limits=None):
        self.limits = limits
        self.x_vel_cmd, self.y_vel_cmd, self.yaw_vel_cmd = 0.0, 0.0, 0.0

    def get_command(self):
        command = np.zeros((3,))
        command[0] = self.x_vel_cmd
        command[1] = self.y_vel_cmd
        command[2] = self.yaw_vel_cmd
        return command, False


class NeuroDiffSimAgent:
    def __init__(self, command_profile, robot):
        self.robot = robot
        self.command_profile = command_profile
        # self.lcm_bridge = LCMBridgeClient(robot_name=self.robot_name)
        self.sim_dt = 0.001
        self.decimation = 20
        self.dt = self.sim_dt * self.decimation
        self.timestep = 0

        self.device = "cpu"

        joint_names = [
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
        ]

        policy_joint_names = joint_names

        unitree_joint_names = [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ]

        policy_to_unitree_map = []
        unitree_to_policy_map = []
        for i, policy_joint_name in enumerate(policy_joint_names):
            id = np.where([name == policy_joint_name for name in unitree_joint_names])[0][0]
            policy_to_unitree_map.append((i, id))
        self.policy_to_unitree_map = np.array(policy_to_unitree_map).astype(np.uint32)

        for i, unitree_joint_name in enumerate(unitree_joint_names):
            id = np.where([name == unitree_joint_name for name in policy_joint_names])[0][0]
            unitree_to_policy_map.append((i, id))
        self.unitree_to_policy_map = np.array(unitree_to_policy_map).astype(np.uint32)

        default_joint_angles = {
            "FL_hip_joint": 0.1,
            "RL_hip_joint": 0.1,
            "FR_hip_joint": -0.1,
            "RR_hip_joint": -0.1,
            "FL_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "FR_thigh_joint": 0.8,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RR_calf_joint": -1.5
        }
        self.default_dof_pos = np.array(
            [
                default_joint_angles[name]
                for name in joint_names
            ]
        )

        self.default_dof_pos = self.default_dof_pos

        self.p_gains = np.zeros(12)
        self.d_gains = np.zeros(12)
        for i in range(12):
            self.p_gains[i] = 20.0
            self.d_gains[i] = 0.5

        print(f"p_gains: {self.p_gains}")

        self.commands = np.zeros(3)
        self.actions = np.zeros((1, 12))
        self.last_actions = np.zeros((1,12))
        self.gravity_vector = np.zeros(3)
        self.dof_pos = np.zeros(12)
        self.dof_vel = np.zeros(12)
        self.pos = np.zeros(3)
        self.body_linear_vel = np.zeros(3)
        self.body_angular_vel = np.zeros(3)
        self.joint_pos_target = np.zeros(12)
        self.joint_vel_target = np.zeros(12)
        self.prev_joint_acc = None
        self.torques = np.zeros(12)
        self.contact_state = np.ones(4)
        self.foot_contact_forces_mag = np.zeros(4)
        self.prev_foot_contact_forces_mag = np.zeros(4)
        self.test = 0

        self.gait_indices = np.zeros(1, dtype=np.float32)
        self.clock_inputs = np.zeros(4, dtype=np.float32)

    def wait_for_state(self):
        # return self.lcm_bridge.getStates(timeout=2)
        pass

    def get_obs(self):
        cmds, reset_timer = self.command_profile.get_command()
        self.commands[:] = cmds

        # self.state = self.wait_for_state()
        joint_state = self.robot.getJointStates()
        if joint_state is not None:
            self.gravity_vector = self.robot.getGravityInBody()
            self.prev_dof_pos = self.dof_pos.copy()
            self.dof_pos = np.array(joint_state['q'])[self.unitree_to_policy_map[:, 1]]
            self.prev_dof_vel = self.dof_vel.copy()
            self.dof_vel = np.array(joint_state['dq'])[self.unitree_to_policy_map[:, 1]]
            self.body_angular_vel = self.robot.getIMU()["gyro"]
            
            try:
                self.foot_contact_forces_mag = self.robot.getFootContact()
                self.body_linear_vel = self.robot.getLinVel()
                self.pos, _ = self.robot.getPose()
            except:
                pass

        ob = np.concatenate(
            (
                self.body_angular_vel * 0.25,
                self.commands * np.array([2.0, 2.0, 0.25]),
                self.gravity_vector[:, 0],
                self.dof_pos * 1.0,
                #((self.dof_pos - self.prev_dof_pos) / self.dt) * 0.05,
                self.dof_vel * 0.05,
                self.last_actions[0],
                self.clock_inputs
            ),
            axis=0,
        )

        #return torch.tensor(ob, device=self.device).float()
        return ob

    def publish_action(self, action, hard_reset=False):
        # command_for_robot = UnitreeLowCommand()
        #self.joint_pos_target = (
        #    action[0, :12].detach().cpu().numpy() * 0.25
        #).flatten()
        self.joint_pos_target = (
            action[0, :12] * 0.25
        ).flatten()
        self.joint_pos_target += self.default_dof_pos
        self.joint_vel_target = np.zeros(12)
        # command_for_robot.q_des = self.joint_pos_target
        # command_for_robot.dq_des = self.joint_vel_target
        # command_for_robot.kp = self.p_gains
        # command_for_robot.kd = self.d_gains
        # command_for_robot.tau_ff = np.zeros(12)
        if hard_reset:
            command_for_robot.id = -1

        self.torques = (self.joint_pos_target - self.dof_pos) * self.p_gains + (
            self.joint_vel_target - self.dof_vel
        ) * self.d_gains
        # self.lcm_bridge.sendCommands(command_for_robot)
        self.robot.setCommands(self.joint_pos_target[self.policy_to_unitree_map[:, 1]],
                               self.joint_vel_target[self.policy_to_unitree_map[:, 1]],
                               self.p_gains[self.policy_to_unitree_map[:, 1]],
                               self.d_gains[self.policy_to_unitree_map[:, 1]],
                               np.zeros(12))

    def reset(self):
        self.actions = torch.zeros((1, 12))
        self.time = time.time()
        self.timestep = 0
        return self.get_obs()

    def step(self, actions, hard_reset=False):
        self.last_actions = self.actions[:]
        self.actions = actions
        self.publish_action(self.actions, hard_reset=hard_reset)
        # time.sleep(max(self.dt - (time.time() - self.time), 0))
        # if self.timestep % 100 == 0:
        #     print(f"frq: {1 / (time.time() - self.time)} Hz")
        self.time = time.time()
        #obs = self.get_obs()

        joint_acc = np.abs(self.prev_dof_vel - self.dof_vel) / self.dt
        if self.prev_joint_acc is None:
            self.prev_joint_acc = np.zeros_like(joint_acc)
        joint_jerk = np.abs(self.prev_joint_acc - joint_acc) / self.dt
        self.prev_joint_acc = joint_acc.copy()

        # clock accounting
        frequencies = 3.
        phases = 0.5
        offsets = 0.
        bounds = 0
        self.gait_indices = np.remainder(
            self.gait_indices + self.dt * frequencies, 1.0
        )

        self.foot_indices = [
            self.gait_indices + phases + offsets + bounds,
            self.gait_indices + offsets,
            self.gait_indices + bounds,
            self.gait_indices + phases,
        ]

        self.clock_inputs[0] = np.sin(2 * np.pi * self.foot_indices[0])
        self.clock_inputs[1] = np.sin(2 * np.pi * self.foot_indices[1])
        self.clock_inputs[2] = np.sin(2 * np.pi * self.foot_indices[2])
        self.clock_inputs[3] = np.sin(2 * np.pi * self.foot_indices[3])

        foot_contact_rate = np.abs(self.foot_contact_forces_mag - self.prev_foot_contact_forces_mag)
        self.prev_foot_contact_forces_mag = self.foot_contact_forces_mag.copy()

        infos = {
            "joint_pos": self.dof_pos[np.newaxis, :],
            "joint_vel": self.dof_vel[np.newaxis, :],
            "joint_pos_target": self.joint_pos_target[np.newaxis, :],
            "joint_vel_target": self.joint_vel_target[np.newaxis, :],
            "body_linear_vel": self.body_linear_vel[np.newaxis, :],
            "body_angular_vel": self.body_angular_vel[np.newaxis, :],
            "body_pos": self.pos[np.newaxis, :].copy(),
            "contact_state": self.contact_state[np.newaxis, :],
            "body_linear_vel_cmd": self.commands[np.newaxis, 0:2],
            "body_angular_vel_cmd": self.commands[np.newaxis, 2:],
            "torques": self.torques,
            "foot_contact_forces_mag": self.foot_contact_forces_mag.copy(),
            "joint_acc": joint_acc[np.newaxis, :],
            "joint_jerk": joint_jerk[np.newaxis, :],
            "foot_contact_rate": foot_contact_rate[np.newaxis, :],
        }

        self.timestep += 1
        return None, None, None, infos
