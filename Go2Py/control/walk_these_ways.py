import os
import pickle as pkl
import time

import numpy as np
import torch


def loadParameters(path):
    with open(path + "/parameters.pkl", "rb") as file:
        pkl_cfg = pkl.load(file)
        cfg = pkl_cfg["Cfg"]
    return cfg


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_") or key == "terrain":
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


class HistoryWrapper:
    def __init__(self, env):
        self.env = env

        if isinstance(self.env.cfg, dict):
            self.obs_history_length = self.env.cfg["env"]["num_observation_history"]
        else:
            self.obs_history_length = self.env.cfg.env.num_observation_history
        self.num_obs_history = self.obs_history_length * self.env.num_obs
        self.obs_history = torch.zeros(
            self.env.num_envs,
            self.num_obs_history,
            dtype=torch.float,
            device=self.env.device,
            requires_grad=False,
        )
        self.num_privileged_obs = self.env.num_privileged_obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        privileged_obs = info["privileged_obs"]

        self.obs_history = torch.cat(
            (self.obs_history[:, self.env.num_obs:], obs), dim=-1
        )
        return (
            {
                "obs": obs,
                "privileged_obs": privileged_obs,
                "obs_history": self.obs_history,
            },
            rew,
            done,
            info,
        )

    def get_observations(self):
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        self.obs_history = torch.cat(
            (self.obs_history[:, self.env.num_obs:], obs), dim=-1
        )
        return {
            "obs": obs,
            "privileged_obs": privileged_obs,
            "obs_history": self.obs_history,
        }

    def get_obs(self):
        obs = self.env.get_obs()
        privileged_obs = self.env.get_privileged_observations()
        self.obs_history = torch.cat(
            (self.obs_history[:, self.env.num_obs:], obs), dim=-1
        )
        return {
            "obs": obs,
            "privileged_obs": privileged_obs,
            "obs_history": self.obs_history,
        }

    def reset_idx(
        self, env_ids
    ):  # it might be a problem that this isn't getting called!!
        ret = self.env.reset_idx(env_ids)
        self.obs_history[env_ids, :] = 0
        return ret

    def reset(self):
        ret = self.env.reset()
        privileged_obs = self.env.get_privileged_observations()
        self.obs_history[:, :] = 0
        return {
            "obs": ret,
            "privileged_obs": privileged_obs,
            "obs_history": self.obs_history,
        }

    def __getattr__(self, name):
        return getattr(self.env, name)


class CommandInterface:
    def __init__(self, limits=None):
        self.limits = limits
        self.gaits = {
            "pronking": [0, 0, 0],
            "trotting": [0.5, 0, 0],
            "bounding": [0, 0.5, 0],
            "pacing": [0, 0, 0.5],
        }
        self.x_vel_cmd, self.y_vel_cmd, self.yaw_vel_cmd = 0.0, 0.0, 0.0
        self.body_height_cmd = 0.0
        self.step_frequency_cmd = 3.0
        self.gait = torch.tensor(self.gaits["trotting"])
        self.footswing_height_cmd = 0.06
        self.pitch_cmd = 0.0
        self.roll_cmd = 0.0
        self.stance_width_cmd = 0.25

    def setGaitType(self, gait_type):
        assert gait_type in [key for key in self.gaits.keys()], f'The gain type should be in {self.gaits.keys()}'
        self.gait = torch.tensor(self.gaits[gait_type])

    def get_command(self):
        command = np.zeros((19,))
        command[0] = self.x_vel_cmd
        command[1] = self.y_vel_cmd
        command[2] = self.yaw_vel_cmd
        command[3] = self.body_height_cmd
        command[4] = self.step_frequency_cmd
        command[5:8] = self.gait
        command[8] = 0.5
        command[9] = self.footswing_height_cmd
        command[10] = self.pitch_cmd
        command[11] = self.roll_cmd
        command[12] = self.stance_width_cmd
        return command, False


class Policy:
    def __init__(self, checkpoint_path):
        self.body = torch.jit.load(
            os.path.join(checkpoint_path, "checkpoints/body_latest.jit")
        )
        self.adaptation_module = torch.jit.load(
            os.path.join(checkpoint_path, "checkpoints/adaptation_module_latest.jit")
        )
        self.step_counter = 0
        self.perturbation = None

    def __call__(self, obs, info):
        with torch.no_grad():
            if self.perturbation is None:
                latent = self.adaptation_module.forward(obs["obs_history"].to("cpu"))
                self.perturbation = torch.rand_like(latent)*0.9
            else:
                latent = self.adaptation_module.forward(obs["obs_history"].to("cpu"))+self.perturbation
            if self.step_counter%50==0:
                self.perturbation = torch.rand_like(latent)*0.9
            self.step_counter +=1
            # print(latent, self.perturbation)
            info["latent"] = latent
            action = self.body.forward(
                torch.cat((obs["obs_history"].to("cpu"), latent), dim=-1)
            )
        return action


class WalkTheseWaysAgent:
    def __init__(self, cfg, command_profile, robot):
        self.robot = robot
        if not isinstance(cfg, dict):
            cfg = class_to_dict(cfg)
        self.cfg = cfg
        self.command_profile = command_profile
        # self.lcm_bridge = LCMBridgeClient(robot_name=self.robot_name)
        self.dt = self.cfg["control"]["decimation"] * self.cfg["sim"]["dt"]
        self.timestep = 0

        self.num_obs = self.cfg["env"]["num_observations"]
        self.num_envs = 1
        self.num_privileged_obs = self.cfg["env"]["num_privileged_obs"]
        self.num_actions = self.cfg["env"]["num_actions"]
        self.num_commands = self.cfg["commands"]["num_commands"]
        self.device = "cpu"

        if "obs_scales" in self.cfg.keys():
            self.obs_scales = self.cfg["obs_scales"]
        else:
            self.obs_scales = self.cfg["normalization"]["obs_scales"]

        self.commands_scale = np.array(
            [
                self.obs_scales["lin_vel"],
                self.obs_scales["lin_vel"],
                self.obs_scales["ang_vel"],
                self.obs_scales["body_height_cmd"],
                1,
                1,
                1,
                1,
                1,
                self.obs_scales["footswing_height_cmd"],
                self.obs_scales["body_pitch_cmd"],
                # 0, self.obs_scales["body_pitch_cmd"],
                self.obs_scales["body_roll_cmd"],
                self.obs_scales["stance_width_cmd"],
                self.obs_scales["stance_length_cmd"],
                self.obs_scales["aux_reward_cmd"],
                1,
                1,
                1,
                1,
                1,
                1,
            ]
        )[: self.num_commands]

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

        self.default_dof_pos = np.array(
            [
                self.cfg["init_state"]["default_joint_angles"][name]
                for name in joint_names
            ]
        )
        try:
            self.default_dof_pos_scale = np.array(
                [
                    self.cfg["init_state"]["default_hip_scales"],
                    self.cfg["init_state"]["default_thigh_scales"],
                    self.cfg["init_state"]["default_calf_scales"],
                    self.cfg["init_state"]["default_hip_scales"],
                    self.cfg["init_state"]["default_thigh_scales"],
                    self.cfg["init_state"]["default_calf_scales"],
                    self.cfg["init_state"]["default_hip_scales"],
                    self.cfg["init_state"]["default_thigh_scales"],
                    self.cfg["init_state"]["default_calf_scales"],
                    self.cfg["init_state"]["default_hip_scales"],
                    self.cfg["init_state"]["default_thigh_scales"],
                    self.cfg["init_state"]["default_calf_scales"],
                ]
            )
        except KeyError:
            self.default_dof_pos_scale = np.ones(12)
        self.default_dof_pos = self.default_dof_pos * self.default_dof_pos_scale

        self.p_gains = np.zeros(12)
        self.d_gains = np.zeros(12)
        for i in range(12):
            joint_name = joint_names[i]
            found = False
            for dof_name in self.cfg["control"]["stiffness"].keys():
                if dof_name in joint_name:
                    self.p_gains[i] = self.cfg["control"]["stiffness"][dof_name]
                    self.d_gains[i] = self.cfg["control"]["damping"][dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.0
                self.d_gains[i] = 0.0
                if self.cfg["control"]["control_type"] in ["P", "V"]:
                    print(
                        f"PD gain of joint {joint_name} were not defined, setting them to zero"
                    )

        print(f"p_gains: {self.p_gains}")

        self.commands = np.zeros((1, self.num_commands))
        self.actions = torch.zeros(12)
        self.last_actions = torch.zeros(12)
        self.gravity_vector = np.zeros(3)
        self.dof_pos = np.zeros(12)
        self.dof_vel = np.zeros(12)
        self.body_linear_vel = np.zeros(3)
        self.body_angular_vel = np.zeros(3)
        self.joint_pos_target = np.zeros(12)
        self.joint_vel_target = np.zeros(12)
        self.torques = np.zeros(12)
        self.contact_state = np.ones(4)
        self.foot_contact_forces_mag = np.zeros(4)
        self.prev_foot_contact_forces_mag = np.zeros(4)
        self.test = 0

        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float)
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float)

        if "obs_scales" in self.cfg.keys():
            self.obs_scales = self.cfg["obs_scales"]
        else:
            self.obs_scales = self.cfg["normalization"]["obs_scales"]

    def wait_for_state(self):
        # return self.lcm_bridge.getStates(timeout=2)
        pass

    def get_obs(self):
        cmds, reset_timer = self.command_profile.get_command()
        self.commands[:, :] = cmds[: self.num_commands]

        # self.state = self.wait_for_state()
        joint_state = self.robot.getJointStates()
        if joint_state is not None:
            self.gravity_vector = self.robot.getGravityInBody()
            self.dof_pos = np.array(joint_state['q'])[self.unitree_to_policy_map[:, 1]]
            self.dof_vel = np.array(joint_state['dq'])[self.unitree_to_policy_map[:, 1]]
            try:
                self.foot_contact_forces_mag = self.robot.getFootContact()
            except:
                pass

        if reset_timer:
            self.reset_gait_indices()

        ob = np.concatenate(
            (
                self.gravity_vector.reshape(1, -1),
                self.commands * self.commands_scale,
                (self.dof_pos - self.default_dof_pos).reshape(1, -1)
                * self.obs_scales["dof_pos"],
                self.dof_vel.reshape(1, -1) * self.obs_scales["dof_vel"],
                torch.clip(
                    self.actions,
                    -self.cfg["normalization"]["clip_actions"],
                    self.cfg["normalization"]["clip_actions"],
                )
                .cpu()
                .detach()
                .numpy()
                .reshape(1, -1),
            ),
            axis=1,
        )

        if self.cfg["env"]["observe_two_prev_actions"]:
            ob = np.concatenate(
                (ob, self.last_actions.cpu().detach().numpy().reshape(1, -1)), axis=1
            )

        if self.cfg["env"]["observe_clock_inputs"]:
            ob = np.concatenate((ob, self.clock_inputs), axis=1)

        return torch.tensor(ob, device=self.device).float()

    def get_privileged_observations(self):
        return None

    def publish_action(self, action, hard_reset=False):
        # command_for_robot = UnitreeLowCommand()
        self.joint_pos_target = (
            action[0, :12].detach().cpu().numpy() * self.cfg["control"]["action_scale"]
        ).flatten()
        self.joint_pos_target[[0, 3, 6, 9]] *= self.cfg["control"][
            "hip_scale_reduction"
        ]
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
        self.actions = torch.zeros(12)
        self.time = time.time()
        self.timestep = 0
        return self.get_obs()

    def reset_gait_indices(self):
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float)

    def step(self, actions, hard_reset=False):
        clip_actions = self.cfg["normalization"]["clip_actions"]
        self.last_actions = self.actions[:]
        self.actions = torch.clip(actions[0:1, :], -clip_actions, clip_actions)
        self.publish_action(self.actions, hard_reset=hard_reset)
        # time.sleep(max(self.dt - (time.time() - self.time), 0))
        # if self.timestep % 100 == 0:
        #     print(f"frq: {1 / (time.time() - self.time)} Hz")
        self.time = time.time()
        obs = self.get_obs()

        # clock accounting
        frequencies = self.commands[:, 4]
        phases = self.commands[:, 5]
        offsets = self.commands[:, 6]
        if self.num_commands == 8:
            bounds = 0
        else:
            bounds = self.commands[:, 7]
        self.gait_indices = torch.remainder(
            self.gait_indices + self.dt * frequencies, 1.0
        )

        if (
            "pacing_offset" in self.cfg["commands"]
            and self.cfg["commands"]["pacing_offset"]
        ):
            self.foot_indices = [
                self.gait_indices + phases + offsets + bounds,
                self.gait_indices + bounds,
                self.gait_indices + offsets,
                self.gait_indices + phases,
            ]
        else:
            self.foot_indices = [
                self.gait_indices + phases + offsets + bounds,
                self.gait_indices + offsets,
                self.gait_indices + bounds,
                self.gait_indices + phases,
            ]
        self.clock_inputs[:, 0] = torch.sin(2 * np.pi * self.foot_indices[0])
        self.clock_inputs[:, 1] = torch.sin(2 * np.pi * self.foot_indices[1])
        self.clock_inputs[:, 2] = torch.sin(2 * np.pi * self.foot_indices[2])
        self.clock_inputs[:, 3] = torch.sin(2 * np.pi * self.foot_indices[3])

        foot_contact_rate = np.abs(self.foot_contact_forces_mag - self.prev_foot_contact_forces_mag)
        self.prev_foot_contact_forces_mag = self.foot_contact_forces_mag.copy()

        infos = {
            "joint_pos": self.dof_pos[np.newaxis, :],
            "joint_vel": self.dof_vel[np.newaxis, :],
            "joint_pos_target": self.joint_pos_target[np.newaxis, :],
            "joint_vel_target": self.joint_vel_target[np.newaxis, :],
            "body_linear_vel": self.body_linear_vel[np.newaxis, :],
            "body_angular_vel": self.body_angular_vel[np.newaxis, :],
            "contact_state": self.contact_state[np.newaxis, :],
            "clock_inputs": self.clock_inputs[np.newaxis, :],
            "body_linear_vel_cmd": self.commands[:, 0:2],
            "body_angular_vel_cmd": self.commands[:, 2:],
            "privileged_obs": None,
            "foot_contact_rate": foot_contact_rate[np.newaxis, :],
            "foot_contact_forces_mag": self.foot_contact_forces_mag.copy(),
            "actions": actions.cpu().numpy(),
        }

        self.timestep += 1
        return obs, None, None, infos
