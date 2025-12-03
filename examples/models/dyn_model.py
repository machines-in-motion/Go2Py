# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
import numpy as np

from models import model_utils

from torch.autograd import Function

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, activation_fn, use_layer_norm=True):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = activation_fn
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim, eps=1e-03)
        # Ensure dimensions match for residual connection
        self.match_dimensions = input_dim != output_dim
        if self.match_dimensions:
            self.dimension_matcher = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        identity = x
        out = self.linear(x)
        if self.use_layer_norm:
            out = self.layer_norm(out)
        out = self.activation(out)
        if self.match_dimensions:
            identity = self.dimension_matcher(identity)
        out = out + identity  # Residual connection
        return out

class SimNorm(nn.Module):
    """
    Simplicial normalization.
    Adapted from https://arxiv.org/abs/2204.00616.
    """

    def __init__(self, simnorm_dim):
        super().__init__()
        self.dim = simnorm_dim

    def forward(self, x):
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shp)

    def __repr__(self):
        return f"SimNorm(dim={self.dim})"

class StochSSM(nn.Module):
    def __init__(self, obs_dim, action_dim, cfg_network, learn_reward, device='cuda:0'):
        super().__init__()

        self.device = device

        self.recurrent = cfg_network["dyn_model_mlp"].get("recurrent", False)
        if self.recurrent:
            self.hidden_size = int(cfg_network["dyn_model_mlp"].get("hidden_size", 128))
            self.gru = nn.GRU(obs_dim + action_dim, self.hidden_size, batch_first=True).to(device)
            self.layer_dims = [obs_dim + action_dim + self.hidden_size] + cfg_network['dyn_model_mlp']['units'] + [obs_dim * 2]
        else:
            self.layer_dims = [obs_dim + action_dim] + cfg_network['dyn_model_mlp']['units'] + [obs_dim * 2]

        modules = []
        for i in range(len(self.layer_dims) - 1):
            modules.append(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1]))
            if i < len(self.layer_dims) - 2:
                if i == 0:
                    modules.append(SimNorm(8))
                else:
                    modules.append(model_utils.get_activation_func(cfg_network['dyn_model_mlp']['activation']))
                #modules.append(model_utils.get_activation_func(cfg_network['dyn_model_mlp']['activation']))
                #modules.append(SimNorm(8))
                modules.append(torch.nn.LayerNorm(self.layer_dims[i+1], eps=1e-03))
        """if i < len(self.layer_dims) - 2:
                if i == 0:
                    modules.append(ResidualBlock(self.layer_dims[i], self.layer_dims[i + 1], SimNorm(8)))
                else:
                    modules.append(ResidualBlock(self.layer_dims[i], self.layer_dims[i + 1], model_utils.get_activation_func(cfg_network['dyn_model_mlp']['activation'])))
            else:
                modules.append(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1]))"""

        self.dyn_model = nn.Sequential(*modules).to(device)

        self.learn_reward = learn_reward
        if self.learn_reward:
            self.num_bins = cfg_network['dyn_model_mlp']['num_bins']
            self.vmin = cfg_network['dyn_model_mlp']['vmin']
            self.vmax = cfg_network['dyn_model_mlp']['vmax']
            self.reward_layer_dims = [obs_dim + action_dim] + cfg_network['dyn_model_mlp']['reward_head_units'] + [self.num_bins]

            modules = []
            for i in range(len(self.reward_layer_dims) - 1):
                modules.append(nn.Linear(self.reward_layer_dims[i], self.reward_layer_dims[i + 1]))
                if i < len(self.reward_layer_dims) - 2:
                    modules.append(model_utils.get_activation_func(cfg_network['dyn_model_mlp']['activation']))
                    modules.append(torch.nn.LayerNorm(self.reward_layer_dims[i+1], eps=1e-03))

            self.reward_model = nn.Sequential(*modules).to(device)

        """self.ssm_fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 1024)
        self.ssm_ln1 = nn.LayerNorm(1024, eps=1e-03)

        self.ssm_fc2 = nn.Linear(1024, 1024)
        self.ssm_ln2 = nn.LayerNorm(1024, eps=1e-03)

        self.ssm_fc3 = nn.Linear(1024, (np.array(env.single_observation_space.shape).prod()) * 2)"""

        # Must be put outside in case of requires_grad=True and ensemble of models
        self.max_logvar = nn.Parameter(torch.ones(1, obs_dim) * 0.5, requires_grad=False).to(device)
        self.min_logvar = nn.Parameter(torch.ones(1, obs_dim) * -10, requires_grad=False).to(device)

        print(self.dyn_model)
        if self.learn_reward:
            print(self.reward_model)

    def forward(self, s, a, l = None):
        x = torch.cat([s, a], dim=-1)
        time_latent = l
        if self.recurrent:
            out, time_latent = self.gru(x, l)
            x = torch.cat([x, out], dim=-1)

        h = self.dyn_model(x)

        mean, logvar = h.chunk(2, dim=-1)
        # Differentiable clip, funny! A.1 in https://arxiv.org/pdf/1805.12114
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        return mean, logvar, time_latent

    def reward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        rew = self.reward_model(x)
        return self.almost_two_hot_inv(rew)

    def raw_reward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        rew = self.reward_model(x)
        return rew

    def almost_two_hot_inv(self, x):
        """Converts a batch of soft two-hot encoded vectors to scalars."""
        if self.num_bins == 0 or self.num_bins == None:
            return x
        elif self.num_bins == 1:
            return symexp(x)
        # TODO this computation below can probably be optimized
        vals = torch.linspace(self.vmin, self.vmax, self.num_bins, device=x.device)
        x = F.softmax(x, dim=-1)
        x = torch.sum(x * vals, dim=-1, keepdim=True)
        return x

    def forward_with_log_ratio(self, s, a, next_obs_delta):
        x = torch.cat([s, a], dim=-1)

        h = self.dyn_model(x)

        mean, logvar = h.chunk(2, dim=-1)
        # Differentiable clip, funny! A.1 in https://arxiv.org/pdf/1805.12114
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        std = torch.sqrt(logvar.exp())
        dist = Normal(mean, std)

        return mean, logvar, dist.log_prob(next_obs_delta).sum(-1) - dist.log_prob(mean).sum(-1)

class VAESSM(nn.Module):
    def __init__(self, obs_dim, action_dim, cfg_network, learn_reward, device='cuda:0'):
        super().__init__()

        self.device = device

        # Encoder
        self.recurrent = cfg_network["dyn_model_mlp"].get("recurrent", False)
        self.latent_dim = int(cfg_network["dyn_model_mlp"].get("latent_dim", 64))
        if self.recurrent:
            self.hidden_size = int(cfg_network["dyn_model_mlp"].get("hidden_size", 128))
            self.gru = nn.GRU(obs_dim + action_dim, self.hidden_size, batch_first=True).to(device)
            self.layer_dims = [obs_dim + action_dim + self.hidden_size + obs_dim] + cfg_network['dyn_model_mlp']['units'] + [self.latent_dim * 2]
        else:
            self.layer_dims = [obs_dim + action_dim + obs_dim] + cfg_network['dyn_model_mlp']['units'] + [self.latent_dim * 2]

        modules = []
        for i in range(len(self.layer_dims) - 1):
            modules.append(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1]))
            if i < len(self.layer_dims) - 2:
                if i == 0:
                    modules.append(SimNorm(8))
                else:
                    modules.append(model_utils.get_activation_func(cfg_network['dyn_model_mlp']['activation']))
                #modules.append(model_utils.get_activation_func(cfg_network['dyn_model_mlp']['activation']))
                #modules.append(SimNorm(8))
                modules.append(torch.nn.LayerNorm(self.layer_dims[i+1], eps=1e-03))

        self.encoder = nn.Sequential(*modules).to(device)

        # Prior
        self.prior_layer_dims = [obs_dim + action_dim] + cfg_network['dyn_model_mlp']['units'] + [self.latent_dim * 2]
        modules = []
        for i in range(len(self.prior_layer_dims) - 1):
            modules.append(nn.Linear(self.prior_layer_dims[i], self.prior_layer_dims[i + 1]))
            if i < len(self.prior_layer_dims) - 2:
                modules.append(model_utils.get_activation_func(cfg_network['dyn_model_mlp']['activation']))
                #modules.append(torch.nn.LayerNorm(self.layer_dims[i+1], eps=1e-03))

        self.prior = nn.Sequential(*modules).to(device)

        # Decoder
        self.decoder_layer_dims = [obs_dim + action_dim + self.latent_dim] + cfg_network['dyn_model_mlp']['units'] + [obs_dim * 2]
        modules = []
        for i in range(len(self.decoder_layer_dims) - 1):
            modules.append(nn.Linear(self.decoder_layer_dims[i], self.decoder_layer_dims[i + 1]))
            if i < len(self.decoder_layer_dims) - 2:
                modules.append(model_utils.get_activation_func(cfg_network['dyn_model_mlp']['activation']))
                #modules.append(torch.nn.LayerNorm(self.layer_dims[i+1], eps=1e-03))

        self.decoder = nn.Sequential(*modules).to(device)

        print("encoder", self.encoder)
        print("decoder", self.decoder)
        print("prior", self.prior)

        # Must be put outside in case of requires_grad=True and ensemble of models
        self.max_logvar = nn.Parameter(torch.ones(1, obs_dim) * 0.5, requires_grad=False).to(device)
        self.min_logvar = nn.Parameter(torch.ones(1, obs_dim) * -10, requires_grad=False).to(device)

    def forward(self, s, a, next_s, l = None):
        x = torch.cat([s, a], dim=-1)
        time_latent = l
        if self.recurrent:
            out, time_latent = self.gru(x, l)
            x = torch.cat([x, out], dim=-1)

        with torch.no_grad():
            enc_x = torch.cat([x, next_s], dim=-1)
            enc_h = self.encoder(enc_x)

        mean_epsilon, logvar_epsilon = enc_h.chunk(2, dim=-1)

        dec_x = torch.cat([x, mean_epsilon], dim=-1)
        h = self.decoder(dec_x)
        mean, logvar = h.chunk(2, dim=-1)

        return mean, logvar, time_latent

    def encode_decode(self, s, a, next_s, l = None):
        x = torch.cat([s, a], dim=-1)
        time_latent = l
        if self.recurrent:
            out, time_latent = self.gru(x, l)
            x = torch.cat([x, out], dim=-1)

        # get epsilon
        enc_x = torch.cat([x, next_s], dim=-1)
        enc_h = self.encoder(enc_x)
        mean_epsilon, logvar_epsilon = enc_h.chunk(2, dim=-1)
        std_epsilon = torch.exp(0.5 * logvar_epsilon)
        dist_epsilon = Normal(mean_epsilon, std_epsilon)
        epsilon = dist_epsilon.rsample()

        # get prior
        prior_h = self.prior(x)
        mean_prior, logvar_prior = prior_h.chunk(2, dim=-1)
        std_prior = torch.exp(0.5 * logvar_prior)
        dist_prior = Normal(mean_prior, std_prior)

        # get decoder
        dec_x = torch.cat([x, epsilon], dim=-1)
        h = self.decoder(dec_x)

        mean, logvar = h.chunk(2, dim=-1)
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        return mean, logvar, mean_epsilon, std_epsilon, mean_prior, std_prior


class StochSSMCor(nn.Module):
    def __init__(self, obs_dim, action_dim, cfg_network, device='cuda:0'):
        super().__init__()

        self.device = device

        self.enc_layer_dims = [obs_dim + action_dim + obs_dim] + cfg_network['dyn_enc_model_mlp']['units'] + [cfg_network['dyn_enc_model_mlp']['last_units']]
        self.dec_layer_dims = [obs_dim + action_dim + cfg_network['dyn_enc_model_mlp']['last_units']] + cfg_network['dyn_dec_model_mlp']['units'] + [obs_dim * 2]

        enc_modules = []
        for i in range(len(self.enc_layer_dims) - 1):
            enc_modules.append(nn.Linear(self.enc_layer_dims[i], self.enc_layer_dims[i + 1]))
            if i < len(self.enc_layer_dims) - 2:
                enc_modules.append(model_utils.get_activation_func(cfg_network['dyn_enc_model_mlp']['activation']))
                enc_modules.append(torch.nn.LayerNorm(self.enc_layer_dims[i+1], eps=1e-03))

        self.dyn_enc_model = nn.Sequential(*enc_modules).to(device)

        dec_modules = []
        for i in range(len(self.dec_layer_dims) - 1):
            dec_modules.append(nn.Linear(self.dec_layer_dims[i], self.dec_layer_dims[i + 1]))
            if i < len(self.dec_layer_dims) - 2:
                dec_modules.append(model_utils.get_activation_func(cfg_network['dyn_dec_model_mlp']['activation']))
                dec_modules.append(torch.nn.LayerNorm(self.dec_layer_dims[i+1], eps=1e-03))

        self.dyn_dec_model = nn.Sequential(*dec_modules).to(device)

        # Must be put outside in case of requires_grad=True and ensemble of models
        self.max_logvar = nn.Parameter(torch.ones(1, obs_dim) * 0.5, requires_grad=False).to(device)
        self.min_logvar = nn.Parameter(torch.ones(1, obs_dim) * -10, requires_grad=False).to(device)

        print(self.dyn_enc_model)
        print(self.dyn_dec_model)

    def encode(self, s, a, next_s):
        x = torch.cat([s, a, next_s], dim=-1)
        h = self.dyn_enc_model(x)
        return h

    def forward(self, s, a, l):
        x = torch.cat([s, a, l], dim=-1)

        h = self.dyn_dec_model(x)

        mean, logvar = h.chunk(2, dim=-1)
        # Differentiable clip, funny! A.1 in https://arxiv.org/pdf/1805.12114
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        return mean, logvar

class DynamicsFunction(Function):
    @staticmethod
    def forward(ctx, state, action, model, envs, obs_rms, env_type):
        # Save state, action, and model for backward
        ctx.save_for_backward(state, action)
        ctx.model = model
        ctx.obs_rms = obs_rms

        if env_type == "dflex":
            with torch.no_grad():
                next_obs, rew, done, extra_info = envs.neurodiff_step(torch.tanh(action.detach()))

            # Here we must wire the real next obs into the backprop graph, which next_obs isn't in case of last obs of the trajectory
            # because next_obs is the obs after the env was reset
            real_next_obs = next_obs.clone()
            if(done.any()):
                done_idx = torch.argwhere(done).squeeze()
                real_next_obs[done_idx] = extra_info['obs_before_reset'][done_idx]

        elif env_type == "isaac_gym":
            with torch.no_grad():
                next_obs, rew, done, extra_info = envs.step(torch.tanh(action.detach()))
            done = extra_info['dones']
            next_obs = envs.dyn_obs_buf.clone()
            real_next_obs = next_obs.clone()
            if(done.any()):
                done_idx = torch.argwhere(done).squeeze()
                real_next_obs[done_idx] = extra_info['obs_before_reset'][done_idx]
        else:
            raise ValueError(
                f"env type {env_type} is not supported."
            )

        return real_next_obs, next_obs, rew, done, extra_info

    @staticmethod
    def backward(ctx, grad_real_next_obs, grad_next_obs=None, grad_rewards=None, grad_terminations=None, grad_infos=None):
        state, action = ctx.saved_tensors
        model = ctx.model
        obs_rms = ctx.obs_rms

        # Compute the gradient of the dynamics model with respect to action and state
        with torch.enable_grad():
            next_state_pred_delta, logvar, _ = model(obs_rms.normalize(state), torch.tanh(action))
            #variance = torch.exp(logvar)
            next_state_pred = next_state_pred_delta + state # get rid of tanh?

            #grad_real_next_obs = grad_real_next_obs / (1 + variance)

            if state.requires_grad:
                grad_state, = torch.autograd.grad(
                    next_state_pred, state, grad_outputs=grad_real_next_obs, retain_graph=True
                )
            else:
                grad_state = None
            grad_action, = torch.autograd.grad(
                next_state_pred, action, grad_outputs=grad_real_next_obs
            )

        # We only return gradients for state and action
        return grad_state, grad_action, None, None, None, None

class RewardsFunction(Function):
    @staticmethod
    def forward(ctx, state, action, reward, model, obs_rms):
        # Save state, action, and model for backward
        ctx.save_for_backward(state, action)
        ctx.model = model
        ctx.obs_rms = obs_rms

        return reward

    @staticmethod
    def backward(ctx, grad_reward):
        state, action = ctx.saved_tensors
        model = ctx.model
        obs_rms = ctx.obs_rms

        # Compute the gradient of the dynamics model with respect to action and state
        with torch.enable_grad():
            reward_pred = model.reward(obs_rms.normalize(state), torch.tanh(action)).squeeze(-1) # get rid of tanh?

            if state.requires_grad:
                grad_state, = torch.autograd.grad(
                    reward_pred, state, grad_outputs=grad_reward, retain_graph=True
                )
            else:
                grad_state = None
            grad_action, = torch.autograd.grad(
                reward_pred, action, grad_outputs=grad_reward
            )

        # We only return gradients for state and action
        return grad_state, grad_action, None, None, None

class DynamicsFunctionCor(Function):
    @staticmethod
    def forward(ctx, state, action, model, envs, obs_rms):
        # Save state, action, and model for backward
        ctx.model = model
        ctx.obs_rms = obs_rms

        next_obs, rew, done, extra_info = envs.neurodiff_step(torch.tanh(action.detach()))

        ctx.save_for_backward(state, action, next_obs)
        # Here we must wire the real next obs into the backprop graph, which next_obs isn't in case of last obs of the trajectory
        # because next_obs is the obs after the env was reset
        real_next_obs = next_obs.clone()
        if(done.any()):
            done_idx = torch.argwhere(done).squeeze()
            real_next_obs[done_idx] = extra_info['obs_before_reset'][done_idx]

        return real_next_obs, next_obs, rew, done, extra_info

    @staticmethod
    def backward(ctx, grad_real_next_obs, grad_next_obs=None, grad_rewards=None, grad_terminations=None, grad_infos=None):
        state, action, next_obs = ctx.saved_tensors
        model = ctx.model
        obs_rms = ctx.obs_rms

        # Compute the gradient of the dynamics model with respect to action and state
        with torch.no_grad():
            latent = model.encode(obs_rms.normalize(state), torch.tanh(action), obs_rms.normalize(next_obs))

        with torch.enable_grad():
            next_state_pred = model(obs_rms.normalize(state), torch.tanh(action), latent)[0] + state # get rid of tanh?

            if state.requires_grad:
                grad_state, = torch.autograd.grad(
                    next_state_pred, state, grad_outputs=grad_real_next_obs, retain_graph=True
                )
            else:
                grad_state = None
            grad_action, = torch.autograd.grad(
                next_state_pred, action, grad_outputs=grad_real_next_obs
              )

        # We only return gradients for state and action
        return grad_state, grad_action, None, None, None

class DynamicsFunctionPMO(Function):
    @staticmethod
    def forward(ctx, state, action, next_state, model, obs_rms):
        # Save state, action, and model for backward
        ctx.save_for_backward(state, action)
        ctx.model = model
        ctx.obs_rms = obs_rms

        return next_state

    @staticmethod
    def backward(ctx, grad_next_state):
        state, action = ctx.saved_tensors
        model = ctx.model
        obs_rms = ctx.obs_rms

        # Compute the gradient of the dynamics model with respect to action and state
        with torch.enable_grad():
            if obs_rms is not None:
                next_state_pred = model(obs_rms.normalize(state), torch.tanh(action))[0] + state # get rid of tanh?
            else:
                next_state_pred = model(state, torch.tanh(action))[0] + state

            if state.requires_grad:
                grad_state, = torch.autograd.grad(
                    next_state_pred, state, grad_outputs=grad_next_state, retain_graph=True, create_graph=True ###
                )
            else:
                grad_state = None
            grad_action, = torch.autograd.grad(
                next_state_pred, action, grad_outputs=grad_next_state, retain_graph=True, create_graph=True ###
            )

        # We only return gradients for state and action
        return grad_state, grad_action, None, None, None

class GradientExtractorFunction(Function):
    @staticmethod
    def forward(ctx, action, action_gradients_ptr):
        ctx.action_gradients_ptr = action_gradients_ptr

        return action

    @staticmethod
    def backward(ctx, grad_action):
        action_gradients_ptr = ctx.action_gradients_ptr
        action_gradients_ptr[:] = grad_action.clone()

        return grad_action, None

class GradientBranchingFunction(Function):
    @staticmethod
    def forward(ctx, action, action_gradients_ptr):
        ctx.action_gradients_ptr = action_gradients_ptr

        return torch.sum(action ** 2, dim = -1)

    @staticmethod
    def backward(ctx, _):
        action_gradients_ptr = ctx.action_gradients_ptr

        return action_gradients_ptr, None

class GradientAnalysorFunction(Function):
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        print(grad_output[:, 65:72])
        if grad_output.isnan().any():
            print("GradientAnalysorFunction NaN in output gradient")
            #exit(0)

        return grad_output, None

class GradientSwapingFunction(Function):
    @staticmethod
    def forward(ctx, img_state, true_state):
        return true_state.clone()

    @staticmethod
    def backward(ctx, grad_true_next_state):
        return grad_true_next_state, None

