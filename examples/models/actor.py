# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np

from models import model_utils


class ActorDeterministicMLP(nn.Module):
    def __init__(self, obs_dim, action_dim, cfg_network, device='cuda:0'):
        super(ActorDeterministicMLP, self).__init__()

        self.device = device

        self.layer_dims = [obs_dim] + cfg_network['actor_mlp']['units'] + [action_dim]

        init_ = lambda m: model_utils.init(m, nn.init.orthogonal_, lambda x: nn.init.
                        constant_(x, 0), np.sqrt(2))

        modules = []
        for i in range(len(self.layer_dims) - 1):
            modules.append(init_(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1])))
            if i < len(self.layer_dims) - 2:
                modules.append(model_utils.get_activation_func(cfg_network['actor_mlp']['activation']))
                modules.append(torch.nn.LayerNorm(self.layer_dims[i+1]))

        self.actor = nn.Sequential(*modules).to(device)
        
        self.action_dim = action_dim
        self.obs_dim = obs_dim

        print(self.actor)

    def get_logstd(self):
        # return self.logstd
        return None

    def forward(self, observations, deterministic = False):
        return self.actor(observations)


class ActorStochasticMLP(nn.Module):
    def __init__(self, obs_dim, action_dim, cfg_network, device='cuda:0'):
        super(ActorStochasticMLP, self).__init__()

        self.device = device

        self.recurrent = cfg_network["actor_mlp"].get("recurrent", False)
        if self.recurrent:
            self.hidden_size = int(cfg_network["actor_mlp"].get("hidden_size", 128))
            self.gru = nn.GRU(obs_dim, self.hidden_size, batch_first=True).to(device)
            self.layer_dims = [obs_dim + self.hidden_size] + cfg_network['actor_mlp']['units'] + [action_dim]
        else:
            self.layer_dims = [obs_dim] + cfg_network['actor_mlp']['units'] + [action_dim]

        init_ = lambda m: model_utils.init(m, nn.init.orthogonal_, lambda x: nn.init.
                        constant_(x, 0), np.sqrt(2))

        modules = []
        for i in range(len(self.layer_dims) - 1):
            modules.append(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1]))
            if i < len(self.layer_dims) - 2:
                modules.append(model_utils.get_activation_func(cfg_network['actor_mlp']['activation']))
                modules.append(torch.nn.LayerNorm(self.layer_dims[i+1]))
            else:
                modules.append(model_utils.get_activation_func('identity'))

        self.mu_net = nn.Sequential(*modules).to(device)

        logstd = cfg_network.get('actor_logstd_init', -1.0)

        self.logstd = torch.nn.Parameter(torch.ones(action_dim, dtype=torch.float32, device=device) * logstd)

        self.action_dim = action_dim
        self.obs_dim = obs_dim

        self.min_logstd = -1.427

        print(self.mu_net)
        print(self.logstd)

    def clamp_std(self):
        self.logstd.data = torch.clamp(self.logstd.data, self.min_logstd)

    def get_logstd(self):
        return self.logstd

    def forward(self, obs, deterministic = False, l = None):
        #self.clamp_std()
        x = obs
        time_latent = l
        if self.recurrent:
            out, time_latent = self.gru(obs.unsqueeze(1), l)
            x = torch.cat([obs.unsqueeze(1), out], dim=-1)
        mu = self.mu_net(x)

        if deterministic:
            if self.recurrent:
                return mu.squeeze(1), time_latent
            else:
                return mu, time_latent
        else:
            std = self.logstd.exp() # (num_actions)
            # eps = torch.randn((*obs.shape[:-1], std.shape[-1])).to(self.device)
            # sample = mu + eps * std
            dist = Normal(mu, std)
            sample = dist.rsample()
            if self.recurrent:
                return sample.squeeze(1), time_latent
            else:
                return sample, time_latent

    def forward_with_dist(self, obs, deterministic = False):
        mu = self.mu_net(obs)
        std = self.logstd.exp() # (num_actions)

        if deterministic:
            return mu, mu, std
        else:
            dist = Normal(mu, std)
            sample = dist.rsample()
            return sample, mu, std

    def forward_with_log_probs(self, obs, actions = None):
        mu = self.mu_net(obs)

        std = self.logstd.exp()
        dist = Normal(mu, std)
        sample = dist.rsample()

        if actions is None:
            return sample, dist.log_prob(sample).sum(-1), dist.entropy().sum(-1)
        else:
            return sample, dist.log_prob(actions).sum(-1), dist.entropy().sum(-1)

LOG_STD_MAX = 2
LOG_STD_MIN = -5
class SAPOActorStochasticMLP(nn.Module):
    def __init__(self):
        super(SAPOActorStochasticMLP, self).__init__()

        self.recurrent = False
        obs_dim = 52
        action_dim = 12

        if self.recurrent:
            self.hidden_size = 256
            self.gru = nn.GRU(obs_dim, self.hidden_size, batch_first=True)
            self.layer_dims = [obs_dim + self.hidden_size] + [256, 128, 64]
        else:
            self.layer_dims = [obs_dim] + [256, 128, 64]

        init_ = lambda m: model_utils.init(m, nn.init.orthogonal_, lambda x: nn.init.
                        constant_(x, 0), np.sqrt(2))

        modules = []
        for i in range(len(self.layer_dims) - 1):
            modules.append(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1]))
            modules.append(model_utils.get_activation_func('silu'))
            modules.append(torch.nn.LayerNorm(self.layer_dims[i+1]))

        self.mlp = nn.Sequential(*modules)
        self.fc_mean = nn.Linear(self.layer_dims[-1], action_dim)
        self.fc_logstd = nn.Linear(self.layer_dims[-1], action_dim)

        self.action_dim = action_dim
        self.obs_dim = obs_dim

        print(self.mlp)

    def forward(self, obs, l = None):
        #self.clamp_std()
        x = obs
        time_latent = l
        if self.recurrent:
            out, time_latent = self.gru(obs.unsqueeze(1), l)
            x = torch.cat([obs.unsqueeze(1), out], dim=-1)
        h = self.mlp(x)
        mean = self.fc_mean(h)

        log_std = self.fc_logstd(h)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

        std = log_std.exp()
        dist = Normal(mean, std)
        sample = dist.rsample()  # for reparameterization trick (mean + std * N(0,1))

        """# Compute entropy
        dim = mean.size(-1)
        ent = 0.5 * dim * (1 + torch.log(torch.tensor(2 * torch.pi))) + torch.sum(log_std, dim=-1)
        ent += torch.log((1 - torch.tanh(sample).pow(2)) + 1e-6).sum(-1)"""
        log_prob = dist.log_prob(sample)
        log_prob -= torch.log((1 - torch.tanh(sample).pow(2)) + 1e-6)
        log_prob = -log_prob.sum(-1)

        if self.recurrent:
            return sample.squeeze(1), log_prob.squeeze(1), time_latent
        else:
            return sample, log_prob, time_latent