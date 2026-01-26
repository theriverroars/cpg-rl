#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

"""
Actor-Critic module with CPG support.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules.actor_critic import get_activation
from rsl_rl.modules.cpg import CPGNetwork


class ActorCriticCPG(nn.Module):
    """
    Actor-Critic module with optional CPG support.
    
    When CPG is enabled:
    - Actor outputs CPG parameters: mu, frequency, offset for each oscillator
    - CPG network converts these to joint commands
    
    When CPG is disabled:
    - Acts as standard ActorCritic with direct joint commands
    """
    
    is_recurrent = False
    
    # Constants for CPG parameters
    NUM_CPG_PARAMS = 3  # mu, frequency, offset
    
    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        enable_cpg=False,
        cpg_config=None,
        **kwargs,
    ):
        """
        Args:
            num_actor_obs: Number of actor observations
            num_critic_obs: Number of critic observations
            num_actions: Number of actions (12 for quadruped)
            actor_hidden_dims: Hidden dimensions for actor MLP
            critic_hidden_dims: Hidden dimensions for critic MLP
            activation: Activation function name
            init_noise_std: Initial standard deviation for action noise
            enable_cpg: Whether to enable CPG mode
            cpg_config: CPG configuration dict with keys:
                - num_envs: Number of environments
                - dt: Time step for integration
                - coupling_strength: Intra-leg coupling strength
                - frequency_range: (min, max) frequency in Hz
                - mu_range: (min, max) amplitude
                - offset_range: (min, max) offset
            **kwargs: Additional arguments (ignored)
        """
        if kwargs:
            print(
                "ActorCriticCPG.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        
        self.enable_cpg = enable_cpg
        self.num_actions = num_actions
        
        activation = get_activation(activation)
        
        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
        
        # Determine actor output dimension
        if enable_cpg:
            # Output: mu, frequency, offset for each oscillator
            actor_output_dim = num_actions * self.NUM_CPG_PARAMS
            
            # CPG network
            if cpg_config is None:
                cpg_config = {
                    "num_envs": 1,
                    "dt": 0.02,
                    "coupling_strength": 1.0,
                }
            
            self.cpg_network = CPGNetwork(
                num_envs=cpg_config.get("num_envs", 1),
                num_legs=4,
                num_joints_per_leg=3,
                dt=cpg_config.get("dt", 0.02),
                coupling_strength=cpg_config.get("coupling_strength", 1.0),
                device=cpg_config.get("device", "cpu"),
            )
            
            # Parameter ranges for CPG
            self.frequency_range = cpg_config.get("frequency_range", (1.0, 3.0))
            self.mu_range = cpg_config.get("mu_range", (0.0, 1.0))
            self.offset_range = cpg_config.get("offset_range", (-0.5, 0.5))
            # Pre-compute offset scale for efficiency
            self.offset_scale = max(abs(self.offset_range[0]), abs(self.offset_range[1]))
            
        else:
            # Standard mode: direct joint commands
            actor_output_dim = num_actions
            self.cpg_network = None
        
        # Policy (Actor)
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_output_dim))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)
        
        # Value function (Critic)
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)
        
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f"CPG Enabled: {self.enable_cpg}")
        
        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(actor_output_dim))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
    
    def reset(self, dones=None):
        """Reset CPG states for environments that are done."""
        if self.enable_cpg and self.cpg_network is not None and dones is not None:
            # Get indices of done environments
            done_env_ids = dones.nonzero(as_tuple=False).flatten()
            if len(done_env_ids) > 0:
                self.cpg_network.reset(done_env_ids)
    
    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean
    
    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
    
    def process_cpg_params_to_joint_commands(self, actor_output: torch.Tensor) -> torch.Tensor:
        """
        Process actor output to generate final joint commands.
        
        If CPG is enabled, actor_output contains CPG parameters which are
        passed through the CPG network to generate joint commands.
        Otherwise, actor_output is used directly as joint commands.
        
        Args:
            actor_output: Raw output from actor network [batch_size, output_dim]
            
        Returns:
            joint_commands: Joint commands to send to the environment [batch_size, num_actions]
        """
        if self.enable_cpg:
            # Split actor output into mu, frequency, offset
            # actor_output shape: [batch_size, num_actions * NUM_CPG_PARAMS]
            batch_size = actor_output.shape[0]
            
            # Reshape to [batch_size, NUM_CPG_PARAMS, num_actions] and transpose to [batch_size, num_actions, NUM_CPG_PARAMS]
            params = actor_output.view(batch_size, self.NUM_CPG_PARAMS, self.num_actions).transpose(1, 2)
            
            # Extract parameters and apply ranges
            mu_raw = params[..., 0]  # [batch_size, num_actions]
            freq_raw = params[..., 1]
            offset_raw = params[..., 2]
            
            # Apply sigmoid/tanh to bound parameters
            mu = torch.sigmoid(mu_raw) * (self.mu_range[1] - self.mu_range[0]) + self.mu_range[0]
            frequency = torch.sigmoid(freq_raw) * (self.frequency_range[1] - self.frequency_range[0]) + self.frequency_range[0]
            offset = torch.tanh(offset_raw) * self.offset_scale
            
            # Generate joint commands via CPG
            joint_commands = self.cpg_network(mu, frequency, offset)
            
        else:
            # Direct joint commands
            joint_commands = actor_output
        
        return joint_commands
    
    def update_distribution(self, observations):
        """Update action distribution based on observations."""
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean * 0.0 + self.std)
    
    def act(self, observations, **kwargs):
        """
        Sample actions from the current policy.
        
        Returns CPG parameters when CPG is enabled (for storage in rollout buffer),
        or joint commands when CPG is disabled.
        """
        self.update_distribution(observations)
        
        # Sample from distribution
        actions = self.distribution.sample()
        
        return actions
    
    def get_joint_commands(self, cpg_params_or_actions: torch.Tensor) -> torch.Tensor:
        """
        Convert actions to joint commands.
        
        When CPG is enabled, this converts CPG parameters to joint commands.
        When CPG is disabled, this is an identity operation.
        
        Args:
            cpg_params_or_actions: CPG parameters (if CPG enabled) or joint commands (if disabled)
            
        Returns:
            joint_commands: Joint commands to send to the environment [batch_size, num_actions]
        """
        if self.enable_cpg:
            return self.process_cpg_params_to_joint_commands(cpg_params_or_actions)
        else:
            return cpg_params_or_actions
    
    def get_actions_log_prob(self, actions):
        """
        Get log probability of actions.
        
        Note: When CPG is enabled, this returns log prob of CPG parameters,
        not the final joint commands. This is correct for PPO training.
        """
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_inference(self, observations):
        """Deterministic action selection for inference/testing."""
        actor_output = self.actor(observations)
        
        if self.enable_cpg:
            # Use mean CPG parameters (no sampling)
            actions = self.process_cpg_params_to_joint_commands(actor_output)
        else:
            actions = actor_output
        
        return actions
    
    def evaluate(self, critic_observations, **kwargs):
        """Evaluate value function."""
        value = self.critic(critic_observations)
        return value
