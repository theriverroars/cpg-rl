#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

"""
Environment wrapper for CPG-based control.
"""

import torch
from rsl_rl.env import VecEnv


class CPGVecEnvWrapper(VecEnv):
    """
    Wrapper for vectorized environments to support CPG-based control.
    
    When CPG is enabled, the policy outputs CPG parameters (mu, frequency, offset)
    for each oscillator. This wrapper converts those parameters to joint commands
    before passing to the underlying environment.
    """
    
    def __init__(self, env: VecEnv, actor_critic):
        """
        Args:
            env: The vectorized environment to wrap
            actor_critic: The actor-critic module (must be ActorCriticCPG with CPG enabled)
        """
        self.env = env
        self.actor_critic = actor_critic
        
        # Check if CPG is enabled
        if not hasattr(actor_critic, 'enable_cpg') or not actor_critic.enable_cpg:
            raise ValueError("CPGVecEnvWrapper requires ActorCriticCPG with enable_cpg=True")
        
        if not hasattr(actor_critic, 'cpg_network') or actor_critic.cpg_network is None:
            raise ValueError("CPGVecEnvWrapper requires actor_critic to have cpg_network")
    
    def __getattr__(self, name):
        """Forward attribute access to the wrapped environment."""
        if name.startswith('_'):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        return getattr(self.env, name)
    
    def step(self, cpg_params: torch.Tensor):
        """
        Step the environment with CPG parameters.
        
        Args:
            cpg_params: CPG parameters [num_envs, num_oscillators * 3]
            
        Returns:
            observations, rewards, dones, infos
        """
        # Convert CPG parameters to joint commands
        joint_commands = self.actor_critic._process_actor_output(cpg_params)
        
        # Step the environment with joint commands
        return self.env.step(joint_commands)
    
    def reset(self):
        """Reset the environment."""
        return self.env.reset()
    
    def get_observations(self):
        """Get current observations."""
        return self.env.get_observations()
    
    def close(self):
        """Close the environment."""
        return self.env.close()
