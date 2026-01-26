#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

"""
Central Pattern Generator (CPG) module using Hopf oscillators.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HopfOscillator(nn.Module):
    """
    Hopf oscillator for CPG implementation.
    
    The Hopf oscillator generates rhythmic patterns using the following equations:
    dx/dt = mu * (mu - r^2) * x - omega * y
    dy/dt = mu * (mu - r^2) * y + omega * x
    
    where r^2 = x^2 + y^2, and omega = 2 * pi * frequency
    
    For locomotion, we use the x component as the output signal.
    """
    
    def __init__(self, dt: float = 0.02, device: str = "cpu"):
        """
        Args:
            dt: Time step for integration (default: 0.02s for 50Hz control)
            device: Device to run computations on
        """
        super().__init__()
        self.dt = dt
        self.device = device
        
    def forward(
        self,
        state: torch.Tensor,
        mu: torch.Tensor,
        frequency: torch.Tensor,
        coupling: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Integrate the Hopf oscillator for one time step.
        
        Args:
            state: Current oscillator state [batch_size, num_oscillators, 2] where 2 = [x, y]
            mu: Amplitude parameter [batch_size, num_oscillators]
            frequency: Frequency parameter in Hz [batch_size, num_oscillators]
            coupling: Optional coupling from other oscillators [batch_size, num_oscillators]
            
        Returns:
            New state [batch_size, num_oscillators, 2]
        """
        x = state[..., 0]  # [batch_size, num_oscillators]
        y = state[..., 1]  # [batch_size, num_oscillators]
        
        # Compute radius squared
        r_squared = x ** 2 + y ** 2
        
        # Angular frequency
        omega = 2 * torch.pi * frequency
        
        # Hopf oscillator dynamics
        dx = mu * (mu - r_squared) * x - omega * y
        dy = mu * (mu - r_squared) * y + omega * x
        
        # Add coupling if provided
        if coupling is not None:
            dx = dx + coupling
        
        # Euler integration
        x_new = x + dx * self.dt
        y_new = y + dy * self.dt
        
        # Stack back into state
        state_new = torch.stack([x_new, y_new], dim=-1)
        
        return state_new


class CPGNetwork(nn.Module):
    """
    CPG network for quadruped locomotion with 12 oscillators (3 per leg).
    
    Each leg has 3 oscillators (hip, thigh, calf) with intra-leg coupling.
    The RL policy outputs mu, phi (phase offset), and offset for each oscillator.
    """
    
    def __init__(
        self,
        num_envs: int,
        num_legs: int = 4,
        num_joints_per_leg: int = 3,
        dt: float = 0.02,
        coupling_strength: float = 1.0,
        device: str = "cpu",
    ):
        """
        Args:
            num_envs: Number of parallel environments
            num_legs: Number of legs (default: 4 for quadruped)
            num_joints_per_leg: Number of joints per leg (default: 3 for hip, thigh, calf)
            dt: Time step for integration
            coupling_strength: Strength of intra-leg coupling
            device: Device to run computations on
        """
        super().__init__()
        
        self.num_envs = num_envs
        self.num_legs = num_legs
        self.num_joints_per_leg = num_joints_per_leg
        self.num_oscillators = num_legs * num_joints_per_leg  # 12 for quadruped
        self.coupling_strength = coupling_strength
        self.device = device
        
        # Hopf oscillator
        self.oscillator = HopfOscillator(dt=dt, device=device)
        
        # Initialize oscillator states: [num_envs, num_oscillators, 2]
        # Start with small random values to break symmetry
        self.register_buffer(
            "oscillator_states",
            0.1 * torch.randn(num_envs, self.num_oscillators, 2, device=device)
        )
        
        # Coupling matrix: oscillators on the same leg are coupled
        # Shape: [num_oscillators, num_oscillators]
        coupling_matrix = torch.zeros(self.num_oscillators, self.num_oscillators, device=device)
        
        for leg_idx in range(num_legs):
            # Get indices for this leg's oscillators
            start_idx = leg_idx * num_joints_per_leg
            end_idx = start_idx + num_joints_per_leg
            
            # Couple all oscillators within this leg
            for i in range(start_idx, end_idx):
                for j in range(start_idx, end_idx):
                    if i != j:
                        coupling_matrix[i, j] = 1.0
        
        self.register_buffer("coupling_matrix", coupling_matrix)
        
    def reset(self, env_ids: torch.Tensor = None):
        """
        Reset oscillator states for specified environments.
        
        Args:
            env_ids: Environment indices to reset. If None, reset all.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        # Reset to small random values
        self.oscillator_states[env_ids] = 0.1 * torch.randn(
            len(env_ids), self.num_oscillators, 2, device=self.device
        )
    
    def forward(
        self,
        mu: torch.Tensor,
        frequency: torch.Tensor,
        offset: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute joint commands from CPG parameters.
        
        Args:
            mu: Amplitude parameter [num_envs, num_oscillators]
            frequency: Frequency parameter in Hz [num_envs, num_oscillators]
            offset: Offset parameter [num_envs, num_oscillators]
            
        Returns:
            Joint commands [num_envs, num_oscillators]
        """
        # Compute coupling influence
        # For each oscillator, sum the states of coupled oscillators
        x_values = self.oscillator_states[..., 0]  # [num_envs, num_oscillators]
        
        # Matrix multiply to get coupling: [num_envs, num_oscillators]
        coupling = torch.matmul(x_values, self.coupling_matrix.T) * self.coupling_strength
        
        # Update oscillator states
        self.oscillator_states = self.oscillator(
            self.oscillator_states,
            mu,
            frequency,
            coupling
        )
        
        # Generate joint commands: amplitude * oscillator_output + offset
        # Use x component of oscillator state
        joint_commands = mu * self.oscillator_states[..., 0] + offset
        
        return joint_commands
