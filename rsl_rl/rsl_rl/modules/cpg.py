#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

"""
Central Pattern Generator (CPG) module using Hopf oscillators.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HopfOscillator(nn.Module):
    """First-order Hopf oscillator (kept for compatibility, unused by default)."""

    def __init__(self, dt: float = 0.02, device: str = "cpu"):
        super().__init__()
        self.dt = dt
        self.device = device
        
    def forward(
        self,
        state: torch.Tensor,
        mu: torch.Tensor,
        frequency: torch.Tensor,
        coupling: torch.Tensor = None,
    ) -> torch.Tensor:
        x = state[..., 0]
        y = state[..., 1]
        r_squared = x ** 2 + y ** 2
        omega = 2 * torch.pi * frequency

        # print("Dimensions mu:",  mu.shape)

        # exit()
        dx = mu * (mu - r_squared) * x - omega * y
        dy = mu * (mu - r_squared) * y + omega * x
        if coupling is not None:
            dx = dx + coupling
        x_new = x + dx * self.dt
        y_new = y + dy * self.dt
        return torch.stack([x_new, y_new], dim=-1)


class SecondOrderOscillator(nn.Module):
    """Second-order CPG oscillator with explicit acceleration on amplitude."""

    def __init__(self, dt: float = 0.02, sub_dt: float = 0.001, device: str = "cpu"):
        super().__init__()
        self.dt = dt
        self.sub_dt = sub_dt
        self.device = device
        # Gain for critically damped convergence toward sqrt(mu)
        self._a = 150.0
        self.register_buffer("two_pi", torch.tensor(2.0 * torch.pi, device=device))

    def forward(
        self,
        state: torch.Tensor,
        velocity: torch.Tensor,
        mu: torch.Tensor,
        frequency: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate amplitude/phase with a second-order system.

        state: [batch, num_osc, 2] -> [r, theta]
        velocity: [batch, num_osc, 2] -> [r_dot, theta_dot]
        mu: target amplitude (>=0), [batch, num_osc]
        frequency: Hz, [batch, num_osc]
        """
        r = state[..., 0]
        theta = state[..., 1]
        r_dot = velocity[..., 0]
        theta_dot = velocity[..., 1]

        # Target radius follows sqrt(mu)
        target_r = torch.sqrt(torch.clamp(mu, min=0.0))
        # Constant angular velocity from frequency
        theta_dot = self.two_pi * frequency

        # Sub-steps for stability
        n_substeps = max(1, int(self.dt / self.sub_dt))
        sub_dt = self.dt / n_substeps

        a = self._a
        for _ in range(n_substeps):
            d2r = a * (0.25 * a * (target_r - r) - r_dot)
            r_dot = r_dot + d2r * sub_dt
            r = r + r_dot * sub_dt
            theta = theta + theta_dot * sub_dt
        theta = torch.remainder(theta, self.two_pi)

        new_state = torch.stack([r, theta], dim=-1)
        new_velocity = torch.stack([r_dot, theta_dot], dim=-1)
        return new_state, new_velocity


class CPGNetwork(nn.Module):
    """
    CPG network for quadruped locomotion with 12 oscillators (3 per leg).
    
    Each leg has 3 oscillators (hip, thigh, calf). Coupling is disabled.
    The RL policy outputs mu, frequency, and offset for each oscillator.
    """
    
    # Constants
    OSCILLATOR_STATE_DIM = 2  # [x, y] coordinates
    INITIAL_STATE_SCALE = 0.1  # Small random values to break symmetry
    
    def __init__(
        self,
        num_envs: int,
        num_legs: int = 4,
        num_joints_per_leg: int = 3,
        dt: float = 0.02,
        coupling_strength: float = 0.0,
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
        
        # Use second-order oscillator for better stability and expressiveness
        self.oscillator = SecondOrderOscillator(dt=dt, sub_dt=0.001, device=device)

        # Initialize oscillator states/velocities: [num_envs, num_oscillators, 2]
        self.register_buffer(
            "oscillator_states",
            self.INITIAL_STATE_SCALE * torch.randn(
                num_envs, self.num_oscillators, self.OSCILLATOR_STATE_DIM, device=device
            ),
            persistent=False,
        )
        self.register_buffer(
            "oscillator_velocities",
            torch.zeros(num_envs, self.num_oscillators, self.OSCILLATOR_STATE_DIM, device=device),
            persistent=False,
        )
        
    def reset(self, env_ids: torch.Tensor = None):
        """
        Reset oscillator states for specified environments.
        
        Args:
            env_ids: Environment indices to reset. If None, reset all.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        # Reset to small random values
        self.oscillator_states[env_ids] = self.INITIAL_STATE_SCALE * torch.randn(
            len(env_ids), self.num_oscillators, self.OSCILLATOR_STATE_DIM, device=self.device
        )
        self.oscillator_velocities[env_ids] = torch.zeros(
            len(env_ids), self.num_oscillators, self.OSCILLATOR_STATE_DIM, device=self.device
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
        # print("Dimensions mu:", mu.shape)
        
        # Update oscillator states (no coupling)
        self.oscillator_states, self.oscillator_velocities = self.oscillator(
            self.oscillator_states,
            self.oscillator_velocities,
            mu,
            frequency,
            # self.coupling_strength
        )
        
        # Generate joint commands: r * cos(theta) + offset
        r = self.oscillator_states[..., 0]
        theta = self.oscillator_states[..., 1]
        joint_commands = r * torch.cos(theta) + offset
        
        return joint_commands
