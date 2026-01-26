"""
Unit tests for CPG modules.
"""

import torch
import pytest
from rsl_rl.modules.cpg import HopfOscillator, CPGNetwork
from rsl_rl.modules.actor_critic_cpg import ActorCriticCPG


def test_hopf_oscillator():
    """Test Hopf oscillator basic functionality."""
    device = "cpu"
    dt = 0.02
    batch_size = 4
    num_oscillators = 12
    
    oscillator = HopfOscillator(dt=dt, device=device)
    
    # Initialize state
    state = torch.randn(batch_size, num_oscillators, 2, device=device)
    mu = torch.ones(batch_size, num_oscillators, device=device) * 0.5
    frequency = torch.ones(batch_size, num_oscillators, device=device) * 2.0  # 2 Hz
    
    # Run one step
    new_state = oscillator(state, mu, frequency)
    
    # Check output shape
    assert new_state.shape == (batch_size, num_oscillators, 2)
    
    # Check that state is updated (not identical to input)
    assert not torch.allclose(state, new_state)
    
    print("✓ Hopf oscillator test passed")


def test_cpg_network():
    """Test CPG network with leg coupling."""
    device = "cpu"
    num_envs = 4
    num_legs = 4
    num_joints_per_leg = 3
    
    cpg = CPGNetwork(
        num_envs=num_envs,
        num_legs=num_legs,
        num_joints_per_leg=num_joints_per_leg,
        dt=0.02,
        coupling_strength=1.0,
        device=device,
    )
    
    # Check initialization
    assert cpg.oscillator_states.shape == (num_envs, 12, 2)
    assert cpg.coupling_matrix.shape == (12, 12)
    
    # Check that coupling matrix has correct structure
    # Oscillators on the same leg should be coupled
    for leg_idx in range(num_legs):
        start_idx = leg_idx * num_joints_per_leg
        end_idx = start_idx + num_joints_per_leg
        for i in range(start_idx, end_idx):
            for j in range(start_idx, end_idx):
                if i != j:
                    assert cpg.coupling_matrix[i, j] == 1.0
                else:
                    assert cpg.coupling_matrix[i, j] == 0.0
    
    # Oscillators on different legs should not be coupled
    assert cpg.coupling_matrix[0, 3] == 0.0  # leg 0 to leg 1
    assert cpg.coupling_matrix[0, 6] == 0.0  # leg 0 to leg 2
    
    # Test forward pass
    mu = torch.ones(num_envs, 12, device=device) * 0.5
    frequency = torch.ones(num_envs, 12, device=device) * 2.0
    offset = torch.zeros(num_envs, 12, device=device)
    
    joint_commands = cpg(mu, frequency, offset)
    
    # Check output shape
    assert joint_commands.shape == (num_envs, 12)
    
    # Test reset
    initial_state = cpg.oscillator_states.clone()
    cpg.reset(torch.tensor([0, 1], device=device))
    # States for env 0 and 1 should be different
    assert not torch.allclose(initial_state[0], cpg.oscillator_states[0])
    assert not torch.allclose(initial_state[1], cpg.oscillator_states[1])
    # States for env 2 and 3 should be unchanged
    assert torch.allclose(initial_state[2], cpg.oscillator_states[2])
    assert torch.allclose(initial_state[3], cpg.oscillator_states[3])
    
    print("✓ CPG network test passed")


def test_actor_critic_cpg():
    """Test ActorCriticCPG initialization and forward pass."""
    device = "cpu"
    num_envs = 4
    num_obs = 48
    num_actions = 12
    
    # Test with CPG enabled
    cpg_config = {
        "num_envs": num_envs,
        "dt": 0.02,
        "coupling_strength": 1.0,
        "device": device,
        "frequency_range": (1.0, 3.0),
        "mu_range": (0.0, 1.0),
        "offset_range": (-0.5, 0.5),
    }
    
    actor_critic = ActorCriticCPG(
        num_actor_obs=num_obs,
        num_critic_obs=num_obs,
        num_actions=num_actions,
        enable_cpg=True,
        cpg_config=cpg_config,
    )
    
    # Check that actor outputs correct dimension
    # Should output 36 values (3 params * 12 oscillators)
    obs = torch.randn(num_envs, num_obs, device=device)
    actor_output = actor_critic.actor(obs)
    assert actor_output.shape == (num_envs, num_actions * 3)
    
    # Test act method
    actions = actor_critic.act(obs)
    assert actions.shape == (num_envs, num_actions * 3)  # CPG params
    
    # Test get_joint_commands
    joint_commands = actor_critic.get_joint_commands(actions)
    assert joint_commands.shape == (num_envs, num_actions)  # Joint commands
    
    # Test evaluate
    values = actor_critic.evaluate(obs)
    assert values.shape == (num_envs, 1)
    
    # Test reset with dones
    dones = torch.tensor([True, False, True, False], device=device)
    actor_critic.reset(dones)
    
    print("✓ ActorCriticCPG test passed")
    
    # Test with CPG disabled
    actor_critic_no_cpg = ActorCriticCPG(
        num_actor_obs=num_obs,
        num_critic_obs=num_obs,
        num_actions=num_actions,
        enable_cpg=False,
    )
    
    # Should output joint commands directly
    actor_output = actor_critic_no_cpg.actor(obs)
    assert actor_output.shape == (num_envs, num_actions)
    
    actions = actor_critic_no_cpg.act(obs)
    assert actions.shape == (num_envs, num_actions)
    
    joint_commands = actor_critic_no_cpg.get_joint_commands(actions)
    assert torch.allclose(joint_commands, actions)  # Should be identity
    
    print("✓ ActorCriticCPG (disabled) test passed")


if __name__ == "__main__":
    print("Running CPG module tests...")
    test_hopf_oscillator()
    test_cpg_network()
    test_actor_critic_cpg()
    print("\n✅ All tests passed!")
