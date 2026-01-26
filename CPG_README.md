# CPG (Central Pattern Generator) with Hopf Oscillators

This implementation adds CPG functionality to the locomotion training using Hopf oscillators.

## Overview

The CPG system uses Hopf oscillators to generate rhythmic patterns for quadruped locomotion. Instead of directly outputting joint positions, the RL policy learns to output CPG parameters (amplitude, frequency, and offset) for each joint's oscillator.

### Key Features

- **12 Hopf Oscillators**: One for each joint (4 legs × 3 joints per leg)
- **Intra-leg Coupling**: Oscillators on the same leg are coupled to coordinate movement
- **RL-Controlled Parameters**: The policy outputs `mu` (amplitude), `frequency`, and `offset` for each oscillator
- **Switchable**: Can easily toggle between CPG mode and standard direct control

## Architecture

### Hopf Oscillator

Each oscillator follows the Hopf equations:
```
dx/dt = mu * (mu - r²) * x - omega * y
dy/dt = mu * (mu - r²) * y + omega * x
```

where:
- `r² = x² + y²`
- `omega = 2π * frequency`
- The x component is used as the output signal

### CPG Network

- **12 oscillators** organized in 4 legs with 3 joints each
- **Coupling matrix**: Only oscillators on the same leg are coupled
- **Output**: `joint_command = mu * oscillator_x + offset`

### Actor-Critic with CPG

When CPG is enabled:
- **Actor output**: 36 values (3 parameters × 12 oscillators)
  - `mu`: Amplitude range [0.0, 1.0]
  - `frequency`: Frequency range [1.0, 3.0] Hz
  - `offset`: Offset range [-0.5, 0.5]
- **Storage**: CPG parameters are stored in rollout buffer
- **Environment**: Joint commands (12 values) are sent to the robot

## Usage

### Training with CPG

```bash
python scripts/train.py --task=go2_base --enable_cpg --run_name=cpg_test --headless
```

### Training without CPG (standard mode)

```bash
python scripts/train.py --task=go2_base --run_name=standard_test --headless
```

### Playing/Testing

```bash
python scripts/play.py --task=go2_base_play --enable_cpg --load_run=cpg_test
```

## Configuration

CPG parameters can be configured in the runner. Default values:
- `dt`: 0.02s (50Hz control frequency)
- `coupling_strength`: 1.0
- `frequency_range`: (1.0, 3.0) Hz
- `mu_range`: (0.0, 1.0)
- `offset_range`: (-0.5, 0.5)

## Implementation Details

### Files Modified
- `rsl_rl/rsl_rl/modules/cpg.py`: Hopf oscillator and CPG network implementation
- `rsl_rl/rsl_rl/modules/actor_critic_cpg.py`: CPG-augmented actor-critic
- `rsl_rl/rsl_rl/algorithms/ppo.py`: Updated to convert CPG params to joint commands
- `rsl_rl/rsl_rl/runners/on_policy_runner.py`: CPG configuration handling
- `scripts/train.py`: Added `--enable_cpg` flag
- `scripts/play.py`: Added `--enable_cpg` flag
- `scripts/cli_args.py`: CPG configuration logic

### Training Flow

1. **Observation** → **Actor Network** → **CPG Parameters** (36 dims)
2. **CPG Parameters** stored in rollout buffer for PPO updates
3. **CPG Parameters** → **CPG Network** → **Joint Commands** (12 dims)
4. **Joint Commands** → **Environment** → **Reward**

### Advantages of CPG

- **Structured Exploration**: Oscillators naturally produce rhythmic gaits
- **Reduced Action Space Complexity**: Learning CPG parameters instead of raw joint commands
- **Biological Inspiration**: Mimics how animals generate locomotion
- **Smooth Movements**: Oscillators produce continuous, smooth trajectories

## Testing

Run the unit tests to verify CPG functionality:

```bash
# Install dependencies
pip install torch pytest numpy gitpython

# Run tests
cd /home/runner/work/cpg-rl/cpg-rl
PYTHONPATH=rsl_rl:$PYTHONPATH python3 tests/test_cpg.py
```

## Future Enhancements

Potential improvements:
- Inter-leg phase relationships for different gaits (trot, gallop, etc.)
- Adaptive coupling strength
- Sensory feedback integration into oscillator dynamics
- Terrain-adaptive frequency modulation
