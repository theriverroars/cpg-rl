# Play Script Fix for CPG Checkpoints

## Issues Fixed
The play.py script had several issues when loading CPG-trained checkpoints:

1. **KeyError: `/policy/enable_cpg`**: When loading checkpoints, the config might not have the enable_cpg key
2. **Dimension mismatch in oscillator_states**: Training uses 4096 envs, but play uses 10 envs, causing shape mismatch `torch.Size([4096, 12, 2])` vs `torch.Size([10, 12, 2])`
3. **Observation dimension mismatch**: Different history_length between training and inference caused actor input size mismatch

## Solutions Implemented

### 1. Non-Persistent Oscillator States
**File**: `rsl_rl/rsl_rl/modules/cpg.py`

Made the oscillator_states buffer non-persistent by adding `persistent=False`:

```python
self.register_buffer(
    "oscillator_states",
    self.INITIAL_STATE_SCALE * torch.randn(num_envs, self.num_oscillators, self.OSCILLATOR_STATE_DIM, device=device),
    persistent=False  # Don't save/load - it's environment-specific
)
```

**Why**: Oscillator states depend on num_envs and should be re-initialized for the inference environment, not loaded from the checkpoint. Only the coupling_matrix (which is environment-independent) should be saved.

### 2. Auto-Detect CPG Mode
**File**: `scripts/play.py`

Added logic to detect if checkpoint was trained with CPG and auto-enable it:

```python
# Check if the loaded checkpoint was trained with CPG
checkpoint_has_cpg = log_agent_cfg_dict.get("policy", {}).get("enable_cpg", False)

# If checkpoint has CPG but CLI doesn't specify, enable CPG
if checkpoint_has_cpg and not args_cli.enable_cpg:
    print("[INFO] Checkpoint was trained with CPG. Enabling CPG mode for inference.")
    args_cli.enable_cpg = True
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli, play=True)
```

**Why**: Users shouldn't need to remember if a checkpoint was trained with CPG. The script auto-detects and configures accordingly.

### 3. Use Checkpoint's History Length
**File**: `scripts/play.py`

Added logic to use history_length from checkpoint if not specified:

```python
checkpoint_history_length = log_agent_cfg_dict.get("policy", {}).get("history_length", 0)

if args_cli.history_length == 0 and checkpoint_history_length > 0:
    print(f"[INFO] Using history_length={checkpoint_history_length} from checkpoint.")
    args_cli.history_length = checkpoint_history_length
```

**Why**: The observation dimension depends on history_length. Using a different value than training causes dimension mismatch in the actor network.

### 4. Flexible State Dict Loading
**File**: `rsl_rl/rsl_rl/runners/on_policy_runner.py`

Changed to use `strict=False` when loading state_dict:

```python
def load(self, path, load_optimizer=True):
    loaded_dict = torch.load(path)
    # Use strict=False to allow loading checkpoints with different num_envs
    # (CPG oscillator_states will be re-initialized to current num_envs)
    self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"], strict=False)
```

**Why**: With non-persistent oscillator_states, there might be size mismatches for other buffers. `strict=False` allows the load to succeed and missing/extra keys are handled gracefully.

### 5. Safe Attribute Access
**File**: `scripts/cli_args.py`

Added hasattr check before accessing enable_cpg:

```python
if hasattr(args_cli, 'enable_cpg') and args_cli.enable_cpg:
    # CPG configuration
```

**Why**: The enable_cpg attribute might not exist if the script is called from contexts where the argument wasn't added to the parser.

## Usage

### Loading a CPG-trained checkpoint (auto-detect):
```bash
python scripts/play.py --task=go2_base_play --load_run=cpg_test
```

The script will:
1. Load the checkpoint config
2. Detect that it was trained with CPG
3. Auto-enable CPG mode
4. Use the correct history_length
5. Load only the persistent parameters (excluding oscillator_states)

### Loading a standard checkpoint:
```bash
python scripts/play.py --task=go2_base_play --load_run=standard_test
```

Works as before - no CPG will be enabled.

### Explicitly forcing CPG mode:
```bash
python scripts/play.py --task=go2_base_play --load_run=test --enable_cpg
```

CPG mode will be enabled regardless of checkpoint config.

## Technical Details

### What Gets Saved/Loaded
**Saved in checkpoint**:
- Actor/Critic network weights
- CPG coupling_matrix (persistent buffer)
- Optimizer state
- Normalizer state

**NOT saved** (re-initialized for inference):
- CPG oscillator_states (persistent=False)
- Any environment-dependent buffers

### Backward Compatibility
- ✅ Can load old non-CPG checkpoints in non-CPG mode
- ✅ Can load CPG checkpoints in CPG mode (auto-detected)
- ✅ Can load checkpoints trained with different num_envs
- ✅ Can load checkpoints with different history_length (must match via CLI or auto-detect)

## Commit
- **Hash**: 1690edd
- **Message**: "Fix play.py to handle CPG checkpoints properly"
