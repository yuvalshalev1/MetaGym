# MetaGym

## Installation

### 1. Create a conda environment

```bash
conda create -n metagym python=3.10 -y
conda activate metagym
```

### 2. Install vLLM

```bash
pip install vllm
```

### 3. Install MetaGym

```bash
pip install -e .
```

### 4. Install environment-specific dependencies

#### ALFWorld

```bash
pip install alfworld
```

Download the ALFWorld data:

```bash
export ALFWORLD_DATA=~/.cache/alfworld   # default; override as needed
alfworld-download
```

---

## Usage

```python
from metagym import Actor
from metagym.envs.alfworld import ALFWorldMetaEnv

# Actor auto-launches a vLLM server for the model and shuts it down on exit
actor = Actor("Qwen/Qwen2.5-7B-Instruct")

# Training — advances to new games each step
env = ALFWorldMetaEnv(num_envs=3, actor=actor, split="train", seed=42)
trajectories, info = env.reset()
trajectories, reward, done, info = env.step("You are a helpful agent...")
print(f"Train win rate: {reward:.0%}")

# Validation — same fixed games replayed on every step
val_env = ALFWorldMetaEnv(num_envs=16, actor=actor, split="eval_in_distribution", shuffle=False)
trajectories, info = val_env.reset()
trajectories, reward, done, info = val_env.step("You are a helpful agent...")
print(f"Val win rate: {reward:.0%}")

# Inspect a trajectory
trajectories[0].pretty_print()
```

### Reflector

The `Reflector` analyses trajectories and rewrites the system prompt to improve future performance.

```python
from metagym.reflector import Reflector

reflector = Reflector("Qwen/Qwen2.5-7B-Instruct", base_url=actor.base_url)

current_prompt = "You are a helpful agent..."
for step in range(4):
    trajectories, reward, done, info = env.step(current_prompt)
    print(f"Step {step}  win rate: {reward:.0%}")

    reflection = reflector.reflect(trajectories, current_prompt)
    print(reflection.analysis)           # what the reflector observed
    current_prompt = reflection.improved_prompt
```

---

## Adding a new environment

See [docs/adding-an-environment.md](docs/adding-an-environment.md).
