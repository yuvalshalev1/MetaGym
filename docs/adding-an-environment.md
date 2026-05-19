# Adding a new environment

Each environment lives in its own subpackage under `metagym/envs/`. Use ALFWorld
(`metagym/envs/alfworld/`) or MiniHack (`metagym/envs/minihack/`) as a reference.

### 1. Create the subpackage

```
metagym/envs/yourenv/
    __init__.py   # exports YourMetaEnv and any public constants
    env.py        # _YourInnerEnv + YourMetaEnv
    prompts.py    # SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
```

### 2. Implement the inner env (`_YourInnerEnv`)

Wraps a single episode of the underlying environment. Must expose:

```python
class _YourInnerEnv:
    # Set by YourMetaEnv._reset_env_to_game before every reset()
    _next_game_id: Optional[str]

    def reset(self) -> tuple[str, list[str]]:
        """Return (observation_text, available_actions)."""

    def step(self, action: str) -> tuple[str, float, bool, list[str]]:
        """Return (observation_text, reward, done, available_actions)."""

    def close(self): ...

    @property
    def game_id(self) -> str:
        """Unique identifier for the current episode (used for replay)."""

    @property
    def task(self) -> str:
        """Task description shown to the agent, or '' if none."""
```

`reset()` reads `_next_game_id` to know which game/seed to load. If the underlying
library is not deterministic by default, recreate the env object whenever `_next_game_id`
changes (see `_MiniHackInnerEnv._make_gym_env` for an example).

### 3. Implement the outer env (`YourMetaEnv`)

Subclass `MetaGymEnv` and implement three methods:

```python
from metagym.env import MetaGymEnv

class YourMetaEnv(MetaGymEnv):

    def _make_env(self) -> _YourInnerEnv:
        """Create one inner env instance; stagger its starting cursor."""

    def _get_user_prompt_template(self) -> str:
        """Return the per-step prompt template.
        Must contain {observation} and {available_actions}.
        Include {task} only if the environment presents a task to the agent."""

    def _reset_env_to_game(self, env: _YourInnerEnv, game_id: Optional[str]) -> None:
        """Point env at the next game.
        If game_id is None, pick the next one from your internal list and
        advance the cursor. If game_id is provided, use it directly (replay)."""
```

Game selection works through a list of string `game_id`s maintained by
`YourMetaEnv`. A per-env cursor advances on each call to `_reset_env_to_game`.
Stagger cursors in `_make_env` so that envs in the same batch start at different
positions and therefore play distinct games:

```python
self._cursors[id(env)] = len(self._cursors) * self.num_envs
```

For `shuffle=False` (fixed validation set), override `step()` and `reset()` to
pin `game_ids` to the first `num_envs` entries — see `ALFWorldMetaEnv.step` /
`MiniHackMetaEnv.step` for the exact pattern.

### 4. Write the prompts (`prompts.py`)

```python
SYSTEM_PROMPT = "..."          # role + ReACT format instructions

USER_PROMPT_TEMPLATE = """\
{observation}

Available actions:
{available_actions}

What is your next thought and action?"""
```

The template is rendered by the base class at every step. `{task}` is available
as a placeholder but only include it if the environment actually surfaces a goal
description to the agent.

### 5. Wire up exports

**`metagym/envs/yourenv/__init__.py`**
```python
from .env import YourMetaEnv
__all__ = ["YourMetaEnv"]
```

**`metagym/envs/__init__.py`** — add:
```python
from .yourenv import YourMetaEnv
```

**`pyproject.toml`** — add an optional extra so users can install just what they need:
```toml
[project.optional-dependencies]
yourenv = ["your-package-name"]
```

### 6. Add to the determinism tests

Open `tests/test_determinism.py` and add one entry to each registry:

```python
from metagym.envs.yourenv import YourMetaEnv

ENV_FACTORIES["yourenv"] = lambda actor: YourMetaEnv(num_envs=NUM_ENVS, actor=actor, ...)
SEEDED_FACTORIES["yourenv"] = lambda actor, seed: YourMetaEnv(num_envs=NUM_ENVS, actor=actor, seed=seed, ...)
```

The existing parameterised tests then verify automatically that:
- each env in a batch plays a **distinct** game
- consecutive steps advance to **new** games
- replaying the same `game_ids` produces **identical** initial observations
