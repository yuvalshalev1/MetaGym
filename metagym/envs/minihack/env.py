"""MiniHack integration for MetaGym."""

import random
import re
from typing import List, Optional

from ...env import MetaGymEnv
from ...actor import Actor, find_best_action
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

# Fixed seed pools — train and val are disjoint to prevent contamination.
MINIHACK_TRAIN_SEEDS: List[int] = list(range(0, 10000))
MINIHACK_VAL_SEEDS: List[int] = list(range(10000, 10100))

_COMPASS_DIRS = ("n", "s", "e", "w", "ne", "nw", "se", "sw")

# Actions exposed to the agent — allowlist intersected with what the env provides.
# WAIT (noop) stays in _action_map for the fallback in step() but is not shown.
_ALLOWED_ACTION_NAMES = (
    [f"step {d}" for d in _COMPASS_DIRS]
    + [f"run {d}" for d in _COMPASS_DIRS]
    + ["read", "eat", "puton", "zap", "wield", "wear", "pickup"]
)



def _obs_to_str(obs: dict) -> str:
    """Convert a MiniHack observation dict to a human-readable string."""
    screen = "\n".join(
        bytes(row).decode("utf-8", errors="replace").rstrip()
        for row in obs["tty_chars"]
    )
    message = (
        bytes(obs["message"]).decode("utf-8", errors="replace")
        .rstrip("\x00")
        .strip()
    )
    return f"{message}\n\n{screen}" if message else screen


def _parse_game_id(game_id: str):
    """Parse 'env_name:seed' → (env_name, seed_int).

    Returns (env_name, None) if no seed suffix is present.
    """
    if ":" in game_id:
        name, seed_str = game_id.rsplit(":", 1)
        try:
            return name, int(seed_str)
        except ValueError:
            pass
    return game_id, None


# ---------------------------------------------------------------------------
# Inner single-game environment
# ---------------------------------------------------------------------------

class _MiniHackInnerEnv:
    """Minimal MiniHack wrapper: plays one game at a time.

    Game selection is handled externally by
    :class:`MiniHackMetaEnv._reset_env_to_game`, which sets
    ``_next_game_id`` before ``reset()`` is called.
    """

    def __init__(self, default_env_name: str, max_steps: int):
        self._default_env_name = default_env_name
        self._max_steps = max_steps

        self._gym_env = None
        self._current_env_name: Optional[str] = None
        self._current_seed: Optional[int] = None
        self._action_map: dict = {}
        self._available_actions: List[str] = []
        self._noop_action: str = "wait"
        self._base_actions = None

        self._next_game_id: Optional[str] = None  # set by MiniHackMetaEnv

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def game_id(self) -> str:
        name = self._current_env_name or self._default_env_name
        if self._current_seed is not None:
            return f"{name}:{self._current_seed}"
        return name

    @property
    def task(self) -> str:
        return self._current_env_name or self._default_env_name

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self):
        if self._next_game_id is None:
            raise RuntimeError(
                "_next_game_id must be set before calling reset(). "
                "Ensure MiniHackMetaEnv._reset_env_to_game is called first."
            )
        target_env_name, target_seed = _parse_game_id(self._next_game_id)

        # (Re)create the gym env when env name or seed changes.
        if (
            self._gym_env is None
            or target_env_name != self._current_env_name
            or target_seed != self._current_seed
        ):
            if self._gym_env is not None:
                try:
                    self._gym_env.close()
                except Exception:
                    pass
            self._gym_env = self._make_gym_env(target_env_name, target_seed)
            self._current_env_name = target_env_name
            self._current_seed = target_seed

        obs, _ = self._gym_env.reset()
        return _obs_to_str(obs), self._available_actions

    def step(self, action: str):
        action = find_best_action(action, self._available_actions)
        idx = self._action_map.get(action, self._action_map[self._noop_action])
        obs, reward, terminated, truncated, _ = self._gym_env.step(idx)
        obs, reward, terminated, truncated = self._auto_confirm(
            obs, reward, terminated, truncated
        )
        done = terminated or truncated
        return _obs_to_str(obs), float(reward), done, self._available_actions

    def close(self):
        if self._gym_env is not None:
            try:
                self._gym_env.close()
            except Exception:
                pass
            self._gym_env = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_gym_env(self, env_name: str, seed: Optional[int]):
        """Create (or recreate) the underlying gymnasium env."""
        import sys
        import gymnasium as gym
        # minihack/base.py imports pkg_resources which may be missing in some environments
        try:
            import pkg_resources  # noqa: F401
        except ImportError:
            import pip._vendor.pkg_resources as _pr
            sys.modules["pkg_resources"] = _pr
        import minihack  # noqa: F401 — registers MiniHack envs
        from nle import nethack

        # Build base action set once (reuse across recreations for the same env_name).
        if self._base_actions is None or env_name != self._current_env_name:
            _tmp = gym.make(env_name, observation_keys=("tty_chars", "message"))
            base = _tmp.unwrapped.actions
            _tmp.close()
            wait = nethack.actions.MiscDirection.WAIT
            self._base_actions = tuple(base) + ((wait,) if wait not in base else ())

        seeds_arg = [seed] if seed is not None else None
        env = gym.make(
            env_name,
            observation_keys=("tty_chars", "message"),
            max_episode_steps=self._max_steps,
            actions=self._base_actions,
            seeds=seeds_arg,
        )

        # Build action map from the new env.
        # CompassDirection → "step {dir}", CompassDirectionLonger → "run {dir}",
        # giving both distinct keys despite sharing the same .name in NLE.
        # All other actions use their lowercase NLE name; first occurrence wins.
        env_actions = env.unwrapped.actions
        self._action_map = {}
        for idx, action in enumerate(env_actions):
            if isinstance(action, nethack.actions.CompassDirection):
                name = f"step {action.name.lower()}"
            elif isinstance(action, nethack.actions.CompassDirectionLonger):
                name = f"run {action.name.lower()}"
            else:
                name = action.name.lower()
            if name not in self._action_map:
                self._action_map[name] = idx
        # Add single-char a-z aliases for auto-confirm and inventory-selection
        # prompts (e.g. "What to eat? [f or ?*]" → press 'f').  Not exposed in
        # _available_actions but needed by _auto_confirm.
        for i, action in enumerate(env_actions):
            char_val = int(action)
            if 97 <= char_val <= 122:
                char = chr(char_val)
                if char not in self._action_map:
                    self._action_map[char] = i
        self._available_actions = [
            a for a in _ALLOWED_ACTION_NAMES
            if a in self._action_map
        ]
        return env

    def _auto_confirm(self, obs, reward, terminated, truncated):
        """Automatically handle NetHack confirmation and item-selection prompts.

        Two prompt types are handled so the agent never needs to respond:
          - [yn] prompts (e.g. "Are you sure? [yn] (n)") → press 'y'
          - Item-selection prompts (e.g. "What to read? [f or ?*]") → press
            the first inventory letter listed in the brackets.

        The MiniHack reward manager two-step mechanism (primary action sets a
        flag, next 'y' keystroke triggers the win reward) still fires correctly
        because the auto-confirm goes through env.step() as usual.
        """
        while not (terminated or truncated):
            text = _obs_to_str(obs)
            if "[yn]" in text or "(n)" in text or "(y)" in text:
                key = "y"
            else:
                bracket = re.search(r"\[([^\]]*) or \?\*\]", text)
                letters = re.findall(r"[a-z]", bracket.group(1)) if bracket else []
                if letters:
                    key = letters[-1]  # last letter = most recently acquired item
                elif "?" in text and "[*]" in text:
                    key = "f"
                else:
                    break
            idx = self._action_map.get(key, self._action_map[self._noop_action])
            obs, r2, terminated, truncated, _ = self._gym_env.step(idx)
            reward += r2
        return obs, reward, terminated, truncated


# ---------------------------------------------------------------------------
# Public MetaGym wrapper
# ---------------------------------------------------------------------------

class MiniHackMetaEnv(MetaGymEnv):
    """MetaGym wrapper for MiniHack.

    Each :meth:`~metagym.env.MetaGymEnv.step` call runs ``num_envs`` full
    MiniHack episodes using the given system prompt and returns the fraction
    that succeeded.

    Splits are managed entirely through seeds — train and validation sets use
    disjoint seed ranges so the procedurally generated maps never overlap::

        MINIHACK_TRAIN_SEEDS = list(range(0, 10000))   # 10 000 training maps
        MINIHACK_VAL_SEEDS   = list(range(10000, 10100))  # 100 validation maps

    Game IDs use the format ``"env_name:seed"`` (e.g.
    ``"MiniHack-Room-Random-5x5-v0:42"``).  They are returned in ``info``
    after every step so exact games can be replayed::

        trajectories, reward, done, info = env.step(prompt)
        trajectories2, _, _, _ = env.step(new_prompt, game_ids=list(info.keys()))

    Args:
        num_envs: Number of episodes per :meth:`step`.
        actor: :class:`~metagym.actor.Actor` connected to a vLLM server.
        env_names: Pool of MiniHack gymnasium env IDs to sample from.  When
                   more than one is provided, env names are randomly assigned
                   across seeds.  Defaults to
                   ``["MiniHack-Room-Random-5x5-v0"]``.
        seeds: Seed list that determines the split.  Defaults to
               ``MINIHACK_TRAIN_SEEDS``.  Pass ``MINIHACK_VAL_SEEDS`` for a
               held-out validation environment.
        seed_start_index: Starting index into ``seeds`` for the first reset.
                          Use ``run_idx * episodes_per_run`` to give parallel
                          runs disjoint seed slices.
        rng_seed: Seed for the env-name sampling RNG when ``env_names`` has
                  more than one entry.
        shuffle: If ``True`` (default), the game-ID list is shuffled and each
                 :meth:`step` advances to a new batch.  If ``False``, every
                 :meth:`step` replays the same fixed first ``num_envs`` games
                 — useful for a consistent validation set.
        default_system_prompt: Prompt used by :meth:`reset`.  Defaults to the
                               built-in MiniHack system prompt.
    """

    def __init__(
        self,
        num_envs: int,
        actor: Actor,
        env_names: Optional[List[str]] = None,
        seeds: Optional[List[int]] = None,
        seed_start_index: int = 0,
        rng_seed: Optional[int] = None,
        shuffle: bool = True,
        progress_bar: bool = True,
        default_system_prompt: Optional[str] = None,
    ):
        self.env_names = env_names or ["MiniHack-Room-Random-5x5-v0"]
        self.seeds = seeds if seeds is not None else MINIHACK_TRAIN_SEEDS
        self.seed_start_index = seed_start_index
        self.rng_seed = rng_seed
        self.shuffle = shuffle

        self._game_ids: List[str] = []  # "env_name:seed" strings, populated lazily
        self._cursors: dict = {}         # env object id → next index in _game_ids

        super().__init__(
            num_envs=num_envs,
            actor=actor,
            default_system_prompt=default_system_prompt or SYSTEM_PROMPT,
            progress_bar=progress_bar,
        )

    # ------------------------------------------------------------------
    # MetaGymEnv abstract methods
    # ------------------------------------------------------------------

    def _make_env(self) -> _MiniHackInnerEnv:
        if not self._game_ids:
            self._load_game_ids()
        env = _MiniHackInnerEnv(self.env_names[0], max_steps=self.actor.max_steps)
        # Stagger starting positions so envs don't all begin at game 0.
        self._cursors[id(env)] = len(self._cursors) * self.num_envs + self.seed_start_index
        return env

    def _get_user_prompt_template(self) -> str:
        return USER_PROMPT_TEMPLATE

    def _reset_env_to_game(self, env: _MiniHackInnerEnv, game_id: Optional[str]) -> None:
        if game_id is not None:
            env._next_game_id = game_id
        else:
            cursor = self._cursors.get(id(env), 0)
            env._next_game_id = self._game_ids[cursor % len(self._game_ids)]
            self._cursors[id(env)] = cursor + 1

    # ------------------------------------------------------------------
    # Game ID management
    # ------------------------------------------------------------------

    def _load_game_ids(self):
        """Build the list of 'env_name:seed' game IDs from seeds × env_names."""
        rng = random.Random(self.rng_seed)
        if len(self.env_names) == 1:
            game_ids = [f"{self.env_names[0]}:{s}" for s in self.seeds]
        else:
            # Assign a (randomly sampled) env name to each seed.
            game_ids = [f"{rng.choice(self.env_names)}:{s}" for s in self.seeds]

        if self.shuffle:
            random.Random(self.rng_seed).shuffle(game_ids)

        self._game_ids = game_ids

        if self.num_envs > len(self._game_ids):
            print(
                f"Warning: num_envs={self.num_envs} exceeds available games "
                f"({len(self._game_ids)}). Using {len(self._game_ids)} envs."
            )
            self.num_envs = len(self._game_ids)

    @property
    def all_game_ids(self) -> List[str]:
        """All game IDs in the current seed pool."""
        if not self._game_ids:
            self._load_game_ids()
        return list(self._game_ids)

    # ------------------------------------------------------------------
    # Fixed-game overrides (shuffle=False)
    # ------------------------------------------------------------------

    def _fixed_game_ids(self) -> List[str]:
        if not self._game_ids:
            self._load_game_ids()
        return self._game_ids[:self.num_envs]

    def close(self):
        # Clear cursors so the next pool creation always starts with fresh stagger
        # indices.  Without this, Python id() reuse (after inner envs are freed)
        # causes all new envs to overwrite the same _cursors entries and receive
        # identical game positions.
        super().close()
        self._cursors.clear()

    def step(self, system_prompt, game_ids=None):
        if not self.shuffle and game_ids is None:
            game_ids = self._fixed_game_ids()
        return super().step(system_prompt, game_ids=game_ids)

    def reset(self, system_prompt=None, game_ids=None, skip_batch=False):
        if not self._game_ids:
            self._load_game_ids()
        if not self.shuffle and game_ids is None and not skip_batch:
            game_ids = self._fixed_game_ids()
        return super().reset(system_prompt=system_prompt, game_ids=game_ids, skip_batch=skip_batch)
