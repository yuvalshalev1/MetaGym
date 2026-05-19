"""ALFWorld integration for MetaGym."""

import hashlib
import json
import os
import random
import threading
from typing import List, Optional

# Reentrant lock for TextWorld/fast_downward operations that use thread-unsafe
# shared singletons:
#   1. fast_downward.translate.options  — pddl2sas() mutates add_implied_preconditions
#   2. textworld.envs.pddl.textgen._PARSER — tatsu parser with mutable stacks
# RLock because pddl2sas() (locked) is called inside _load_game (also locked).
_textworld_lock = threading.RLock()
_pddl2sas_patched = False


def _patch_pddl2sas_thread_safety():
    """Wrap fast_downward.pddl2sas() with a lock to prevent thread-safety bugs.

    pddl2sas() mutates sys.argv and the options module's attributes (e.g.
    add_implied_preconditions) which are shared across threads.  Without a
    lock, concurrent calls race on these globals, causing UnboundLocalError
    inside build_sas_operator().
    """
    global _pddl2sas_patched
    if _pddl2sas_patched:
        return
    try:
        import fast_downward
        import fast_downward.interface as _fd_iface

        _original = _fd_iface.pddl2sas

        def _locked_pddl2sas(*args, **kwargs):
            with _textworld_lock:
                return _original(*args, **kwargs)

        _fd_iface.pddl2sas = _locked_pddl2sas
        fast_downward.pddl2sas = _locked_pddl2sas
        _pddl2sas_patched = True
    except ImportError:
        pass

from ...env import MetaGymEnv
from ...actor import Actor
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


def _alfworld_data_root() -> str:
    """Return the ALFWorld data root, resolved at call time."""
    return os.environ.get("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld"))


def _to_relative(full_path: str) -> str:
    """Strip the ALFWORLD_DATA prefix so game_ids are portable."""
    root = _alfworld_data_root()
    if full_path.startswith(root):
        return full_path[len(root):].lstrip("/")
    return full_path


def _to_full(relative_path: str) -> str:
    """Restore the full path from a relative game_id."""
    if os.path.isabs(relative_path):
        return relative_path  # already absolute (e.g. from an older run)
    return os.path.join(_alfworld_data_root(), relative_path)


# ---------------------------------------------------------------------------
# Game-file cache (avoids repeated AlfredTWEnv os.walk on every MetaEnv init)
# ---------------------------------------------------------------------------
# Two levels, keyed by (abs_data_path, split, task_types_tuple):
#   1. In-memory dict  — shared within a process/Ray worker.
#   2. Disk JSON file  — shared across processes; survives job restarts.
# Different task_type combinations get separate entries and never collide.

_game_files_mem_cache: dict = {}  # key → list[relative_path]


def _game_file_cache_dir() -> str:
    return os.path.join(_alfworld_data_root(), ".game_file_cache")


def _game_file_cache_key(data_path: str, split: str, task_types: tuple) -> str:
    """MD5-based filename so the key is filesystem-safe regardless of path content."""
    raw = f"{os.path.abspath(data_path)}|{split}|{task_types}"
    return hashlib.md5(raw.encode()).hexdigest() + ".json"


def _load_cached_game_files(data_path: str, split: str, task_types: tuple):
    """Return cached relative-path game list, or None if not yet cached."""
    key = (os.path.abspath(data_path), split, task_types)
    if key in _game_files_mem_cache:
        return _game_files_mem_cache[key]
    disk_path = os.path.join(_game_file_cache_dir(), _game_file_cache_key(data_path, split, task_types))
    if os.path.exists(disk_path):
        with open(disk_path) as f:
            files = json.load(f)
        _game_files_mem_cache[key] = files
        return files
    return None


def _save_cached_game_files(data_path: str, split: str, task_types: tuple, files: list):
    """Write game list to in-memory and disk cache."""
    key = (os.path.abspath(data_path), split, task_types)
    _game_files_mem_cache[key] = files
    cache_dir = _game_file_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    disk_path = os.path.join(cache_dir, _game_file_cache_key(data_path, split, task_types))
    with open(disk_path, "w") as f:
        json.dump(files, f)


# ---------------------------------------------------------------------------
# Inner single-game environment
# ---------------------------------------------------------------------------

class _ALFWorldInnerEnv:
    """Minimal ALFWorld wrapper: loads and plays one game at a time.

    Game selection is handled externally by
    :class:`ALFWorldMetaEnv._reset_env_to_game`, which sets
    ``_next_game_id`` before ``reset()`` is called.
    """

    def __init__(self, config: dict, max_steps: int):
        self._config = config
        self._max_steps = max_steps
        self._tw_env = None             # reusable TextWorld gym env
        self._game_id: str = ""         # relative path of the current game
        self._next_game_id: Optional[str] = None  # set by ALFWorldMetaEnv

    @property
    def game_id(self) -> str:
        return self._game_id

    def reset(self):
        if self._next_game_id is None:
            raise RuntimeError(
                "_next_game_id must be set before calling reset(). "
                "Ensure ALFWorldMetaEnv._reset_env_to_game is called first."
            )
        return self._load_game(_to_full(self._next_game_id))

    def step(self, action: str):
        # Lock for the same reason as _load_game: tatsu and fast_downward
        # touch shared module-level state that is not thread-safe.
        with _textworld_lock:
            obs, scores, dones, infos = self._tw_env.step([action])
        available_actions = infos.get("admissible_commands", [[]])[0]
        return obs[0], float(scores[0]), bool(dones[0]), available_actions

    def close(self):
        if self._tw_env is not None:
            try:
                self._tw_env.close()
            except Exception:
                pass
            self._tw_env = None

    @staticmethod
    def _reset_textworld_parser():
        """Reset the singleton tatsu parser in TextWorld to clear corrupted state.

        TextWorld uses a module-level ``_PARSER`` (a tatsu parser) in
        ``textworld.envs.pddl.textgen``.  If a previous parse failed its
        internal stacks (_rule_stack, _statestack) are left dirty, causing all
        subsequent parses to crash with ``IndexError: pop from empty list``.
        Re-creating the parser before each game load prevents this.
        """
        try:
            import textworld.envs.pddl.textgen as _textgen
            _textgen._PARSER = _textgen.CSGParser(
                semantics=_textgen.CSGModelBuilderSemantics(), parseinfo=True
            )
        except Exception:
            pass

    def _load_game(self, game_file: str):
        """Load a specific game file and return (observation, available_actions)."""
        import textworld
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

        # One-time monkey-patch: wraps pddl2sas() with _textworld_lock so
        # concurrent Ray workers don't race on its shared globals.
        _patch_pddl2sas_thread_safety()

        # Hold the lock for the full load+reset sequence: _reset_textworld_parser
        # touches the singleton tatsu _PARSER, and env.reset() calls pddl2sas()
        # (already locked inside) plus tatsu parsing — all non-thread-safe.
        with _textworld_lock:
            # Clear any corrupted tatsu parser state from a previous failed load.
            self._reset_textworld_parser()

            if self._tw_env is None:
                request_infos = textworld.EnvInfos(
                    won=True,
                    admissible_commands=True,
                    extras=["gamefile"],
                )
                env_id = textworld.gym.register_games(
                    [game_file],
                    request_infos,
                    batch_size=1,
                    asynchronous=False,
                    max_episode_steps=self._max_steps,
                    wrappers=[AlfredDemangler(shuffle=False), AlfredInfos],
                )
                self._tw_env = textworld.gym.make(env_id)
            else:
                self._tw_env.gamefiles = [game_file]
                self._tw_env.seed(0)

            obs, info = self._tw_env.reset()

        available_actions = info.get("admissible_commands", [[]])[0]
        game_files_in_info = info.get("extra.gamefile", [])
        full_gid = game_files_in_info[0] if game_files_in_info else game_file
        self._game_id = _to_relative(full_gid)
        return obs[0], available_actions


# ---------------------------------------------------------------------------
# Public MetaGym wrapper
# ---------------------------------------------------------------------------

class ALFWorldMetaEnv(MetaGymEnv):
    """MetaGym wrapper for ALFWorld.

    Each :meth:`~metagym.env.MetaGymEnv.step` call runs ``num_envs`` full
    ALFWorld episodes using the given system prompt and returns the fraction
    that succeeded.

    Game IDs are ALFWorld scenario file paths stored relative to
    ``$ALFWORLD_DATA`` (default: ``~/.cache/alfworld``), making them portable
    across machines.  They are returned in ``info`` after every step::

        trajectories, reward, done, info = env.step(prompt)
        # Replay the exact same games with a different prompt:
        trajectories, reward, done, info = env.step(new_prompt, game_ids=list(info.keys()))

    Args:
        num_envs: Number of episodes per :meth:`step`.
        actor: :class:`~metagym.actor.Actor` connected to a vLLM server.
        split: Dataset split — ``"train"``, ``"eval_in_distribution"``, or
               ``"eval_out_of_distribution"``.
        task_types: ALFWorld task types to include (1–6).  ``None`` = all six.
        seed: Seed for shuffling the game list (controls episode order).
        shuffle: If ``True`` (default), games are shuffled according to ``seed``
                 and each :meth:`step` advances to a new batch.  If ``False``,
                 games are kept in their natural order and every :meth:`step`
                 replays the same fixed first ``num_envs`` games — useful for
                 a consistent validation set.
        default_system_prompt: Prompt used by :meth:`reset`.  Defaults to the
                               built-in ALFWorld system prompt.
    """

    _SPLIT_SUBDIR = {
        "train": "train",
        "eval_in_distribution": "valid_seen",
        "eval_out_of_distribution": "valid_unseen",
    }

    _TASK_PATTERNS = {
        1: "pick_and_place_simple",
        2: "look_at_obj_in_light",
        3: "pick_clean_then_place_in_recep",
        4: "pick_heat_then_place_in_recep",
        5: "pick_cool_then_place_in_recep",
        6: "pick_two_obj_and_place",
    }

    def __init__(
        self,
        num_envs: int,
        actor: Actor,
        split: str = "train",
        task_types: Optional[List[int]] = None,
        seed: Optional[int] = None,
        shuffle: bool = True,
        progress_bar: bool = True,
        default_system_prompt: Optional[str] = None,
    ):
        if split not in self._SPLIT_SUBDIR:
            raise ValueError(f"split must be one of {list(self._SPLIT_SUBDIR)}, got '{split}'")

        self.split = split
        self.task_types = task_types or list(range(1, 7))
        self.seed = seed
        self.shuffle = shuffle

        self._game_files: List[str] = []  # relative paths, populated lazily
        self._alf_config: dict = {}

        # Per-env cursor: maps env object id → next index in _game_files
        self._cursors: dict = {}

        super().__init__(
            num_envs=num_envs,
            actor=actor,
            default_system_prompt=default_system_prompt or SYSTEM_PROMPT,
            progress_bar=progress_bar,
        )

    # ------------------------------------------------------------------
    # MetaGymEnv abstract methods
    # ------------------------------------------------------------------

    def _make_env(self) -> _ALFWorldInnerEnv:
        if not self._game_files:
            self._load_game_files()
        env = _ALFWorldInnerEnv(self._alf_config, max_steps=self.actor.max_steps)
        # Stagger starting positions so envs don't all begin at game 0
        self._cursors[id(env)] = len(self._cursors) * self.num_envs
        return env

    def _get_user_prompt_template(self) -> str:
        return USER_PROMPT_TEMPLATE

    def _reset_env_to_game(self, env: _ALFWorldInnerEnv, game_id: Optional[str]) -> None:
        if game_id is not None:
            env._next_game_id = game_id
        else:
            cursor = self._cursors.get(id(env), 0)
            env._next_game_id = self._game_files[cursor % len(self._game_files)]
            self._cursors[id(env)] = cursor + 1

    # ------------------------------------------------------------------
    # Game file management
    # ------------------------------------------------------------------

    def _load_game_files(self):
        """Populate self._game_files from cache or by scanning ALFWorld data.

        Game-file scanning (AlfredTWEnv + os.walk) is expensive on network
        filesystems.  Results are cached in memory (per process) and on disk
        (across processes / Ray workers) keyed by (data_path, split, task_types)
        so the scan runs at most once per unique combination.
        """
        from alfworld.agents.environment import get_environment

        root = _alfworld_data_root()
        subdir = self._SPLIT_SUBDIR[self.split]
        data_path = os.path.join(root, "json_2.1.1", subdir)
        task_types_key = tuple(sorted(self.task_types))

        # Build the ALFWorld config (needed for _ALFWorldInnerEnv even when
        # the file list comes from cache).
        self._alf_config = {
            "dataset": {
                "data_path": data_path,
                "eval_id_data_path": os.path.join(root, "json_2.1.1", "valid_seen"),
                "eval_ood_data_path": os.path.join(root, "json_2.1.1", "valid_unseen"),
                "num_train_games": -1,
                "num_eval_games": -1,
            },
            "logic": {
                "domain": os.path.join(root, "logic", "alfred.pddl"),
                "grammar": os.path.join(root, "logic", "alfred.twl2"),
            },
            "env": {
                "type": "AlfredTWEnv",
                "regen_game_files": False,
                "domain_randomization": False,
                "task_types": self.task_types,
                "expert_type": "handcoded",
                "goal_desc_human_anns_prob": 0,
            },
            # ALFWorld's config format requires this section;
            # max_nb_steps_per_episode is driven by actor.max_steps via _make_env
            "general": {"random_seed": self.seed or 42, "training_method": "dagger"},
            "dagger": {"training": {"max_nb_steps_per_episode": self.actor.max_steps}},
        }

        # 1) Try in-memory / disk cache first — no os.walk needed.
        cached = _load_cached_game_files(data_path, self.split, task_types_key)
        if cached is not None:
            # Copy so that an in-place shuffle below doesn't corrupt the cache.
            all_files_relative = list(cached)
        else:
            # 2) Cache miss: run the expensive scan and persist the result.
            AlfredTWEnv = get_environment("AlfredTWEnv")
            alf_env = AlfredTWEnv(self._alf_config, train_eval=self.split)
            all_files = list(alf_env.game_files)

            patterns = [self._TASK_PATTERNS[t] for t in self.task_types if t in self._TASK_PATTERNS]
            if patterns:
                all_files = [f for f in all_files if any(p in f for p in patterns)]

            all_files_relative = [_to_relative(f) for f in all_files]
            # Save before shuffling so the cache always holds canonical order.
            _save_cached_game_files(data_path, self.split, task_types_key, all_files_relative)
            all_files_relative = list(all_files_relative)  # copy before in-place shuffle

        if self.shuffle and self.seed is not None:
            random.Random(self.seed).shuffle(all_files_relative)

        self._game_files = all_files_relative

        if self.num_envs > len(self._game_files):
            print(
                f"Warning: num_envs={self.num_envs} exceeds available games "
                f"({len(self._game_files)}). Using {len(self._game_files)} envs."
            )
            self.num_envs = len(self._game_files)

    @property
    def all_game_ids(self) -> List[str]:
        """All available game IDs (relative paths) for this split + task config."""
        if not self._game_files:
            self._load_game_files()
        return list(self._game_files)

    # ------------------------------------------------------------------
    # Fixed-game overrides (shuffle=False)
    # ------------------------------------------------------------------

    def _fixed_game_ids(self) -> List[str]:
        if not self._game_files:
            self._load_game_files()
        return self._game_files[:self.num_envs]

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
        if not self._game_files:
            self._load_game_files()  # ensures num_envs is clamped before pool creation
        if not self.shuffle and game_ids is None and not skip_batch:
            game_ids = self._fixed_game_ids()
        return super().reset(system_prompt=system_prompt, game_ids=game_ids, skip_batch=skip_batch)
