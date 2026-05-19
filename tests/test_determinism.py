"""Determinism tests for MetaGym environments.

Tests the same three properties for every registered environment:

  1. Each env in a batch plays a distinct game.
  2. Consecutive steps advance to new games.
  3. Replaying the same game_ids reproduces identical initial observations.

To add a new environment, append an entry to ENV_FACTORIES at the top of
this file.  Each factory receives a mock actor and must return a ready-to-use
MetaGymEnv subclass.

Requirements: the environment-specific package must be installed and its data
must be reachable (e.g. $ALFWORLD_DATA for ALFWorld).  No LLM server is needed
— a mock actor is used in all tests.
"""

import pytest

from metagym.actor import Actor
from metagym.envs.alfworld import ALFWorldMetaEnv
from metagym.envs.minihack import MiniHackMetaEnv


NUM_ENVS = 4


# ---------------------------------------------------------------------------
# Mock actor (no LLM required)
# ---------------------------------------------------------------------------

class _MockActor(Actor):
    """Returns 'look' at every step without calling any LLM server."""

    def __init__(self):
        self.model = "mock"
        self.max_steps = 1   # one step is enough; we only care about reset()
        self.temperature = 0.0
        self.max_tokens = 10
        self.base_url = ""
        self._client = None

    def call(self, messages):
        return "Thought: exploring.\nAction: look"


# ---------------------------------------------------------------------------
# Environment registry
# Add one entry here for each new environment integration.
# ---------------------------------------------------------------------------

ENV_FACTORIES = {
    "alfworld": lambda actor: ALFWorldMetaEnv(
        num_envs=NUM_ENVS,
        actor=actor,
        split="train",
        task_types=[1],   # pick_and_place_simple — smallest set, loads fast
        seed=42,
    ),
    # Each MiniHack episode uses a distinct seed so distinct maps are guaranteed.
    # 16 seeds = 4 envs × 4 steps; enough for all parameterised determinism tests.
    "minihack": lambda actor: MiniHackMetaEnv(
        num_envs=NUM_ENVS,
        actor=actor,
        seeds=list(range(16)),
        rng_seed=42,
    ),
    # "webshop": lambda actor: WebShopMetaGymEnv(num_envs=NUM_ENVS, actor=actor, ...),
}

# Factories that support seed-based reproducibility (env name → factory accepting seed kwarg)
SEEDED_FACTORIES = {
    "alfworld": lambda actor, seed: ALFWorldMetaEnv(
        num_envs=NUM_ENVS,
        actor=actor,
        split="train",
        task_types=[1],
        seed=seed,
    ),
    # rng_seed controls game-ID list shuffle, making two instances visit the same sequence.
    "minihack": lambda actor, seed: MiniHackMetaEnv(
        num_envs=NUM_ENVS,
        actor=actor,
        seeds=list(range(16)),
        rng_seed=seed,
    ),
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=list(ENV_FACTORIES.keys()), scope="module")
def env(request):
    """Parameterized fixture: yields one env per entry in ENV_FACTORIES."""
    actor = _MockActor()
    e = ENV_FACTORIES[request.param](actor)
    yield e
    e.close()


# ---------------------------------------------------------------------------
# Tests (run once per environment in ENV_FACTORIES)
# ---------------------------------------------------------------------------

def test_batch_has_distinct_games(env):
    """All envs in a single batch must play different games."""
    trajectories, info = env.reset()

    assert len(info) == NUM_ENVS, (
        f"Expected {NUM_ENVS} entries in info, got {len(info)}"
    )
    assert len(set(info.keys())) == NUM_ENVS, (
        f"Expected {NUM_ENVS} distinct game_ids, got duplicates: {list(info.keys())}"
    )
    obs = [t.initial_observation for t in trajectories]
    assert len(set(obs)) == NUM_ENVS, (
        "Initial observations should be distinct across games in the same batch"
    )


def test_step_advances_to_new_games(env):
    """A step must play a different set of games than the previous batch."""
    _, info_1 = env.reset()
    _, _, _, info_2 = env.step("some prompt")

    games_1 = set(info_1.keys())
    games_2 = set(info_2.keys())

    assert games_1.isdisjoint(games_2), (
        f"Step should advance to new games.\n"
        f"batch 1: {sorted(games_1)}\n"
        f"batch 2: {sorted(games_2)}"
    )


def test_replay_reproduces_observations(env):
    """Replaying the same game_ids must produce identical initial observations."""
    env.reset(skip_batch=True)
    # Run a batch and record observations keyed by game_id
    trajectories_1, _, _, info_1 = env.step("prompt v1")
    obs_1 = {t.game_id: t.initial_observation for t in trajectories_1}

    # Advance past this batch so we know replay is loading from scratch
    env.step("prompt v1")

    # Replay the first batch with a different prompt
    game_ids_to_replay = list(info_1.keys())
    trajectories_2, _, _, info_2 = env.step("prompt v2", game_ids=game_ids_to_replay)
    obs_2 = {t.game_id: t.initial_observation for t in trajectories_2}

    assert set(info_2.keys()) == set(info_1.keys()), (
        "Replayed game_ids should match the requested ones."
    )
    assert obs_2 == obs_1, (
        "Replaying the same game_ids must reproduce identical initial observations.\n"
        f"Mismatched: {[gid for gid in obs_1 if obs_1[gid] != obs_2.get(gid)]}"
    )


def test_fixed_games_mode():
    """With shuffle=False, reset() and every step() must replay the same fixed game set."""
    actor = _MockActor()
    env = ALFWorldMetaEnv(
        num_envs=NUM_ENVS,
        actor=actor,
        split="train",
        task_types=[1],
        shuffle=False,
    )
    try:
        _, info_reset = env.reset()
        _, _, _, info_step1 = env.step("prompt a")
        _, _, _, info_step2 = env.step("prompt b")

        fixed = set(info_reset.keys())
        assert set(info_step1.keys()) == fixed, (
            f"step 1 game_ids differ from reset:\n"
            f"reset:  {sorted(fixed)}\nstep 1: {sorted(info_step1.keys())}"
        )
        assert set(info_step2.keys()) == fixed, (
            f"step 2 game_ids differ from reset:\n"
            f"reset:  {sorted(fixed)}\nstep 2: {sorted(info_step2.keys())}"
        )
    finally:
        env.close()


def test_skip_batch_reset_then_step_works(env):
    """reset(skip_batch=True) initialises the pool; subsequent step() must work correctly."""
    trajs, info = env.reset(skip_batch=True)
    assert trajs == [], "skip_batch=True must return empty trajectories"
    assert info == {}, "skip_batch=True must return empty info"

    # step() must function normally after a skip_batch reset
    trajs2, reward, done, info2 = env.step("some prompt")
    assert len(trajs2) == env.num_envs
    assert len(info2) == env.num_envs
    assert 0.0 <= reward <= 1.0


def test_game_file_cache_avoids_rescan():
    """Second ALFWorldMetaEnv init with same (split, task_types) must hit the cache."""
    import metagym.envs.alfworld.env as alf_mod

    actor = _MockActor()
    task_types = [1]

    # Prime the cache with the first instance.
    env_a = ALFWorldMetaEnv(num_envs=1, actor=actor, split="train", task_types=task_types, seed=0)
    env_a._load_game_files()  # ensure populated
    files_a = list(env_a._game_files)
    env_a.close()

    # The in-memory cache must now be populated.
    from metagym.envs.alfworld.env import _game_files_mem_cache, _alfworld_data_root
    import os
    root = _alfworld_data_root()
    data_path = os.path.join(root, "json_2.1.1", "train")
    cache_key = (os.path.abspath(data_path), "train", tuple(sorted(task_types)))
    assert cache_key in _game_files_mem_cache, "Cache was not populated after first init"

    # Second instance must return identical file list (order may differ due to shuffle/seed).
    env_b = ALFWorldMetaEnv(num_envs=1, actor=actor, split="train", task_types=task_types, seed=0)
    env_b._load_game_files()
    files_b = list(env_b._game_files)
    env_b.close()

    assert sorted(files_a) == sorted(files_b), (
        "Game file list changed between two inits with same config"
    )


def test_seed_reproducibility():
    """Two env instances with the same seed must visit the same games in the same order."""
    actor = _MockActor()
    for env_name, factory in SEEDED_FACTORIES.items():
        env_a = factory(actor, seed=42)
        env_b = factory(actor, seed=42)
        try:
            _, info_a_reset = env_a.reset()
            _, info_b_reset = env_b.reset()
            assert set(info_a_reset.keys()) == set(info_b_reset.keys()), (
                f"[{env_name}] reset() game_ids differ despite same seed.\n"
                f"env_a: {sorted(info_a_reset.keys())}\n"
                f"env_b: {sorted(info_b_reset.keys())}"
            )

            _, _, _, info_a_step = env_a.step("prompt")
            _, _, _, info_b_step = env_b.step("prompt")
            assert set(info_a_step.keys()) == set(info_b_step.keys()), (
                f"[{env_name}] step() game_ids differ despite same seed.\n"
                f"env_a: {sorted(info_a_step.keys())}\n"
                f"env_b: {sorted(info_b_step.keys())}"
            )
        finally:
            env_a.close()
            env_b.close()
