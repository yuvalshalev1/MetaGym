"""Tests for MiniHack action map collision fix and find_best_action resolver."""

import os
import sys

import pytest

os.environ.setdefault("SETUPTOOLS_USE_DISTUTILS", "stdlib")
try:
    import pkg_resources  # noqa: F401
except ImportError:
    import pip._vendor.pkg_resources as _pr
    sys.modules["pkg_resources"] = _pr

from metagym.actor import find_best_action, _edit_distance


# ---------------------------------------------------------------------------
# _edit_distance
# ---------------------------------------------------------------------------

def test_edit_distance_identical():
    assert _edit_distance("eat", "eat") == 0


def test_edit_distance_case_sensitive():
    # case-sensitive: "N" vs "n" differ by 1 substitution
    assert _edit_distance("N", "n") == 1


def test_edit_distance_insertion():
    assert _edit_distance("eat", "eat apple") == 6


def test_edit_distance_empty():
    assert _edit_distance("", "abc") == 3
    assert _edit_distance("abc", "") == 3


# ---------------------------------------------------------------------------
# find_best_action
# ---------------------------------------------------------------------------

AVAILABLE = ["N", "E", "S", "W", "NE", "SE", "SW", "NW", "EAT", "LOOK", "WAIT", "k", "y"]


def test_exact_match():
    assert find_best_action("N", AVAILABLE) == "N"


def test_case_insensitive_match():
    # LLM outputs lowercase "n"; must resolve to "N" not "SE" (whose char is 'n')
    assert find_best_action("n", AVAILABLE) == "N"


def test_case_insensitive_eat():
    assert find_best_action("eat", AVAILABLE) == "EAT"


def test_edit_distance_compound():
    # "eat apple" → distance 6 from "EAT", which exceeds default threshold of 5,
    # so falls back to first available. Use a higher threshold to resolve it.
    assert find_best_action("eat apple", AVAILABLE, threshold=10) == "EAT"


def test_edit_distance_compound_default_threshold():
    # With default threshold=5, "eat apple" (distance 6) exceeds it → first available
    assert find_best_action("eat apple", AVAILABLE) == AVAILABLE[0]


def test_edit_distance_typo():
    assert find_best_action("lok", AVAILABLE) == "LOOK"


def test_fallback_on_large_distance():
    # "xyzxyzxyz" is far from everything; should fall back to first available
    result = find_best_action("xyzxyzxyz", AVAILABLE, threshold=5)
    assert result == AVAILABLE[0]


def test_empty_available():
    # No available actions: return the proposed string unchanged
    assert find_best_action("n", []) == "n"


def test_char_alias():
    # "k" is the vi-key alias for north (CompassDirection.N)
    assert find_best_action("k", AVAILABLE) == "k"


# ---------------------------------------------------------------------------
# Action map: collision and char-alias tests (requires MiniHack installed)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def minihack_inner_env():
    """Create a _MiniHackInnerEnv for MiniHack-Eat-v0 and yield its action_map."""
    from metagym.envs.minihack.env import _MiniHackInnerEnv
    env = _MiniHackInnerEnv("MiniHack-Eat-v0", max_steps=25)
    env._next_game_id = "MiniHack-Eat-v0:0"
    env.reset()
    yield env
    env.close()


def test_north_maps_to_single_step(minihack_inner_env):
    """'step n' must resolve to CompassDirection.N (index 0), not CompassDirectionLonger.N."""
    action_map = minihack_inner_env._action_map
    assert "step n" in action_map, "'step n' must be a key in the action map"
    assert action_map["step n"] == 0, (
        f"'step n' should map to index 0 (CompassDirection.N), got {action_map['step n']}"
    )


def test_run_n_distinct_from_step_n(minihack_inner_env):
    """'run n' must be present and map to a different index than 'step n'."""
    action_map = minihack_inner_env._action_map
    assert "run n" in action_map, "'run n' must be a key in the action map"
    assert action_map["run n"] != action_map["step n"], (
        "'run n' and 'step n' must map to different indices"
    )


def test_vi_key_k_maps_to_single_step_north(minihack_inner_env):
    """'k' (char alias for CompassDirection.N, int=107) must map to index 0."""
    action_map = minihack_inner_env._action_map
    assert "k" in action_map, "'k' vi-key alias must be present"
    assert action_map["k"] == 0, (
        f"'k' should map to CompassDirection.N (index 0), got {action_map['k']}"
    )


def test_y_alias_maps_to_nw(minihack_inner_env):
    """'y' (char 121) must map to CompassDirection.NW (index 7), not CompassDirectionLonger.NW."""
    action_map = minihack_inner_env._action_map
    assert "y" in action_map, "'y' vi-key alias must be present"
    assert action_map["y"] == 7, (
        f"'y' should map to CompassDirection.NW (index 7), got {action_map['y']}"
    )


def test_noop_action_in_map(minihack_inner_env):
    """The noop action 'WAIT' must be resolvable in the action map."""
    noop = minihack_inner_env._noop_action
    assert noop in minihack_inner_env._action_map, (
        f"noop action '{noop}' must be in the action map"
    )
