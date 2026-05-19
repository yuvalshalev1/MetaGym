"""Reflector: analyses trajectories and produces an improved system prompt."""

import re
import time
from dataclasses import dataclass
from typing import List, Optional

from .trajectory import Trajectory


# ---------------------------------------------------------------------------
# Default prompts
# ---------------------------------------------------------------------------

_DEFAULT_SYSTEM_PROMPT = """\
You are an expert at analysing game-playing trajectories and improving agent instructions.

You will be given one or more episodes in which an agent attempts to complete a task \
guided by a system prompt. Your goal is to learn from these episodes to improve the \
agent's performance on future tasks.

Your job is to:
1. Identify what went well and what went wrong in the agent's behaviour.
2. Diagnose how the current system prompt contributed to those outcomes.
3. Write an improved system prompt that addresses the identified weaknesses to improve \
the agent's performance in future episodes.

Respond in EXACTLY this format — no additional text outside these two sections:

ANALYSIS:
[Describe what succeeded, what failed, how the system prompt contributed to those \
outcomes, and what specific changes the improved prompt should make.]

IMPROVED PROMPT:
[The full improved system prompt, written to be used directly without modification.]\
"""

_DEFAULT_USER_PROMPT_TEMPLATE = """\
## System Prompt Used

{previous_prompt}

## Episode Trajectories

{trajectories_text}

Analyse the trajectories and provide an improved system prompt.\
"""


# ---------------------------------------------------------------------------
# Reflection result
# ---------------------------------------------------------------------------

@dataclass
class Reflection:
    """Output of a single :meth:`Reflector.reflect` call."""
    analysis: str
    improved_prompt: str
    raw_response: str = ""
    prompt_sent: str = ""


# ---------------------------------------------------------------------------
# Reflector
# ---------------------------------------------------------------------------

class Reflector:
    """Analyses episode trajectories and returns an improved system prompt.

    The reflector can share a vLLM server with the :class:`~metagym.actor.Actor`
    (same model, same ``base_url``) or point to a separate server / remote API::

        actor = Actor("Qwen/Qwen2.5-7B-Instruct", base_url="http://localhost:8000/v1")

        # Shared — reflector re-uses the same vLLM server as the actor
        reflector = Reflector("Qwen/Qwen2.5-7B-Instruct", base_url=actor.base_url)

        # Separate vLLM server running a different model
        reflector = Reflector("meta-llama/Llama-3-70B", base_url="http://gpu2:8000/v1")

        # Remote OpenAI-compatible API (omit base_url)
        reflector = Reflector("gpt-4o")

    Usage::

        trajectories, reward, done, info = env.step(current_prompt)
        reflection = reflector.reflect(trajectories, current_prompt)
        current_prompt = reflection.improved_prompt

    Args:
        model: Model name — a vLLM-served model ID or an OpenAI model name.
        base_url: Base URL of the OpenAI-compatible server (e.g.
                  ``"http://localhost:8000/v1"``).  ``"auto"`` (default)
                  launches a vLLM server automatically.  ``None`` uses the
                  real OpenAI API (``OPENAI_API_KEY`` must be set).
        system_prompt: Override the built-in reflector system prompt.
        user_prompt_template: Override the built-in user prompt template.
                              Must contain ``{previous_prompt}`` and
                              ``{trajectories_text}`` placeholders.
        max_tokens: Max tokens for the reflection response.
        seed: Sampling seed passed to the API for reproducibility.
        **server_kwargs: Extra keyword arguments forwarded to
                         :func:`~metagym.vllm_server.launch_vllm_server`
                         when auto-launching (e.g. ``gpu_memory_utilization``,
                         ``max_model_len``).
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        base_url: Optional[str] = "auto",
        system_prompt: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
        max_tokens: int = 2000,
        seed: Optional[int] = None,
        **server_kwargs,
    ):
        self.model = model
        self.system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT
        self.user_prompt_template = user_prompt_template or _DEFAULT_USER_PROMPT_TEMPLATE
        self.max_tokens = max_tokens
        self.seed = seed

        if base_url == "auto":
            from .vllm_server import launch_vllm_server
            base_url = launch_vllm_server(model, **server_kwargs)

        self.base_url = base_url

        from openai import OpenAI
        self._client = OpenAI(
            base_url=base_url,
            api_key="none" if base_url else None,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reflect(
        self,
        trajectories: List[Trajectory],
        current_prompt: str,
        include_thoughts: bool = False,
        include_available_actions: bool = False,
    ) -> Reflection:
        """Analyse trajectories and return an improved system prompt.

        Args:
            trajectories: Episodes to analyse (returned by ``env.step()`` or
                          ``env.reset()``).
            current_prompt: The system prompt used to produce these trajectories.
            include_thoughts: Include the agent's ``Thought:`` lines in the
                              formatted trajectories sent to the reflector.
                              Defaults to ``False``.
            include_available_actions: Include the available-actions list in
                                       the formatted trajectories.
                                       Defaults to ``False``.

        Returns:
            :class:`Reflection` with ``improved_prompt`` and ``analysis``.
        """
        trajectories_text = self._format_trajectories(
            trajectories, include_thoughts, include_available_actions
        )
        user_prompt = self.user_prompt_template.format(
            previous_prompt=current_prompt,
            trajectories_text=trajectories_text,
        )

        api_kwargs: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "max_tokens": self.max_tokens,
        }
        if self.seed is not None:
            api_kwargs["seed"] = self.seed

        raw = self._call_with_retry(
            lambda: self._client.chat.completions.create(**api_kwargs)
        )
        reflection_text = raw.choices[0].message.content

        analysis, improved_prompt = self._parse_reflection(reflection_text)

        if not improved_prompt:
            print(
                "Warning: reflector did not return an improved prompt. "
                "Using the current prompt as fallback."
            )
            improved_prompt = current_prompt

        return Reflection(
            analysis=analysis,
            improved_prompt=improved_prompt,
            raw_response=reflection_text,
            prompt_sent=user_prompt,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _format_trajectories(
        self,
        trajectories: List[Trajectory],
        include_thoughts: bool,
        include_available_actions: bool,
    ) -> str:
        parts = []
        for i, traj in enumerate(trajectories):
            parts.append(f"=== Episode {i + 1} ===")
            parts.append(f"Success: {'Yes' if traj.success else 'No'}")
            parts.append("")
            for j, step in enumerate(traj.steps):
                parts.append(f"--- Step {j + 1} ---")
                pre_obs = traj.initial_observation if j == 0 else traj.steps[j - 1].observation
                parts.append(f"Observation: {pre_obs}")
                if include_available_actions and step.available_actions:
                    parts.append(f"Available actions: {', '.join(step.available_actions)}")
                if include_thoughts and step.thought:
                    parts.append(f"Thought: {step.thought}")
                parts.append(f"Action: {step.action}")
                is_last = j == len(traj.steps) - 1
                if is_last:
                    if not traj.success:
                        parts.append("Step limit reached.")
                    parts.append(f"Total reward: {traj.total_reward}")
                parts.append("")
            parts.append("")
        return "\n".join(parts)

    def _parse_reflection(self, response: str):
        analysis_m = re.search(
            r"ANALYSIS:\s*(.+?)(?=IMPROVED PROMPT:|$)", response, re.DOTALL | re.IGNORECASE
        )
        analysis = analysis_m.group(1).strip() if analysis_m else ""

        prompt_m = re.search(
            r"IMPROVED PROMPT:\s*(.+)", response, re.DOTALL | re.IGNORECASE
        )
        improved_prompt = prompt_m.group(1).strip() if prompt_m else ""

        return analysis, improved_prompt

    @staticmethod
    def _call_with_retry(func, max_retries: int = 5, base_delay: float = 0.5):
        from openai import RateLimitError
        for attempt in range(max_retries):
            try:
                return func()
            except RateLimitError:
                if attempt == max_retries - 1:
                    raise
                time.sleep(base_delay * (2 ** attempt))
