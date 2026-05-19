"""Actor: LLM client for running ReACT-style episodes."""

import re
from typing import List, Optional, Tuple


def _edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for ch in a:
        curr = [prev[0] + 1]
        for j, bch in enumerate(b):
            curr.append(min(prev[j] + (0 if ch == bch else 1),
                            curr[-1] + 1,
                            prev[j + 1] + 1))
        prev = curr
    return prev[-1]


def find_best_action(proposed: str, available: List[str], threshold: int = 5) -> str:
    """Resolve a raw LLM action string to the closest valid action.

    Resolution order:
      1. Exact match.
      2. Case-insensitive match (handles LLM lowercasing "N" → "n").
      3. Lowest edit distance (handles "eat apple" → "EAT").
      Falls back to the first available action if no match is within threshold.
    """
    if not available:
        return proposed
    if proposed in available:
        return proposed
    lower = proposed.lower()
    for a in available:
        if a.lower() == lower:
            return a
    best = min(available, key=lambda a: _edit_distance(lower, a.lower()))
    if _edit_distance(lower, best.lower()) > threshold:
        return available[0]
    return best

from .trajectory import Step, Trajectory


def _parse_response(response: str) -> Tuple[str, str]:
    """Extract (thought, action) from a ReACT-style LLM response."""
    thought = ""
    action = ""
    thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|$)", response, re.DOTALL | re.IGNORECASE)
    if thought_match:
        thought = thought_match.group(1).strip()
    action_match = re.search(r"Action:\s*(.+?)(?=Thought:|$)", response, re.DOTALL | re.IGNORECASE)
    if action_match:
        action = action_match.group(1).strip().split("\n")[0].strip()
    return thought, action


class Actor:
    """LLM client that runs ReACT-style episodes via a vLLM server.

    When ``base_url`` is not provided, a vLLM server is started automatically
    for the given model and shut down when the Python process exits::

        actor = Actor("Qwen/Qwen2.5-7B-Instruct")

    The :meth:`call` method sends a single list of messages and returns the
    model response.  :meth:`run_episode` wraps it into a full ReACT loop for
    convenience.  :class:`~metagym.env.MetaGymEnv` uses :meth:`call` directly
    to batch N requests across N environments in lockstep.

    Args:
        model: HuggingFace model name or path.
        max_steps: Maximum environment steps per episode.
        temperature: Sampling temperature (0 = greedy/deterministic).
        max_tokens: Maximum tokens to generate per call.
        base_url: Base URL of an already-running vLLM server.  If ``None``,
                  a server is started automatically.
        history_steps: Number of most-recent steps to keep in the message
                       history sent to the LLM.  ``None`` = no limit (full
                       history).  The full history is always preserved in the
                       returned trajectories.
        server_kwargs: Extra keyword arguments forwarded to
                       :func:`~metagym.vllm_server.launch_vllm_server` when
                       auto-launching (e.g. ``tensor_parallel_size``,
                       ``max_model_len``).
    """

    def __init__(
        self,
        model: str,
        max_steps: int = 25,
        temperature: float = 0.0,
        max_tokens: int = 500,
        base_url: Optional[str] = None,
        history_steps: Optional[int] = None,
        **server_kwargs,
    ):
        self.model = model
        self.max_steps = max_steps
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.history_steps = history_steps

        if base_url is None:
            from .vllm_server import launch_vllm_server
            base_url = launch_vllm_server(model, **server_kwargs)
        self.base_url = base_url

        self._client = None  # lazy-init

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.base_url, api_key="dummy")
        return self._client

    def call(self, messages: List[dict]) -> str:
        """Send a chat messages list to the vLLM server and return the response.

        Args:
            messages: OpenAI-format message list
                      (``[{"role": "system", "content": "..."}, ...]``).

        Returns:
            The model's response string.
        """
        try:
            completion = self._get_client().chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            total_chars = sum(len(m.get("content", "")) for m in messages)
            print(f"[Actor] vLLM error: {e}")
            print(f"[Actor] messages: {len(messages)} messages, ~{total_chars} chars total")
            for i, m in enumerate(messages):
                content_preview = m.get("content", "")[:300]
                print(f"[Actor]   [{i}] role={m['role']} ({len(m.get('content',''))} chars): {content_preview!r}")
            raise
        return completion.choices[0].message.content

    def run_episode(
        self,
        env,
        system_prompt: str,
        user_prompt_template: str,
        game_id: str = "",
    ) -> Trajectory:
        """Run a full episode on ``env`` using the given system prompt.

        This is a convenience wrapper that drives a single episode end-to-end.
        For running a batch of episodes efficiently, use
        :meth:`~metagym.env.MetaGymEnv.step` instead, which batches LLM calls
        across all environments.

        Args:
            env: An inner environment with ``reset()`` returning
                 ``(observation, available_actions)`` and ``step(action)``
                 returning ``(observation, reward, done, available_actions)``.
            system_prompt: The system prompt that guides the agent.
            user_prompt_template: Template string with ``{task}``,
                ``{observation}``, and ``{available_actions}`` placeholders.
            game_id: Identifier for this game instance, forwarded to the
                     returned :class:`~metagym.trajectory.Trajectory`.

        Returns:
            A completed :class:`~metagym.trajectory.Trajectory`.
        """
        observation, available_actions = env.reset()
        task = getattr(env, "task", "")

        trajectory = Trajectory(
            game_id=game_id,
            task=task,
            initial_observation=observation,
            system_prompt=system_prompt,
        )

        messages = [{"role": "system", "content": system_prompt}]

        for _ in range(self.max_steps):
            user_content = user_prompt_template.format(
                task=task,
                observation=observation,
                available_actions="\n".join(f"- {a}" for a in available_actions),
            )
            messages.append({"role": "user", "content": user_content})
            response = self.call(messages)
            messages.append({"role": "assistant", "content": response})

            thought, action = _parse_response(response)
            observation, reward, done, available_actions = env.step(action)

            trajectory.add_step(Step(
                observation=observation,
                available_actions=available_actions,
                thought=thought,
                action=action,
                reward=reward,
            ))

            if reward > 0:
                trajectory.success = True
            if done:
                break

        return trajectory
