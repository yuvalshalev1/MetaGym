"""MetaGymEnv: abstract base class for system-prompt-as-action environments."""

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .actor import Actor, _parse_response
from .trajectory import Step, Trajectory


@dataclass
class _EpisodeState:
    """Internal state for one episode during a lockstep batch run."""
    env: object
    game_id: str
    task: str
    observation: str
    available_actions: List[str]
    messages: List[dict]
    trajectory: Trajectory
    done: bool = False


class MetaGymEnv(ABC):
    """Abstract gym-like environment where the action is a system prompt.

    Each call to :meth:`step` runs a batch of full LLM episodes — one per
    inner environment — and returns all trajectories together with the
    fraction of episodes that succeeded.

    Episodes run in **lockstep**: at every step, the LLM is called once per
    active episode to collect all responses, and then each environment is
    stepped sequentially.  This structure ensures that the LLM calls for all
    active episodes at a given step happen before any environment is advanced,
    allowing vLLM's continuous batching scheduler to process them efficiently.

    Subclasses must implement:

    - :meth:`_make_env`: construct and return a single inner environment.
    - :meth:`_get_user_prompt_template`: return the per-step prompt template.

    Subclasses may optionally implement:

    - :meth:`_reset_env_to_game`: configure an inner env to play a specific
      game ID before the episode runs.  Required only if you want to use
      the ``game_ids`` argument of :meth:`step` / :meth:`reset`.

    Args:
        num_envs: Number of inner environments (= episodes per step).
        actor: The :class:`~metagym.actor.Actor` used to run episodes.
        default_system_prompt: System prompt used by :meth:`reset`.
        progress_bar: Show a rich progress bar during each batch run.
    """

    def __init__(
        self,
        num_envs: int,
        actor: Actor,
        default_system_prompt: str = "",
        progress_bar: bool = True,
    ):
        self.num_envs = num_envs
        self.actor = actor
        self.default_system_prompt = default_system_prompt
        self.progress_bar = progress_bar

        self._envs: List = []  # pool of inner envs, created on reset()

    # ------------------------------------------------------------------
    # Abstract interface — subclasses implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def _make_env(self):
        """Create and return a single inner environment instance.

        The returned object must implement::

            obs, available_actions = env.reset()
            obs, reward, done, available_actions = env.step(action)

        It should also expose a ``task`` attribute (str) and a ``game_id``
        attribute (str) after ``reset()`` is called, if available.
        """
        ...

    @abstractmethod
    def _get_user_prompt_template(self) -> str:
        """Return the per-step prompt template for this environment.

        The template must contain the placeholders ``{task}``,
        ``{observation}``, and ``{available_actions}``.
        """
        ...

    def _reset_env_to_game(self, env, game_id: Optional[str]) -> None:
        """Called before every episode to select which game the env will play.

        ``game_id`` is ``None`` when no specific game was requested (the env
        should pick whatever comes next in its own sequence).  Override this
        in subclasses that manage game selection (e.g. ALFWorld).

        The default implementation does nothing.

        Args:
            env: An inner environment created by :meth:`_make_env`.
            game_id: The game to load, or ``None`` for automatic selection.
        """

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_episode(
        self,
        env,
        system_prompt: str,
        game_id: Optional[str],
    ) -> _EpisodeState:
        """Reset an inner env and build the initial episode state."""
        self._reset_env_to_game(env, game_id)
        observation, available_actions = env.reset()
        task = getattr(env, "task", "")
        effective_game_id = getattr(env, "game_id", "") or game_id or ""
        return _EpisodeState(
            env=env,
            game_id=effective_game_id,
            task=task,
            observation=observation,
            available_actions=available_actions,
            messages=[{"role": "system", "content": system_prompt}],
            trajectory=Trajectory(
                game_id=effective_game_id,
                task=task,
                initial_observation=observation,
                system_prompt=system_prompt,
            ),
        )

    def _run_batch(
        self,
        system_prompt: str,
        game_ids: Optional[List[str]] = None,
    ) -> List[Trajectory]:
        """Run one episode per inner env using lockstep LLM calls.

        At each step:

        1. Build the user prompt for every active episode and append it to
           that episode's message history.
        2. Call the LLM for each active episode (all LLM calls happen before
           any environment is advanced).
        3. Step each environment sequentially with its corresponding action.
        4. Repeat until all episodes are done or ``actor.max_steps`` reached.

        Args:
            system_prompt: System prompt for all episodes in this batch.
            game_ids: Optional list of game IDs (one per env).  If provided
                      its length must match ``num_envs``.
        """
        if game_ids is not None and len(game_ids) != self.num_envs:
            raise ValueError(
                f"game_ids length ({len(game_ids)}) must match num_envs ({self.num_envs})"
            )

        game_ids_list = game_ids or [None] * self.num_envs
        template = self._get_user_prompt_template()

        # Initialize all episodes
        states = [
            self._init_episode(env, system_prompt, gid)
            for env, gid in zip(self._envs, game_ids_list)
        ]

        def _windowed(messages):
            history_steps = getattr(self.actor, "history_steps", None)
            if history_steps is None:
                return messages
            header, body = messages[:1], messages[1:]
            return header + body[-(history_steps * 2):]

        def _run_loop(progress=None, task_id=None):
            for step_num in range(self.actor.max_steps):
                active = [s for s in states if not s.done]
                done_count = self.num_envs - len(active)
                if progress:
                    progress.update(task_id, completed=step_num, done=done_count)
                if not active:
                    break

                for s in active:
                    user_content = template.format(
                        task=s.task,
                        observation=s.observation,
                        available_actions="\n".join(f"- {a}" for a in s.available_actions),
                    )
                    s.messages.append({"role": "user", "content": user_content})

                with ThreadPoolExecutor(max_workers=len(active)) as pool:
                    responses = list(pool.map(
                        lambda s: self.actor.call(_windowed(s.messages)), active
                    ))

                for s, response in zip(active, responses):
                    s.messages.append({"role": "assistant", "content": response})
                    thought, action = _parse_response(response)
                    obs, reward, done, available_actions = s.env.step(action)
                    s.observation = obs
                    s.available_actions = available_actions
                    s.done = done

                    s.trajectory.add_step(Step(
                        observation=obs,
                        available_actions=available_actions,
                        thought=thought,
                        action=action,
                        reward=reward,
                    ))

                    if reward > 0:
                        s.trajectory.success = True

                if progress:
                    progress.update(task_id, completed=step_num + 1,
                                    done=self.num_envs - sum(1 for s in states if not s.done))

        if self.progress_bar:
            from rich.progress import Progress, BarColumn, TextColumn, TaskProgressColumn, SpinnerColumn
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("[cyan]{task.completed}/{task.total} steps"),
                TextColumn("[dim]|[/dim] [green]{task.fields[done]}/{task.fields[num_envs]}[/green] done"),
                transient=True,
            ) as progress:
                task_id = progress.add_task(
                    f"Running batch ({self.num_envs} envs)",
                    total=self.actor.max_steps,
                    done=0,
                    num_envs=self.num_envs,
                )
                _run_loop(progress, task_id)
        else:
            _run_loop()

        return [s.trajectory for s in states]

    # ------------------------------------------------------------------
    # Gym-style API
    # ------------------------------------------------------------------

    def reset(
        self,
        system_prompt: Optional[str] = None,
        game_ids: Optional[List[str]] = None,
        skip_batch: bool = False,
    ) -> Tuple[List[Trajectory], Dict[str, bool]]:
        """Create the inner environment pool and run an initial batch.

        Args:
            system_prompt: System prompt for the initial batch.
                           Defaults to ``default_system_prompt``.
            game_ids: Optional list of specific game IDs to run (one per env).
                      Requires :meth:`_reset_env_to_game` to be implemented.
            skip_batch: If ``True``, create the inner env pool but do **not**
                        run any episodes.  Returns ``([], {})``.  Useful for
                        pre-warming the pool before the first :meth:`step` call.

        Returns:
            ``(trajectories, info)`` where ``trajectories`` is a list of
            :class:`~metagym.trajectory.Trajectory` objects and ``info`` is
            a dict mapping each ``game_id`` to whether that episode succeeded
            (``{game_id: is_won}``).  Both are empty when ``skip_batch=True``.
        """
        self.close()
        self._envs = [self._make_env() for _ in range(self.num_envs)]

        if skip_batch:
            return [], {}

        prompt = system_prompt if system_prompt is not None else self.default_system_prompt
        trajectories = self._run_batch(prompt, game_ids)
        info = {t.game_id: t.success for t in trajectories}
        return trajectories, info

    def step(
        self,
        system_prompt: str,
        game_ids: Optional[List[str]] = None,
    ) -> Tuple[List[Trajectory], float, bool, Dict[str, bool]]:
        """Run one batch of episodes using the given system prompt.

        Each inner environment is reset (advancing to its next game, or to a
        specific game if ``game_ids`` is provided) and then played to
        completion.

        Args:
            system_prompt: The system prompt to evaluate.
            game_ids: Optional list of specific game IDs to run (one per env).
                      Requires :meth:`_reset_env_to_game` to be implemented.

        Returns:
            ``(trajectories, reward, done, info)`` where:

            - ``trajectories``: list of :class:`~metagym.trajectory.Trajectory`
            - ``reward``: fraction of episodes that succeeded (float in [0, 1])
            - ``done``: always ``False`` (the meta-env has infinite horizon)
            - ``info``: dict mapping each ``game_id`` to whether it succeeded
              (``{game_id: is_won}``)
        """
        if not self._envs:
            raise RuntimeError("Call reset() before step().")

        trajectories = self._run_batch(system_prompt, game_ids)
        info = {t.game_id: t.success for t in trajectories}
        reward = sum(info.values()) / self.num_envs
        return trajectories, reward, False, info

    def close(self) -> None:
        """Close all inner environments and release resources."""
        for env in self._envs:
            if hasattr(env, "close"):
                env.close()
        self._envs = []
