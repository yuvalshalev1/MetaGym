"""Trajectory and Step dataclasses for MetaGym episodes."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Step:
    """A single step in an episode."""
    observation: str
    available_actions: List[str]
    thought: str
    action: str
    reward: float = 0.0


@dataclass
class Trajectory:
    """A complete episode trajectory."""
    game_id: str
    task: str
    steps: List[Step] = field(default_factory=list)
    success: bool = False
    total_reward: float = 0.0
    initial_observation: str = ""
    system_prompt: str = ""

    def add_step(self, step: Step) -> None:
        self.steps.append(step)
        self.total_reward += step.reward

    def __len__(self) -> int:
        return len(self.steps)

    def to_dict(self) -> dict:
        return {
            "game_id": self.game_id,
            "task": self.task,
            "success": self.success,
            "total_reward": self.total_reward,
            "initial_observation": self.initial_observation,
            "system_prompt": self.system_prompt,
            "steps": [
                {
                    "observation": s.observation,
                    "available_actions": s.available_actions,
                    "thought": s.thought,
                    "action": s.action,
                    "reward": s.reward,
                }
                for s in self.steps
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Trajectory":
        traj = cls(
            game_id=d["game_id"],
            task=d["task"],
            success=d["success"],
            total_reward=d["total_reward"],
            initial_observation=d.get("initial_observation", ""),
            system_prompt=d.get("system_prompt", ""),
        )
        traj.steps = [
            Step(
                observation=s["observation"],
                available_actions=s["available_actions"],
                thought=s["thought"],
                action=s["action"],
                reward=s.get("reward", 0.0),
            )
            for s in d.get("steps", [])
        ]
        return traj

    def _pretty_print_to(self, console, show_thoughts: bool = True, show_available_actions: bool = True) -> None:
        from rich.rule import Rule

        console.print(Rule(f"[bold]Game:[/bold] {self.game_id}"))
        if self.system_prompt:
            console.print(f"[dim]System:[/dim] {self.system_prompt}\n")

        console.print(f"[bold cyan]Observation 0:[/bold cyan]\n{self.initial_observation}\n")

        for i, step in enumerate(self.steps):
            if show_thoughts and step.thought:
                console.print(f"[bold yellow]Thought {i+1}:[/bold yellow] {step.thought}")
            console.print(f"[bold green]Action {i+1}:[/bold green] {step.action}")
            if show_available_actions and step.available_actions:
                actions = ", ".join(step.available_actions)
                console.print(f"[dim]Available: {actions}[/dim]")
            console.print(f"[bold cyan]Observation {i+1}:[/bold cyan]\n{step.observation}\n")

        status = "[bold green]SUCCESS[/bold green]" if self.success else "[bold red]FAILURE[/bold red]"
        console.print(Rule(f"{status} — {len(self.steps)} steps, reward={self.total_reward:.2f}"))

    def pretty_print(self, show_thoughts: bool = True, show_available_actions: bool = True) -> None:
        """Print the full episode as a readable conversation."""
        from rich.console import Console
        self._pretty_print_to(Console(), show_thoughts, show_available_actions)

    def save_to_file(self, path: str, show_thoughts: bool = True, show_available_actions: bool = True) -> None:
        """Write the full episode as plain text to a file."""
        import os
        from rich.console import Console
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            self._pretty_print_to(
                Console(file=f, no_color=True, highlight=False),
                show_thoughts, show_available_actions,
            )
