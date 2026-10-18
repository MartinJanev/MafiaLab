from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mafialab.llm.types import LLMResponse


# saves every raw model reply of one game as JSON files in <output_dir>/turns, numbered in order
class TurnWriter:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.turns_dir = self.output_dir / "turns"
        self.turns_dir.mkdir(parents=True, exist_ok=True)
        self.count = 0

    def _sanitize_name(self, name: str) -> str:
        """
        Make a name safe for a file name, e.g. "Bob Smith" becomes "bob_smith".
        """
        return name.replace(" ", "_").replace("/", "_").lower()

    def save_turn(
        self,
        day: int,
        action_type: str,
        actor: str,
        response: LLMResponse,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        """
        Save one model reply as 07_vote_alice_response.json and so on, plus a _thinking file if
        the model returned its reasoning and a _metadata file with the provider's raw reply.
        :param day: The day or night number.
        :param action_type: The turn's label, e.g. "discussion", "vote" or "kill".
        :param actor: The player who replied.
        :param response: The model's reply.
        :param metadata: Extra fields to add to the _metadata file.
        :return: The written file paths, relative to output_dir.
        """
        actor_safe = self._sanitize_name(actor)
        self.count += 1
        base_name = f"{self.count:02d}_{action_type}_{actor_safe}"

        response_data = {
            "day": day,
            "action": action_type,
            "actor": actor,
            "model": response.model,
            "provider": response.provider,
            "latency_s": response.latency_s,
            "cached": response.cached,
            "finish_reason": response.finish_reason,
            "text": response.text,
            "usage": response.usage.model_dump(),
            "cost_usd": response.cost_usd,
        }
        response_file = self.turns_dir / f"{base_name}_response.json"
        response_file.write_text(json.dumps(response_data, indent=2) + "\n")

        result = {"response_file": str(response_file.relative_to(self.output_dir))}

        # only models that return their reasoning have this
        if response.thinking:
            thinking_data = {
                "day": day,
                "action": action_type,
                "actor": actor,
                "thinking": response.thinking,
            }
            thinking_file = self.turns_dir / f"{base_name}_thinking.json"
            thinking_file.write_text(json.dumps(thinking_data, indent=2) + "\n")
            result["thinking_file"] = str(thinking_file.relative_to(self.output_dir))

        # the provider's full raw JSON reply, plus any extra metadata
        if metadata or response.raw:
            metadata_data = {
                "day": day,
                "action": action_type,
                "actor": actor,
                "raw_response": response.raw,
            }
            if metadata:
                metadata_data.update(metadata)
            metadata_file = self.turns_dir / f"{base_name}_metadata.json"
            metadata_file.write_text(json.dumps(metadata_data, indent=2) + "\n")
            result["metadata_file"] = str(metadata_file.relative_to(self.output_dir))

        return result
