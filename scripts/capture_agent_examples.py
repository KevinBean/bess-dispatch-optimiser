"""Capture real LangGraph agent answers offline → docs/agent_examples.json.

The public demo runs keyless, so the Advisor tab can't call the LLM live. This
captures genuine agent output (one dispatch question, one market-mechanics
question with citations) so the demo can SHOW what the agent produces.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bess.agent import ask  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "agent_examples.json"
QUESTIONS = [
    "What's the optimal dispatch for a 100 MW / 200 MWh battery in SA1 tomorrow, and how much could it make?",
    "Why do negative prices happen in the NEM, and should my battery charge during them?",
]


def main() -> None:
    examples = []
    for q in QUESTIONS:
        print("Q:", q)
        a = ask(q)
        print("A:", a[:120], "...\n")
        examples.append({"question": q, "answer": a})
    OUT.write_text(json.dumps({"examples": examples}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
