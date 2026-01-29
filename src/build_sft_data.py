"""Create a small Alpaca-style JSON dataset for SFT demos."""

from __future__ import annotations

import argparse
import json
import os
import random

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INSTRUCTIONS = [
    "Write a short story about a brave mouse.",
    "Summarize the following topic in one sentence.",
    "Give two tips for learning Python.",
    "Explain what a rainbow is to a child.",
    "List three animals that live in the ocean.",
    "Provide a friendly greeting.",
    "Describe a sunny day in one paragraph.",
    "Suggest a healthy snack.",
    "Name two ways to save water.",
    "Write a haiku about winter.",
]

INPUTS = [
    "",
    "Topic: photosynthesis",
    "Topic: volcanoes",
    "Topic: friendship",
    "Topic: rain",
    "Topic: mountains",
]

OUTPUTS = [
    "The brave mouse crept through the tall grass, helping its friends find a safe path home.",
    "Photosynthesis is how plants use sunlight to make their own food.",
    "Practice small exercises daily and build simple projects to learn Python faster.",
    "A rainbow is colorful light in the sky made after rain when the sun shines through drops.",
    "Dolphins, sea turtles, and octopuses live in the ocean.",
    "Hello! I hope you are having a wonderful day.",
    "The sun warmed the fields, and a gentle breeze carried the scent of flowers across the yard.",
    "Try apple slices with peanut butter for a healthy snack.",
    "Fix leaks and turn off the tap while brushing your teeth to save water.",
    "Frosty wind whispers / Snowflakes dance on quiet roofs / Winter sings softly.",
]


def build_examples(count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    examples = []
    for _ in range(count):
        instruction = rng.choice(INSTRUCTIONS)
        user_input = rng.choice(INPUTS)
        output = rng.choice(OUTPUTS)
        examples.append(
            {
                "instruction": instruction,
                "input": user_input,
                "output": output,
            }
        )
    return examples


def main():
    parser = argparse.ArgumentParser(description="Build a tiny Alpaca-style JSON dataset")
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(ROOT_DIR, "data", "alpaca_sample.json"),
    )
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    examples = build_examples(args.count, args.seed)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(examples, file, ensure_ascii=False, indent=2)

    print(f"Wrote {len(examples)} examples to {args.output}")


if __name__ == "__main__":
    main()
