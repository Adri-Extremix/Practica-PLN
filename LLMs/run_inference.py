from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from LLMs.client import OpenAICompatibleClient  # noqa: E402
from LLMs.config import LLMConfig  # noqa: E402
from LLMs.postprocess import parse_prediction  # noqa: E402
from LLMs.prompts import SYSTEM_PROMPT, build_user_prompt  # noqa: E402
from shared.dataset import load_posts, save_jsonl  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HOPE-EXP inference with an OpenAI-compatible LLM endpoint.")
    parser.add_argument("--input", required=True, help="Path to the dataset file.")
    parser.add_argument("--output", required=True, help="Path to output predictions as JSONL.")
    parser.add_argument("--env-file", default=".env", help="Environment file with model settings.")
    parser.add_argument("--limit", type=int, help="Optional maximum number of posts to process.")
    parser.add_argument("--verbose", action="store_true", help="Print one-line progress for each row.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = LLMConfig.from_env(args.env_file)
    client = OpenAICompatibleClient(config)
    posts = load_posts(args.input)

    if args.limit is not None:
        posts = posts[: args.limit]

    predictions: list[dict[str, object]] = []
    for post in posts:
        user_prompt = build_user_prompt(post.lang, post.title, post.selftext)
        raw_output = client.chat_completion(SYSTEM_PROMPT, user_prompt)
        prediction = parse_prediction(raw_output, post)
        row_prediction = {"row_id": post.row_id, **prediction.to_dict()}
        predictions.append(row_prediction)
        if args.verbose:
            print(f"Processed row_id={post.row_id} label={prediction.primary_label}")

    save_jsonl(predictions, args.output)
    print(f"Saved {len(predictions)} predictions to {args.output}")
    print(json.dumps({"model": config.model, "base_url": config.base_url}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
