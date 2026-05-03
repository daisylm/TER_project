from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any


PROMPTS_FILE = Path("generated_prompts/controlled_prompts.json")
OUTPUT_FILE = Path("generated_prompts/single_prompt_result.json")


def post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def load_prompt(path: Path, prompt_id: str | None) -> dict[str, Any]:
    prompts = json.loads(path.read_text(encoding="utf-8"))["prompts"]
    if prompt_id is None:
        return prompts[0]
    for prompt in prompts:
        if prompt["id"] == prompt_id:
            return prompt
    raise SystemExit(f"Prompt not found: {prompt_id}")


def run_command(command: list[str], output: Path | None = None) -> None:
    if output is None:
        subprocess.run(command, check=True)
        return
    with output.open("w", encoding="utf-8") as file:
        subprocess.run(command, stdout=file, check=True)


def run(args: argparse.Namespace) -> None:
    prompt = load_prompt(args.prompts, args.prompt_id)
    payload = {
        "model": args.model,
        "prompt": prompt["prompt"],
        "stream": False,
        "options": {"num_predict": args.num_predict},
    }

    perf = subprocess.Popen(
        ["sudo", "perf", "record", "-F", "99", "-e", "cycles", "-p", str(args.pid), "-g"]
    )

    start = time.perf_counter()
    error = None
    response = {}
    try:
        response = post_json(f"{args.server}/api/generate", payload, args.timeout)
    except Exception as exc:
        error = str(exc)
    elapsed = time.perf_counter() - start

    perf.terminate()
    perf.wait()

    result = {
        "prompt_id": prompt["id"],
        "category": prompt["category"],
        "title": prompt["title"],
        "model": args.model,
        "server": args.server,
        "execution_seconds": round(elapsed, 3),
        "response": response.get("response", ""),
        "ollama_metrics": {
            "total_duration": response.get("total_duration"),
            "load_duration": response.get("load_duration"),
            "prompt_eval_count": response.get("prompt_eval_count"),
            "prompt_eval_duration": response.get("prompt_eval_duration"),
            "eval_count": response.get("eval_count"),
            "eval_duration": response.get("eval_duration"),
        },
        "error": error,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    run_command(["sudo", "perf", "script"], Path("out.perf"))
    run_command(["./FlameGraph/stackcollapse-perf.pl", "out.perf"], Path("out.folded"))
    run_command(["./FlameGraph/flamegraph.pl", "out.folded"], Path("flamegraph.svg"))

    print(f"Wrote metrics to {args.output}")
    print("Wrote flamegraph.svg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one generated prompt through Ollama with perf.")
    parser.add_argument("--prompts", type=Path, default=PROMPTS_FILE)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--prompt-id")
    parser.add_argument("--server", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="tinyllama")
    parser.add_argument("--pid", type=int, default=2284)
    parser.add_argument("--num-predict", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=600)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
