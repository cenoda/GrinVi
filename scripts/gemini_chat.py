#!/usr/bin/env python3
"""
scripts/gemini_chat.py

Simple Gemini helper that reads secrets from .env or environment variables.

Usage:
  python scripts/gemini_chat.py --prompt "안녕하세요"
  python scripts/gemini_chat.py                 # interactive mode
"""
from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path


def load_dotenv(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for raw_line in p.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def build_model():
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Missing GEMINI_API_KEY. Put it in .env or export it in your shell.")

    model_name = os.getenv("GEMINI_MODEL", "gemini-flash-latest").strip()
    system_prompt = os.getenv(
        "GEMINI_SYSTEM_PROMPT",
        "You are a helpful Korean assistant. Respond in Korean whenever possible.",
    )

    try:
        genai = importlib.import_module("google.genai")
    except ModuleNotFoundError:
        raise SystemExit("google-genai is not installed. Run: pip install google-genai")

    client = genai.Client(api_key=api_key)
    return client, model_name, system_prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=None, help="Prompt to send; if omitted, interactive mode starts")
    args = parser.parse_args()

    client, model_name, system_prompt = build_model()
    print(f"Connected to Gemini model: {model_name}")

    def ask(prompt: str) -> str:
        response = client.models.generate_content(
            model=model_name,
            contents=f"{system_prompt}\n\nUser: {prompt}",
        )
        return response.text

    if args.prompt:
        print(ask(args.prompt))
        return

    while True:
        user_input = input(">>> ").strip()
        if user_input.lower() in {"quit", "exit", "나가기"}:
            break
        if not user_input:
            continue
        print(ask(user_input))


if __name__ == "__main__":
    main()

