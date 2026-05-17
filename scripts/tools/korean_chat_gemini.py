#!/usr/bin/env python3
"""
Korean Chat with Google Gemini.

Configuration:
  - GEMINI_API_KEY: required
  - GEMINI_MODEL: optional, defaults to gemini-flash-latest
  - GEMINI_SYSTEM_PROMPT: optional, defaults to a Korean assistant prompt
  - .env: optional local file with KEY=VALUE pairs
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader: supports KEY=VALUE lines and comments."""
    p = Path(path)
    if not p.exists():
        return
    for raw_line in p.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def main() -> None:
    _load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("❌ Missing GEMINI_API_KEY. Put it in .env or export it in your shell.")
        print("   Example: export GEMINI_API_KEY='your_key_here'")
        raise SystemExit(1)

    model_name = os.getenv("GEMINI_MODEL", "gemini-flash-latest").strip()
    system_prompt = os.getenv(
        "GEMINI_SYSTEM_PROMPT",
        "You are a helpful Korean assistant. Respond in Korean whenever possible.",
    )

    try:
        genai = importlib.import_module("google.generativeai")
    except ModuleNotFoundError:
        print("❌ google-generativeai is not installed. Run: pip install google-generativeai")
        raise SystemExit(1)

    genai.configure(api_key=api_key)

    print("\n✅ Connected to Google Gemini!")
    print("=" * 70)
    print("🇰🇷 Korean Chat (Type 'quit' to exit)")
    print(f"Model: {model_name}")
    print("=" * 70)

    model = genai.GenerativeModel(model_name)

    while True:
        user_input = input("\n>>> ").strip()

        if user_input.lower() in {"quit", "exit", "나가기"}:
            print("안녕히 계세요!")
            break

        if not user_input:
            continue

        try:
            response = model.generate_content(f"{system_prompt}\n\nUser: {user_input}")
            print(f"\n🤖 {response.text}")
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()


