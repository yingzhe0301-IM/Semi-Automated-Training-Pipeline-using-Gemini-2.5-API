"""Quick preflight check for the Gemini SDK and API key."""

from __future__ import annotations

import argparse
import os

from google import genai

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
DEFAULT_PROMPT = "Reply with exactly the word OK."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that a Gemini API key is configured and can reach the API."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model used for the connectivity check. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt used for the connectivity check.",
    )
    parser.add_argument(
        "--skip-api-call",
        action="store_true",
        help="Only verify that an API key is present in the environment.",
    )
    return parser.parse_args()


def load_api_key() -> tuple[str | None, str | None]:
    if os.getenv("GOOGLE_API_KEY"):
        return os.getenv("GOOGLE_API_KEY"), "GOOGLE_API_KEY"
    if os.getenv("GEMINI_API_KEY"):
        return os.getenv("GEMINI_API_KEY"), "GEMINI_API_KEY"
    return None, None


def mask_api_key(api_key: str) -> str:
    if len(api_key) <= 8:
        return "*" * len(api_key)
    return f"{api_key[:4]}...{api_key[-4:]}"


def main() -> int:
    args = parse_args()
    api_key, source_name = load_api_key()
    if not api_key or not source_name:
        print("Missing API key. Set GOOGLE_API_KEY or GEMINI_API_KEY first.")
        return 1

    print(f"Found API key in {source_name}: {mask_api_key(api_key)}")
    if args.skip_api_call:
        print("Environment check succeeded. Skipping live API call.")
        return 0

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=args.model, contents=args.prompt)
    except Exception as exc:
        print(f"Gemini API check failed: {exc}")
        return 1

    text = (response.text or "").strip()
    print(f"Gemini API check succeeded with model '{args.model}'.")
    print(f"Response: {text!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
