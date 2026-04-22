import logging
import os
import json
from typing import Dict, Any, List
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

APIFREE_BASE_URL = os.getenv("APIFREE_BASE_URL", "https://api.apifree.ai/v1")
APIFREE_KEY = os.getenv("APIFREE_KEY")

MODELS = {
    "gpt": "openai/gpt-4o-mini",
    "gemini": "google/gemini-2.5-pro/on-demand",
    "deepseek": "deepseek-ai/deepseek-v3.2/thinking",
    "grok": "xai/grok-4-fast",
    "minimax": "minimax/minimax-m2.5",
}

TEST_PROMPTS = [
    'Return exactly this JSON and nothing else: {"ok": true}',
    'Reply with this exact JSON only: {"ok": true}',
    'Output only valid JSON with one key named ok set to true.',
    'Answer with a single JSON object only.',
    '{"ok": true}',
]

def safe_dump(obj: Any) -> str:
    try:
        if hasattr(obj, "model_dump"):
            return json.dumps(obj.model_dump(), ensure_ascii=False, indent=2)
        if isinstance(obj, (dict, list)):
            return json.dumps(obj, ensure_ascii=False, indent=2)
        return str(obj)
    except Exception as e:
        return f"<could not serialize response: {e}>\n{repr(obj)}"


def extract_text(resp: Any) -> str:
    choices = getattr(resp, "choices", None)
    if not choices:
        raise RuntimeError(f"No choices in response. Raw response:\n{safe_dump(resp)}")

    first = choices[0]
    finish_reason = getattr(first, "finish_reason", None)
    message = getattr(first, "message", None)

    if message is None:
        raise RuntimeError(f"First choice has no message. Raw response:\n{safe_dump(resp)}")

    content = getattr(message, "content", None)

    if isinstance(content, str):
        text = content.strip()
        if text:
            return text

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and item.get("text"):
                    parts.append(item["text"])
            else:
                item_type = getattr(item, "type", None)
                item_text = getattr(item, "text", None)
                if item_type == "text" and item_text:
                    parts.append(item_text)

        text = "\n".join(parts).strip()
        if text:
            return text

    refusal = getattr(message, "refusal", None)
    if refusal:
        raise RuntimeError(f"Model refusal: {refusal}")

    raw = safe_dump(resp)
    if finish_reason == "content_filter" or "SAFETY_CHECK" in raw:
        raise RuntimeError(f"Content filtered by provider. Raw response:\n{raw}")

    raise RuntimeError(f"Could not extract text content. Raw response:\n{raw}")


def try_parse_json(text: str) -> Dict[str, Any]:
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        return json.loads(candidate)

    raise ValueError(f"Could not parse JSON from text: {text}")


def provider_error_from_response(resp: Any) -> str | None:
    try:
        raw = resp.model_dump() if hasattr(resp, "model_dump") else resp
    except Exception:
        return None

    if isinstance(raw, dict) and raw.get("error"):
        err = raw["error"]
        return f"{err.get('type')} - {err.get('message')}"
    return None


def test_prompt(client: OpenAI, model_id: str, prompt: str) -> Dict[str, Any]:
    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        provider_error = provider_error_from_response(resp)
        if provider_error:
            return {
                "status": "failed",
                "prompt": prompt,
                "error": f"Provider error: {provider_error}",
                "raw_response": safe_dump(resp),
            }

        text = extract_text(resp)

        parsed = None
        parse_ok = False
        parse_error = None
        try:
            parsed = try_parse_json(text)
            parse_ok = True
        except Exception as e:
            parse_error = str(e)

        return {
            "status": "success",
            "prompt": prompt,
            "text": text,
            "json_parse_ok": parse_ok,
            "parsed_json": parsed,
            "parse_error": parse_error,
        }

    except Exception as e:
        return {
            "status": "failed",
            "prompt": prompt,
            "error": str(e),
        }


def test_model(client: OpenAI, alias: str, model_id: str) -> Dict[str, Any]:
    print("=" * 90)
    print(f"Testing alias={alias} | model={model_id}")

    attempts: List[Dict[str, Any]] = []
    first_success = None

    for i, prompt in enumerate(TEST_PROMPTS, start=1):
        print(f"\nPrompt #{i}: {prompt}")
        result = test_prompt(client, model_id, prompt)
        attempts.append(result)

        if result["status"] == "success":
            print("STATUS: SUCCESS")
            print("TEXT:")
            print(result["text"])
            print("JSON PARSE OK:", result["json_parse_ok"])
            if result["json_parse_ok"]:
                print("PARSED JSON:", result["parsed_json"])
                first_success = result
                break
            else:
                print("PARSE ERROR:", result["parse_error"])
        else:
            print("STATUS: FAILED")
            print("ERROR:", result["error"])

    overall_status = "success" if first_success else "failed"
    summary = {
        "alias": alias,
        "model": model_id,
        "status": overall_status,
        "attempts": attempts,
    }

    if first_success:
        summary["winning_prompt"] = first_success["prompt"]
        summary["text"] = first_success["text"]
        summary["parsed_json"] = first_success["parsed_json"]

    return summary


def main() -> None:
    if not APIFREE_KEY:
        raise RuntimeError(
            "Missing APIFREE_KEY environment variable. "
            "Set it in .env or in the shell before running this script."
        )

    client = OpenAI(api_key=APIFREE_KEY, base_url=APIFREE_BASE_URL)

    print(f"Using base URL: {APIFREE_BASE_URL}")
    print("APIFREE_KEY loaded:", bool(APIFREE_KEY))
    print()

    results = []
    for alias, model_id in MODELS.items():
        results.append(test_model(client, alias, model_id))
        print()

    print("=" * 90)
    print("SUMMARY")
    print(json.dumps(results, ensure_ascii=False, indent=2))

    output_file = "apifree_test_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()