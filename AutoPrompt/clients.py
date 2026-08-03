import json
import os

from openai import AsyncOpenAI


def load_config() -> dict:
    cfg_path = "config.json"
    if not os.path.exists(cfg_path):
        return {}
    with open(cfg_path, encoding="utf-8") as f:
        raw = f.read().strip()
        return json.loads(raw) if raw else {}


cfg = load_config()

client_cheap = AsyncOpenAI(
    api_key=cfg["cheap"]["api_key"],
    base_url=cfg["cheap"]["base_url"],
)
client_quality = AsyncOpenAI(
    api_key=cfg["quality"]["api_key"],
    base_url=cfg["quality"]["base_url"],
)
