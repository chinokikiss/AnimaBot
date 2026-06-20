import json
import os
import httpx
from pathlib import Path

CONFIG_PATH = "config.json"
API_URL = "https://uapis.cn/api/v1/image/nsfw"

_logging_enabled = None

def _load_logging_flag():
    global _logging_enabled
    if _logging_enabled is not None:
        return _logging_enabled
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, encoding="utf-8") as f:
                _logging_enabled = json.load(f).get("logging", True)
        else:
            _logging_enabled = True
    except:
        _logging_enabled = True
    return _logging_enabled

def log(*args, **kwargs):
    if _load_logging_flag():
        print(*args, **kwargs)

async def check_nsfw(data) -> dict:
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(API_URL, files={"file": ("image.png", data, "image/png")})
            resp.raise_for_status()
            result = resp.json()
            nsfw_score = result.get("nsfw_score", 0)
            suggestion = result.get("suggestion", "pass")
            risk_level = result.get("risk_level", "low")
            is_nsfw = result.get("is_nsfw", False)
            return (
                is_nsfw
                or nsfw_score > 0.3
                or suggestion in ("review", "block")
                or risk_level in ("medium", "high")
            )
    except:
        log("审核API出错！")
        return True

def delete_images(folder_path):
    img_exts = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    for file in Path(folder_path).rglob("*"):
        if file.suffix.lower() in img_exts:
            try:
                file.unlink()
            except OSError:
                pass