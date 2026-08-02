import json
import os
import io
from imgutils.validate import anime_rating

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
    return anime_rating(io.BytesIO(data))[0] == "r18"