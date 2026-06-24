import json
import re
from openai import AsyncOpenAI
from .LLM_Node import load_api_config, get_platform_settings


BCOLORS_WARNING = '\033[93m'
BCOLORS_FAIL = '\033[91m'
BCOLORS_ENDC = '\033[0m'

SYSTEM_PROMPT = """You are a drawing request parser. The user will give a request like "画一个女孩，横图，高质量，水彩风格".

Extract structured information from it. Return ONLY a JSON object with these fields:
- "prompt": the original text with resolution/quality keywords/绘图/画图/绘画/画/绘制 removed (keep the core subject description as-is)
- "width": image width in pixels (number, default 1024)
- "height": image height in pixels (number, default 1536)
- "steps": sampling steps (number, default 10)
- "cfg": CFG scale (number, default 1)

Resolution hints:
- 横图/wide/landscape/16:9 -> 1456x816
- 竖图/portrait/9:16/3:4 -> 1024x1536
- 方图/square/1:1 -> 1024x1024

Quality hints:
- 高质量 -> steps=30, cfg=5
- 快速 -> steps=10, cfg=1
- 默认 -> steps=10, cfg=1

Output ONLY valid JSON, no markdown, no explanation."""


async def extract_prompt_params(text: str):
    config = load_api_config()

    api_key = config.get("api_key", "")
    api_url = config.get("api_url", "")
    model_name = config.get("model_name", "")
    thinking = config.get("thinking", "")

    client = AsyncOpenAI(api_key=api_key, base_url=api_url)
    extra_body = get_platform_settings(api_url, model_name, thinking)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]

    response = await client.chat.completions.create(
        model=model_name, messages=messages,
        temperature=0.3, extra_body=extra_body,
    )

    raw = response.choices[0].message.content
    if not raw:
        raise ValueError("LLM 返回为空")

    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if json_match:
        raw = json_match.group()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError(f"LLM 返回非 JSON 格式: {raw[:200]}")
    
    use_agent = bool(re.search(r'[\u4e00-\u9fff]', data.get("prompt", text)))

    return data.get("prompt", text), int(data.get("width", 1024)), int(data.get("height", 1536)), int(data.get("steps", 10)), float(data.get("cfg", 1)), use_agent