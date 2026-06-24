from __future__ import annotations

import asyncio
import json
import sys
import time
import httpx

_MCP_URL_HF = "https://sakizuki-danboorusearch.hf.space/mcp/mcp"
_MCP_URL_MS = "https://sakizuki-danboorusearchonline.ms.show/mcp/mcp"
_TIMEOUT = 90

_active_url: str = _MCP_URL_HF


def get_active_endpoint() -> str:
    return "ms" if _active_url == _MCP_URL_MS else "hf"

_FALLBACK_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_tags",
            "description": (
                "Search Danbooru tags using natural language and return a ready-to-use prompt. "
                "Chinese recommended. Use search_mode to pick the preset strategy that matches your intent: "
                "'full_scene' for complete scene → prompt (e.g. descriptions of a character in an environment), "
                "'concept_explore' for vague concept exploration with broad recall (e.g. 'cyberpunk clothing', 'bunny ears'), "
                "'subject_describe' for describing a specific subject to find matching tags (e.g. 'blue-haired pilot from EVA'), "
                "'precise_lookup' for precise lookup or spell fix (e.g. 'serafuku', 'thighhigh'). "
                "Use category to filter results by tag type (all/general/copyright/character)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query":       {"type": "string", "description": "Natural language description (Chinese recommended)."},
                    "search_mode": {"type": "string", "description": "Preset strategy: 'full_scene' (scene→prompt), 'concept_explore' (broad recall), 'subject_describe' (subject matching), 'precise_lookup' (spell fix). Default 'full_scene'."},
                    "category":    {"type": "string", "description": "Filter results to a tag category. 'all' (default), 'general' (visual attributes/clothing/pose), 'copyright' (anime/game titles), 'character' (named characters)."},
                    "show_nsfw":   {"type": "boolean", "description": "Include NSFW tags (default True)."},
                    "include_wiki":{"type": "boolean", "description": "Append wiki description per tag (default False). Use for disambiguation or explaining tags."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_related_tags",
            "description": (
                "Co-occurrence-based tag recommendations via NPMI scoring. Call after search_tags "
                "to discover complementary tags (accessories, character features, scene atmosphere). "
                "Multi-tag input returns intersection recommendations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tags":         {"type": "array", "items": {"type": "string"}, "description": "Danbooru tag names to base recommendations on."},
                    "limit":        {"type": "integer", "description": "Max recommendations returned."},
                    "show_nsfw":    {"type": "boolean", "description": "Include NSFW tags."},
                    "include_wiki": {"type": "boolean", "description": "Append wiki description per tag (default False). Use for disambiguation or explaining tags."},
                },
                "required": ["tags"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_artist_recommendations",
            "description": (
                "Recommend artists who are skilled at drawing the given tags, based on NPMI co-occurrence data. "
                "Given a list of Danbooru tags (e.g. character names, clothing, styles), this tool returns "
                "artists whose works frequently co-occur with those tags on Danbooru, ranked by aggregated NPMI score."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tags":      {"type": "array", "items": {"type": "string"}, "description": "Danbooru tag names to base artist recommendations on."},
                    "limit":     {"type": "integer", "description": "Max artists returned. Default 30."},
                    "min_cooc":  {"type": "integer", "description": "Minimum co-occurrence count per (tag, artist) pair. Default 3."},
                    "show_nsfw": {"type": "boolean", "description": "Include NSFW artist data. Default True."},
                },
                "required": ["tags"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_anima_format",
            "description": "返回 Anima 文生图模型的 Hybrid 混合提示词格式规范。当用户提到 Anima 提示词/Anima 格式/Anima Prompt/Anima 模型等关键词时，应在搜索标签完成、最终输出前调用此工具，以获取完整的提示词组装规范。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_newbie_format",
            "description": "返回 NewBie 文生图模型的 XML 格式提示词规范。当用户提到 NewBie 提示词/NewBie 格式/NewBie Prompt/NewBie 模型等关键词时，应在搜索标签完成、最终输出前调用此工具，以获取完整的 XML 格式组装规范。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

REPLACE_PROMPT_TOOL = {
    "type": "function",
    "function": {
        "name": "replace_prompt",
        "description": "在最终自检阶段，用于修改已生成的提示词内容。可进行精确字符串替换（提供 old_string 精确匹配要替换的文本），或完全重写（仅提供 new_string 作为全新内容）。",
        "parameters": {
            "type": "object",
            "properties": {
                "old_string": {
                    "type": "string",
                    "description": "要替换的旧文本，需精确匹配原文中的一段字符。不提供此参数则替换整个提示词内容。"
                },
                "new_string": {
                    "type": "string",
                    "description": "替换后的新文本。若省略 old_string，则此参数作为全新的提示词内容。"
                },
            },
            "required": ["new_string"],
        },
    },
}

_HEADERS_BASE = {
    "Content-Type": "application/json",
    "Accept":       "application/json, text/event-stream",
}


async def _new_session_id(client: httpx.AsyncClient, url: str) -> str:
    payload = {
        "jsonrpc": "2.0",
        "id":      0,
        "method":  "initialize",
        "params":  {
            "protocolVersion": "2024-11-05",
            "clientInfo":      {"name": "prompt-agent", "version": "1.0"},
            "capabilities":    {},
        },
    }
    resp = await client.post(url, json=payload, headers=_HEADERS_BASE)
    session_id = resp.headers.get("mcp-session-id")
    if not session_id:
        resp.raise_for_status()
        raise RuntimeError("MCP initialize 响应中没有 mcp-session-id header")
    return session_id


def _parse_response(resp: httpx.Response) -> dict:
    ct = resp.headers.get("content-type", "")
    if "text/event-stream" in ct:
        for line in resp.text.splitlines():
            if line.startswith("data:"):
                return json.loads(line[5:].strip())
        raise RuntimeError("SSE 响应中没有 data: 行")
    return resp.json()


async def _rpc(method: str, params: dict, req_id: int = 1) -> dict:
    global _active_url

    last_error = None
    for url in (_MCP_URL_HF, _MCP_URL_MS):
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                session_id = await _new_session_id(client, url)
                payload = {
                    "jsonrpc": "2.0",
                    "id":      req_id,
                    "method":  method,
                    "params":  params,
                }
                resp = await client.post(
                    url,
                    json=payload,
                    headers={**_HEADERS_BASE, "mcp-session-id": session_id},
                )
                resp.raise_for_status()
                rpc_resp = _parse_response(resp)

            if "error" in rpc_resp:
                raise RuntimeError(rpc_resp["error"].get("message", f"{method} 返回错误"))

            if url == _MCP_URL_MS:
                print("[tools] HF 端点本次失败，已回退 MS", file=sys.stderr)
            _active_url = url
            return rpc_resp.get("result", {})

        except Exception as e:
            last_error = e
            continue

    raise last_error  # type: ignore


def _probe(url: str, timeout: float = 20) -> dict:
    init_payload = {
        "jsonrpc": "2.0",
        "id":      0,
        "method":  "initialize",
        "params":  {
            "protocolVersion": "2024-11-05",
            "clientInfo":      {"name": "prompt-agent-health", "version": "1.0"},
            "capabilities":    {},
        },
    }
    t0 = time.monotonic()
    try:
        init_resp = httpx.post(url, json=init_payload, headers=_HEADERS_BASE, timeout=timeout)
        init_resp.raise_for_status()
        session_id = init_resp.headers.get("mcp-session-id")
        if not session_id:
            return {"ok": False, "error": "响应中没有 mcp-session-id"}

        call_payload = {
            "jsonrpc": "2.0",
            "id":      1,
            "method":  "tools/call",
            "params":  {
                "name": "search_tags",
                "arguments": {"query": "1girl", "search_mode": "precise_lookup"},
            },
        }
        call_resp = httpx.post(
            url, json=call_payload,
            headers={**_HEADERS_BASE, "mcp-session-id": session_id},
            timeout=timeout,
        )
        call_resp.raise_for_status()
        latency = round((time.monotonic() - t0) * 1000)

        rpc_resp = _parse_response(call_resp)
        if "error" in rpc_resp:
            return {"ok": False, "error": rpc_resp["error"].get("message", "tools/call 返回错误")}

        return {"ok": True, "latency_ms": latency}

    except httpx.TimeoutException:
        return {"ok": False, "error": "超时，服务可能正在冷启动"}
    except httpx.HTTPStatusError as e:
        return {"ok": False, "error": f"HTTP {e.response.status_code}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def check_mcp_health(timeout: float = 15) -> dict:
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as pool:
        hf_f = pool.submit(_probe, _MCP_URL_HF, timeout)
        ms_f = pool.submit(_probe, _MCP_URL_MS, timeout)
    return {"hf": hf_f.result(), "ms": ms_f.result(), "active": get_active_endpoint()}


async def load_tools_from_mcp() -> list[dict]:
    try:
        result = await _rpc("tools/list", {})
        mcp_tools = result.get("tools", [])
        if not mcp_tools:
            raise RuntimeError("tools/list 返回的工具列表为空")

        converted = []
        for t in mcp_tools:
            converted.append({
                "type": "function",
                "function": {
                    "name":        t["name"],
                    "description": t.get("description", ""),
                    "parameters":  t.get("inputSchema", {"type": "object", "properties": {}}),
                },
            })

        print(f"[tools] 已从 MCP 加载 {len(converted)} 个工具定义", file=sys.stderr)
        return converted

    except Exception as e:
        print(f"[tools] tools/list 拉取失败，使用回退定义：{e}", file=sys.stderr)
        return _FALLBACK_TOOLS


_TOOLS_CACHE = None

async def get_tools():
    global _TOOLS_CACHE
    if _TOOLS_CACHE is None:
        _TOOLS_CACHE = await load_tools_from_mcp()
    return _TOOLS_CACHE

TOOLS = []


async def _call_mcp(tool_name: str, arguments: dict) -> dict:
    retry_delays = [5, 10, 20]
    last_error = None

    for attempt in range(len(retry_delays) + 1):
        try:
            result = await _rpc("tools/call", {"name": tool_name, "arguments": arguments})

            content_blocks = result.get("content", [])
            for block in content_blocks:
                if block.get("type") == "text":
                    try:
                        return json.loads(block["text"])
                    except Exception:
                        return {"raw": block["text"]}

            return {"error": "MCP 返回内容为空"}

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429 and attempt < len(retry_delays):
                delay = retry_delays[attempt]
                print(
                    f"[tools] 429 限流，{delay}s 后重试（第 {attempt + 1}/{len(retry_delays)} 次）…",
                    file=sys.stderr,
                )
                await asyncio.sleep(delay)
                last_error = e
                continue
            return {"error": str(e)}
        except httpx.TimeoutException:
            return {"error": "请求超时，DanbooruSearch 服务可能正在冷启动，请稍后重试"}
        except Exception as e:
            return {"error": str(e)}

    return {"error": f"429 限流，已重试 {len(retry_delays)} 次仍失败: {str(last_error)}"}


async def execute_search_tags(
    query: str,
    search_mode: str = "full_scene",
    category: str = "all",
    show_nsfw: bool = True,
    include_wiki: bool = False,
) -> str:
    data = await _call_mcp("search_tags", {
        "query":        query,
        "search_mode":  search_mode,
        "category":     category,
        "show_nsfw":    show_nsfw,
        "include_wiki": include_wiki,
    })

    if "error" in data:
        return json.dumps({"error": data["error"]}, ensure_ascii=False)

    return json.dumps(data, ensure_ascii=False, indent=2)


async def execute_get_related_tags(
    tags: list[str],
    limit: int = 30,
    show_nsfw: bool = True,
    include_wiki: bool = False,
) -> str:
    if not tags:
        return json.dumps({"error": "tags 列表为空"}, ensure_ascii=False)

    data = await _call_mcp("get_related_tags", {
        "tags":         tags,
        "limit":        limit,
        "show_nsfw":    show_nsfw,
        "include_wiki": include_wiki,
    })

    if "error" in data:
        return json.dumps({"error": data["error"]}, ensure_ascii=False)

    return json.dumps(data, ensure_ascii=False, indent=2)


async def execute_get_artist_recommendations(
    tags: list[str],
    limit: int = 30,
    min_cooc: int = 3,
    show_nsfw: bool = True,
) -> str:
    if not tags:
        return json.dumps({"error": "tags 列表为空"}, ensure_ascii=False)

    data = await _call_mcp("get_artist_recommendations", {
        "tags":      tags,
        "limit":     limit,
        "min_cooc":  min_cooc,
        "show_nsfw": show_nsfw,
    })

    if "error" in data:
        return json.dumps({"error": data["error"]}, ensure_ascii=False)

    return json.dumps(data, ensure_ascii=False, indent=2)


async def execute_get_anima_format() -> str:
    data = await _call_mcp("get_anima_format", {})

    if "error" in data:
        return json.dumps({"error": data["error"]}, ensure_ascii=False)

    if "raw" in data:
        return data["raw"]
    return json.dumps(data, ensure_ascii=False, indent=2)


async def execute_get_newbie_format() -> str:
    data = await _call_mcp("get_newbie_format", {})

    if "error" in data:
        return json.dumps({"error": data["error"]}, ensure_ascii=False)

    if "raw" in data:
        return data["raw"]
    return json.dumps(data, ensure_ascii=False, indent=2)
