import asyncio
import json
import time
import base64
import httpx
from utils import log

# 在此配置你的 ComfyUI 实例地址
COMFY_HOSTS = ["http://127.0.0.1:8188", "http://127.0.0.1:8189"]

_pick_lock = asyncio.Lock()


async def _get_queue_depth(host):
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{host}/queue", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return len(data.get("queue_running", [])) + len(data.get("queue_pending", []))


async def pick_idle_host():
    candidates = []
    for host in COMFY_HOSTS:
        try:
            depth = await _get_queue_depth(host)
            candidates.append((depth, host))
        except Exception:
            log(f"[ComfyUI] 实例 {host} 不可达，跳过")
            continue
    if not candidates:
        raise RuntimeError("所有 ComfyUI 实例均不可达")
    candidates.sort(key=lambda x: x[0])
    host = candidates[0][1]
    log(f"[ComfyUI] 选择 {host}（队列深度 {candidates[0][0]}）")
    return host


def load_workflow(path, overrides: dict | None = None):
    with open(path, encoding="utf-8") as f:
        workflow = json.load(f)
    if overrides:
        for node_id, inputs in overrides.items():
            if node_id in workflow:
                workflow[node_id]["inputs"].update(inputs)
    return workflow


async def post_prompt(workflow, host=None, client_id=None):
    if host is None:
        host = await pick_idle_host()
    body = {"prompt": workflow}
    if client_id:
        body["client_id"] = client_id
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{host}/prompt",
            json=body,
        )
        resp.raise_for_status()
        return resp.json()


async def get_history(prompt_id, host):
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{host}/history/{prompt_id}")
        resp.raise_for_status()
        return resp.json()


async def run_workflow(workflow, wait=True, poll_interval=1.0, timeout=300):
    async with _pick_lock:
        host = await pick_idle_host()
        result = await post_prompt(workflow, host=host)
    prompt_id = result["prompt_id"]
    log(f"Submitted prompt: {prompt_id} -> {host}")

    if not wait:
        return prompt_id

    start = time.time()
    while True:
        if time.time() - start > timeout:
            raise TimeoutError(f"Prompt {prompt_id} timed out on {host}")
        history_data = await get_history(prompt_id, host)
        if history_data and prompt_id in history_data:
            history = history_data[prompt_id]
            status = history.get("status", {})
            if status.get("completed"):
                results = []
                for node_id, node_output in history.get("outputs", {}).items():
                    if "images_data" in node_output:
                        for img in node_output["images_data"]:
                            results.append(base64.b64decode(img["data"]))
                return results
            elif status.get("status_str") == "error":
                raise RuntimeError(f"Execution error on {host}: {history}")
        await asyncio.sleep(poll_interval)
