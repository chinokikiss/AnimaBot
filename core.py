import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import asyncio
import json
import base64
import re
import random
import time
from pathlib import Path
import httpx
from comfyui_api import load_workflow, run_workflow
from utils import check_nsfw, log
from websockets.asyncio.client import connect

from AutoPrompt.agent_core import agent, extract_prompt_params

WS_URL = "ws://localhost:3001"
TOKEN = ""
MAX_CONCURRENT_REQUESTS_PER_USER = 2

echo_id = 0
pending = {}
active_user_requests = {}


def try_acquire_user_request(user_id):
    """Reserve one request slot for a user, returning whether it was available."""
    key = str(user_id)
    active_requests = active_user_requests.get(key, 0)
    if active_requests >= MAX_CONCURRENT_REQUESTS_PER_USER:
        return False

    active_user_requests[key] = active_requests + 1
    return True


def release_user_request(user_id):
    """Release a request slot and remove the user when no requests remain."""
    key = str(user_id)
    active_requests = active_user_requests.get(key, 0)
    if active_requests <= 1:
        active_user_requests.pop(key, None)
    else:
        active_user_requests[key] = active_requests - 1

async def extract_images_from_msg(msg_array, ws):
    """Download all referenced images while preserving message order."""
    visited_replies = set()

    async def collect_segments(segments, client):
        images = []
        for seg in segments or []:
            seg_type = seg.get("type")
            if seg_type == "image":
                file_url = seg.get("data", {}).get("url", "")
                if not file_url:
                    continue
                try:
                    resp = await client.get(file_url)
                    resp.raise_for_status()
                    images.append(resp.content)
                except Exception as e:
                    log(f"[Image] 下载图片失败: {e}")
            elif seg_type == "reply":
                reply_id = seg.get("data", {}).get("id")
                if not reply_id or reply_id in visited_replies:
                    continue
                visited_replies.add(reply_id)
                try:
                    resp = await call_api(ws, "get_msg", {"message_id": int(reply_id)})
                    orig_msg = resp.get("data", {}).get("message", [])
                    images.extend(await collect_segments(orig_msg, client))
                except Exception as e:
                    log(f"[Reply] 获取历史消息或提取图片失败: {e}")
        return images

    async with httpx.AsyncClient(timeout=30) as client:
        return await collect_segments(msg_array, client)


async def extract_image_from_msg(msg_array, ws):
    """Backward-compatible singular entry point returning the image list."""
    return await extract_images_from_msg(msg_array, ws)

async def anima(
    ws,
    id1,
    id2,
    is_group,
    user_text,
    user_msg_id,
    image=None,
    self_id=None,
    images=None,
):
    action = "send_group_msg" if is_group else "send_private_msg"
    param1 = "group_id" if is_group else "user_id"
    user_id = id2 if is_group else id1

    t0 = time.time()

    resp = await call_api(ws, action, {
        param1: id1,
        "message": [
            {"type": "at", "data": {"qq": str(user_id)}},
            {"type": "text", "data": {"text": " 呜…少女正在构思提示词中，请稍等片刻哦 (｡>﹏<｡)"}}
        ]
    })
    msg_id = resp.get("data", {}).get("message_id")

    prompt, width, height = await extract_prompt_params(user_text)

    reference_images = images if images is not None else image
    if reference_images is None:
        reference_images = []
    elif isinstance(reference_images, (bytes, bytearray, memoryview)):
        reference_images = [bytes(reference_images)]
    else:
        reference_images = list(reference_images)

    tags_prompt, natural_prompt, description, characters = await agent(
        prompt,
        images=reference_images,
    )

    t1 = time.time()

    if msg_id:
        await call_api(ws, "delete_msg", {"message_id": msg_id})

    resp = await call_api(ws, action, {
        param1: id1,
        "message": [
            {"type": "at", "data": {"qq": str(user_id)}},
            {"type": "text", "data": {"text": " 提示词生成完成啦～少女正在努力绘制图片中 ✨ヽ(●´∀`●)ﾉ"}}
        ]
    })
    msg_id = resp.get("data", {}).get("message_id")

    workflow = load_workflow(
        path=Path("workflows") / "image_anima_base_v1.json",
        overrides={
            "8": {"text": tags_prompt},
            "26": {"text": natural_prompt},
            "7": {"width": width, "height": height},
            "10": {"seed": random.randint(0, 2**32 - 1)},
        }
    )
    imgs = await run_workflow(workflow)
    img_bytes = imgs[0]
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    t2 = time.time()
    
    if msg_id:
        await call_api(ws, "delete_msg", {"message_id": msg_id})
    
    if is_group:
        is_nsfw = await check_nsfw(img_bytes)
    else:
        is_nsfw = False

    if not is_group or not is_nsfw:
        if msg_id:
            await call_api(ws, "delete_msg", {"message_id": msg_id})

        await call_api(ws, action, {
            param1: id1,
            "message": [
                {"type": "reply", "data": {"id": str(user_msg_id)}},
                {"type": "at", "data": {"qq": str(user_id)}},
                {"type": "image", "data": {"file": f"base64://{b64}"}}
            ]
        })
    else:
        if msg_id:
            await call_api(ws, "delete_msg", {"message_id": msg_id})

        await call_api(ws, action, {
            param1: id1,
            "message": [
                {"type": "reply", "data": {"id": str(user_msg_id)}},
                {"type": "at", "data": {"qq": str(user_id)}},
                {"type": "text", "data": {"text": "这张图片太害羞了，已经悄悄私发给你了哦 (⁄ ⁄•⁄ω⁄•⁄ ⁄)"}}
            ]
        })

        await call_api(ws, "send_private_msg", {
            "user_id": id2,
            "group_id": id1,
            "message": [
                {"type": "image", "data": {"file": f"base64://{b64}"}}
            ]
        })

    if self_id:
        pt = t1 - t0
        it = t2 - t1
        tt = t2 - t0
        size_str = f"{width}x{height}"
        characters = [f'"{char}"' for char in characters]
        char_str = ", ".join(characters) if characters else "无"
        params_nodes = [
            {
                "type": "node",
                "data": {
                    "name": "Anima",
                    "uin": str(self_id),
                    "content": [
                        {"type": "text", "data": {"text": f"场景描述: {description}"}}
                    ]
                }
            },
            {
                "type": "node",
                "data": {
                    "name": "Anima",
                    "uin": str(self_id),
                    "content": [
                        {"type": "text", "data": {"text": f"涉及角色: {char_str}"}}
                    ]
                }
            },
            {
                "type": "node",
                "data": {
                    "name": "Anima",
                    "uin": str(self_id),
                    "content": [
                        {"type": "text", "data": {"text": f"标签提示词: {tags_prompt}"}}
                    ]
                }
            },
            {
                "type": "node",
                "data": {
                    "name": "Anima",
                    "uin": str(self_id),
                    "content": [
                        {"type": "text", "data": {"text": f"自然语言提示词: {natural_prompt}"}}
                    ]
                }
            },
            {
                "type": "node",
                "data": {
                    "name": "Anima",
                    "uin": str(self_id),
                    "content": [
                        {
                            "type": "text", 
                            "data": {
                                "text": f"尺寸: {size_str}\n提示词耗时: {pt:.2f}s | 绘制耗时: {it:.2f}s | 总耗时: {tt:.2f}s"
                            }
                        }
                    ]
                }
            },
        ]
        if is_group:
            if is_nsfw:
                await call_api(ws, "send_private_forward_msg", {
                    "group_id": id1,
                    "user_id": id2,
                    "messages": params_nodes
                })
            else:
                await call_api(ws, "send_group_forward_msg", {
                    "group_id": id1,
                    "messages": params_nodes
                })
        else:
            await call_api(ws, "send_private_forward_msg", {
                "user_id": id1,
                "messages": params_nodes
            })

async def upscale(ws, id1, id2, is_group, user_msg_id, image):
    action = "send_group_msg" if is_group else "send_private_msg"
    param1 = "group_id" if is_group else "user_id"
    user_id = id2 if is_group else id1

    t0 = time.time()

    resp = await call_api(ws, action, {
        param1: id1,
        "message": [
            {"type": "at", "data": {"qq": str(user_id)}},
            {"type": "text", "data": {"text": " 少女正在努力放大图片中，请稍等一下哦 (๑˃̵ᴗ˂̵)و"}}
        ]
    })
    msg_id = resp.get("data", {}).get("message_id")

    b64 = base64.b64encode(image).decode("utf-8")

    workflow = load_workflow(
        path=Path("workflows") / "4x-upscale.json",
        overrides={
            "1": {"image_data": b64},
        }
    )
    imgs = await run_workflow(workflow)
    img_bytes = imgs[0]
    b64 = base64.b64encode(img_bytes).decode("utf-8")

    tt = time.time() - t0

    if msg_id:
        await call_api(ws, "delete_msg", {"message_id": msg_id})
    
    await call_api(ws, action, {
        param1: id1,
        "message": [
            {"type": "reply", "data": {"id": str(user_msg_id)}},
            {"type": "at", "data": {"qq": str(user_id)}},
            {"type": "image", "data": {"file": f"base64://{b64}"}},
            {"type": "text", "data": {"text": f"tt: {tt:.2f}s"}}
        ]
    })

async def call_api(ws, action: str, params: dict = None):
    global echo_id
    echo_id += 1
    payload = {"action": action, "params": params or {}, "echo": echo_id}
    fut = asyncio.get_event_loop().create_future()
    pending[echo_id] = fut
    await ws.send(json.dumps(payload))
    return await fut


async def handle_event(ws, data: dict):
    if "echo" in data:
        fut = pending.pop(data["echo"], None)
        if fut:
            fut.set_result(data)
        return

    post_type = data.get("post_type")

    if post_type == "message":
        msg_type = data.get("message_type")
        user_id = data.get("user_id")
        self_id = data.get("self_id")
        raw_msg = data.get("raw_message", "")
        message_id = data.get("message_id", "")
        msg_array = data.get("message", [])
        clean_msg = re.sub(r'^(?:\s|@\S+|\[CQ:[^\]]+\])*', '', raw_msg)
        is_drawing_command = clean_msg.startswith('绘图') or clean_msg.startswith('画图') or clean_msg.startswith('绘画') or clean_msg.startswith('画') or clean_msg.startswith('绘制')
        is_upscale_command = clean_msg.startswith('放大')
        
        if msg_type == "private":
            log(f"[私聊] {user_id}: {raw_msg}")
            image_bytes = await extract_images_from_msg(msg_array, ws)
            if is_drawing_command:
                if not try_acquire_user_request(user_id):
                    await call_api(ws, "send_private_msg", {
                        "user_id": user_id,
                        "message": [{"type": "text", "data": {"text": "当前有任务正在进行中，请耐心等待哦 (｡•́︿•̀｡)"}}]
                    })
                else:
                    try:
                        await anima(ws, user_id, None, False, raw_msg, message_id, images=image_bytes, self_id=self_id)
                    finally:
                        release_user_request(user_id)
            if is_upscale_command and image_bytes:
                if not try_acquire_user_request(user_id):
                    await call_api(ws, "send_private_msg", {
                        "user_id": user_id,
                        "message": [{"type": "text", "data": {"text": "当前有任务正在进行中，请耐心等待哦 (｡•́︿•̀｡)"}}]
                    })
                else:
                    try:
                        await upscale(ws, user_id, None, False, message_id, image_bytes[0])
                    finally:
                        release_user_request(user_id)

        elif msg_type == "group":
            group_id = data.get("group_id")
            log(f"[群聊] {group_id} | {user_id}: {raw_msg}")
            image_bytes = await extract_images_from_msg(msg_array, ws)
            if is_drawing_command:
                if not try_acquire_user_request(user_id):
                    await call_api(ws, "send_group_msg", {
                        "group_id": group_id,
                        "message": [
                            {"type": "at", "data": {"qq": str(user_id)}},
                            {"type": "text", "data": {"text": " 当前有任务正在进行中，请耐心等待哦 (｡•́︿•̀｡)"}}
                        ]
                    })
                else:
                    try:
                        await anima(ws, group_id, user_id, True, raw_msg, message_id, images=image_bytes, self_id=self_id)
                    finally:
                        release_user_request(user_id)
            if is_upscale_command and image_bytes:
                if not try_acquire_user_request(user_id):
                    await call_api(ws, "send_group_msg", {
                        "group_id": group_id,
                        "message": [
                            {"type": "at", "data": {"qq": str(user_id)}},
                            {"type": "text", "data": {"text": " 当前有任务正在进行中，请耐心等待哦 (｡•́︿•̀｡)"}}
                        ]
                    })
                else:
                    try:
                        await upscale(ws, group_id, user_id, True, message_id, image_bytes[0])
                    finally:
                        release_user_request(user_id)

    elif post_type == "notice":
        log(f"[通知] {json.dumps(data, ensure_ascii=False)}")

    elif post_type == "request":
        log(f"[请求] {json.dumps(data, ensure_ascii=False)}")


async def main():
    headers = {"Authorization": f"Bearer {TOKEN}"}
    while True:
        try:
            async with connect(WS_URL, additional_headers=headers, open_timeout=5) as ws:
                log(f"已连接到 {WS_URL}")
                async for raw in ws:
                    data = json.loads(raw)
                    asyncio.create_task(handle_event(ws, data))
            break
        except:
            pass

if __name__ == "__main__":
    asyncio.run(main())
