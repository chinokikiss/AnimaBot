from __future__ import annotations

import json
import logging

from .agent_prompts import _REFERENCE_SELECTION_SYSTEM_PROMPT, _THINKING
from .clients import cfg, client_cheap

logger = logging.getLogger(__name__)


def _norm(tag: str) -> str:
    return tag.strip().lower()


def parse_reference_selection(
    raw: str | dict,
    known_by_image: dict[str, list[str]],
) -> dict[str, list[str]]:
    """解析选择节点输出为 图像名 -> keep 标签列表。

    只接受输入中真实存在的标签（防止 LLM 幻觉新增）；
    遗漏未判定的标签一律默认保留（偏保真降级）。"""
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("参考图选择节点输出非 JSON，将默认全部保留。")
            return _all_keep(known_by_image)
    elif isinstance(raw, dict):
        payload = raw
    else:
        return _all_keep(known_by_image)

    known_by_norm = {
        image: {_norm(t): t for t in tags}
        for image, tags in known_by_image.items()
    }

    keep_by_image: dict[str, list[str]] = {}
    entries = payload.get("images") or []
    if not isinstance(entries, list):
        entries = []

    judged_images = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        image = str(entry.get("image") or "").strip()
        known = known_by_norm.get(image)
        if known is None:
            continue
        judged_images.add(image)
        keep = []
        for tag in entry.get("keep") or []:
            tag = str(tag).strip()
            if not tag:
                continue
            if _norm(tag) in known:
                keep.append(known[_norm(tag)])
        drop = set()
        for tag in entry.get("drop") or []:
            tag = str(tag).strip()
            if tag and _norm(tag) in known:
                drop.add(_norm(tag))
        keep_norm = {_norm(t) for t in keep}
        missing = [t for t in known_by_image[image] if _norm(t) not in keep_norm and _norm(t) not in drop]
        if missing:
            logger.warning("参考图选择节点遗漏标签 %s，默认保留。", missing)
            keep.extend(missing)
        keep_by_image[image] = keep

    for image, tags in known_by_image.items():
        if image not in keep_by_image:
            keep_by_image[image] = list(tags)
            logger.warning("参考图选择节点遗漏图像 %s，该图像标签默认全部保留。", image)

    return keep_by_image


def _all_keep(known_by_image: dict[str, list[str]]) -> dict[str, list[str]]:
    return {image: list(tags) for image, tags in known_by_image.items()}


async def select_reference_image_tags(
    user_description: str,
    img_tags: dict[str, dict],
) -> dict[str, list[str]]:
    """参考图意图与标签选择节点：返回 图像名 -> 保留标签列表。

    调用失败、输出非法时降级为全部保留，保证参考图内容不因节点故障丢失。"""
    known_by_image = {}
    for image, tags in img_tags.items():
        known_by_image[image] = list(tags.get("general") or []) + \
            list(tags.get("character") or []) + \
            list(tags.get("artist") or [])

    if not known_by_image:
        return {}

    try:
        resp = await client_cheap.chat.completions.create(
            model=cfg["cheap"]["model"],
            messages=[
                {"role": "system", "content": _REFERENCE_SELECTION_SYSTEM_PROMPT+"\n\n"+_THINKING},
                {
                    "role": "user",
                    "content": (
                        f"【用户原始描述】\n{user_description}\n\n"
                        f"【图像识别标签】\n{json.dumps(img_tags, ensure_ascii=False, indent=2)}"
                    ),
                },
            ],
            response_format={"type": "json_object"},
            reasoning_effort="low",
            extra_body={"thinking": {"type": "enabled"}},
            temperature=0.0,
        )

        print("-"*10)
        print(resp.choices[0].message.reasoning_content)
        print("-"*10)
    except Exception as e:
        logger.warning("参考图选择节点调用失败，将默认全部保留: %s", e)
        return _all_keep(known_by_image)

    return parse_reference_selection(resp.choices[0].message.content or "", known_by_image)
