import os
import re
import io
import asyncio
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from imgutils.tagging.pixai import get_pixai_tags
from .artist_recognition import artist_recognition

CSV_PATH = os.path.join(os.path.dirname(__file__), "tags_enhanced.csv")

_SCORE_RE = re.compile(r"^score_[1-9]$")
_WEIGHT_RE = re.compile(r"^\((?P<body>.+):(?P<weight>\d+(?:\.\d+)?)\)$")
_USER_WEIGHT_RE = re.compile(r"\(\s*([^():]+?)\s*:\s*(\d+(?:\.\d+)?)\s*\)")

TAG_CATEGORY_WEIGHT = {
    "character": 2.5,
    "copyright": 2.0,
    "style": 2.0,
    "appearance": 1.2,
    "clothing": 1.2,
    "composition": 1.0,
    "lighting": 1.0,
    "background": 0.8,
    "pose": 0.5,
    "fetish": 0.3,
    "nsfw": 0.05,
    "quality": 0,
}
DEFAULT_TAG_CATEGORY_WEIGHT = 1.0


def _normalize_tag_body(body: str) -> str:
    """把单个标签正文规范化为 Anima 要求的写法（小写、下划线转空格、括号转义）。"""
    body = body.strip().lower()
    is_artist = body.startswith("@")
    if is_artist:
        body = body[1:].strip()

    body = body.replace("\\(", "(").replace("\\)", ")")
    body = " ".join(
        token if _SCORE_RE.match(token) else token.replace("_", " ")
        for token in body.split()
    )
    body = " ".join(body.split())
    body = body.replace("(", "\\(").replace(")", "\\)")
    return f"@{body}" if is_artist and body else body


def _to_anima_weight(raw: str) -> str:
    """Anima 需要的权重远高于 SDXL（官方示例 (chibi:2)），1.1~1.3 基本不起作用。
    把用户按 SDXL 习惯写的小权重抬到 Anima 的基准强调档，已在区间内的原样保留。"""
    try:
        value = float(raw)
    except ValueError:
        return raw
    if value <= 1.0 or value >= 1.5:
        return raw
    return "2"


def reapply_user_weights(tags_prompt: str, original_text: str) -> str:
    """把用户在原始输入里显式写死的权重重新贴回最终标签上。

    用户权重要穿过「中文改写 → 标签检索 → 英文重组」三道 LLM 环节才能到达输出，
    靠提示词约束保不住（实测会被静默丢弃），因此在代码里兜底。
    仅对完全丢了权重的标签生效；模型已自行加权的标签不覆盖。"""
    wanted = {}
    for body, weight in _USER_WEIGHT_RE.findall(original_text or ""):
        key = _normalize_tag_body(body)
        if key:
            wanted[key] = _to_anima_weight(weight)
    if not wanted:
        return tags_prompt

    restored = []
    for raw in tags_prompt.split(","):
        tag = raw.strip()
        if not tag:
            continue
        if _WEIGHT_RE.match(tag):
            restored.append(tag)
            continue
        restored.append(f"({tag}:{wanted[tag]})" if tag in wanted else tag)
    return ", ".join(restored)


def normalize_anima_tags(tags_prompt: str) -> str:
    """规范化硬锚点层：score_N 保留下划线，其余下划线转空格，标签内括号转义，
    权重语法 (tag:1.2) 的外层括号保持原样。不做去重——多人结构会刻意重复动作标签。"""
    normalized = []
    for raw in tags_prompt.replace("\n", ", ").split(","):
        tag = raw.strip()
        if not tag:
            continue
        match = _WEIGHT_RE.match(tag)
        body, weight = (match.group("body"), match.group("weight")) if match else (tag, None)
        body = _normalize_tag_body(body)
        if not body:
            continue
        normalized.append(f"({body}:{weight})" if weight else body)
    return ", ".join(normalized)


_DEFAULT_TAG_WEIGHT = 1.0


def _artist_candidate_metrics(entry: dict, tag_weights: dict | None = None) -> tuple[float, float, float]:
    """提取画师候选的覆盖率、共现量与画师总帖数。

    覆盖率不再是单纯的 sources 数量：按标签类别权重（如角色 2.5、版权 2.0、
    质量 0）加权求和，角色/版权类标签对画师关联度的贡献远高于质量标签。
    sources 未在权重映射中的标签按中性权重 1.0 计；重复标签去重后只计一次。"""
    sources = entry.get("sources") or []
    if not isinstance(sources, list):
        sources = []
    weights = tag_weights or {}

    coverage = 0.0
    seen = set()
    for source in sources:
        key = str(source).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        coverage += weights.get(key, _DEFAULT_TAG_WEIGHT)

    try:
        cooc = max(float(entry.get("cooc_count") or 0), 0.0)
        post_count = max(float(entry.get("post_count") or 0), 0.0)
    except (TypeError, ValueError):
        cooc = post_count = 0.0
    return coverage, cooc, post_count


def rank_artist_candidates(results: list[dict], top_n: int = 10, tag_weights: dict | None = None) -> list[dict]:
    """画师候选预排序：加权 sources 覆盖率优先，其次按 共现浓度 × log(共现量)。

    覆盖多个输入标签的画师（如同时命中角色与场景）比单标签高共现的画师
    更契合整体需求。次级得分不能只用浓度（cooc/post）——那会让百帖级
    小画师凭 100% 浓度挤掉千帖级的题材大手，而作品量大的画师风格
    被生成模型学习得更充分，故乘 log(cooc) 平衡专精度与证据量。
    排序后只保留 top_n，避免低相关候选进入最终采样池。"""
    def _score(entry: dict) -> tuple:
        coverage, cooc, post_count = _artist_candidate_metrics(entry, tag_weights)
        ratio = cooc / post_count if post_count > 0 else 0.0
        return (coverage, ratio * np.log10(cooc + 1))

    return sorted(results, key=_score, reverse=True)[:top_n]


def sample_artist_candidate(results: list[dict], top_n: int = 10, tag_weights: dict | None = None) -> dict | None:
    """从高相关画师候选中按相关性加权随机抽取一个。

    候选先沿用 ``rank_artist_candidates`` 的排序逻辑限制在 top_n 内，
    再用覆盖率、共现浓度和共现量构成采样权重。这样高相关候选更常被选中，
    但不会像固定取第一名一样让结果失去随机性。
    """
    candidates = [
        result for result in results
        if isinstance(result, dict) and str(result.get("artist") or "").strip()
    ]
    candidates = rank_artist_candidates(candidates, top_n=top_n, tag_weights=tag_weights)
    if not candidates:
        return None

    weights = []
    for candidate in candidates:
        coverage, cooc, post_count = _artist_candidate_metrics(candidate, tag_weights)
        concentration = cooc / post_count if post_count > 0 else 0.0
        evidence = concentration * np.log10(cooc + 1.0)
        weights.append(max(float(coverage), 1.0) * max(evidence, 0.0))

    weights = np.asarray(weights, dtype=float)
    total_weight = weights.sum()
    probabilities = weights / total_weight if np.isfinite(total_weight) and total_weight > 0 else None
    selected_index = np.random.choice(len(candidates), p=probabilities)
    return candidates[int(selected_index)]


def sample_tags(weighted_k=3, random_k=2, max_threshold=None, encoding='utf-8'):
    try:
        df = pd.read_csv(CSV_PATH, encoding=encoding)
    except UnicodeDecodeError:
        df = pd.read_csv(CSV_PATH, encoding='gb18030')

    df['post_count'] = pd.to_numeric(df['post_count'], errors='coerce').fillna(0)
    
    if max_threshold is not None:
        df_filtered = df[(df['post_count'] > 0) & (df['post_count'] <= max_threshold)].copy()
    else:
        df_filtered = df[df['post_count'] > 0].copy()
    
    if df_filtered.empty:
        print("没有找到符合条件的有效 post_count 数据进行采样。")
        return pd.DataFrame()

    total_len = len(df_filtered)
    
    actual_weighted_k = min(weighted_k, total_len)
    actual_random_k = min(random_k, total_len - actual_weighted_k)

    sampled_indices_weighted = []
    sampled_indices_random = []

    if actual_weighted_k > 0:
        total_count = df_filtered['post_count'].sum()
        df_filtered['probability'] = df_filtered['post_count'] / total_count

        sampled_indices_weighted = np.random.choice(
            df_filtered.index, 
            size=actual_weighted_k, 
            replace=False, 
            p=df_filtered['probability']
        )

    if actual_random_k > 0:
        df_remaining = df_filtered.drop(index=sampled_indices_weighted)
        sampled_indices_random = np.random.choice(
            df_remaining.index, 
            size=actual_random_k, 
            replace=False
        )

    final_indices = list(sampled_indices_weighted) + list(sampled_indices_random)
    
    result = df_filtered.loc[final_indices, ['cn_name']].copy()
    return result


def escape_parentheses(data):
    if isinstance(data, str):
        return data.replace('(', '\\(').replace(')', '\\)').replace('_', ' ')
    elif isinstance(data, list):
        return [escape_parentheses(item) for item in data]
    elif isinstance(data, dict):
        return {escape_parentheses(key): escape_parentheses(value) for key, value in data.items()}
    elif isinstance(data, tuple):
        return tuple(escape_parentheses(item) for item in data)
    else:
        return data


def process_tags(tags_dict: Dict[str, float], escape_parentheses: bool = False) -> Dict[str, float]:
    processed = {}
    for tag, score in tags_dict.items():
        new_tag = tag
        if escape_parentheses:
            new_tag = new_tag.replace("(", r"\(").replace(")", r"\)")
        new_tag = new_tag.replace("_", " ")
        processed[new_tag] = round(score, 2)
    return processed


def _get_img_tags(img: bytes) -> Dict[str, Dict[str, float]]:
    general_tags, character_tags = get_pixai_tags(
        io.BytesIO(img),
        model_name="v0.9",
        thresholds={"general": 0.3, "character": 0.5},
    )
    character_tags = dict(
        sorted(character_tags.items(), key=lambda kv: kv[1], reverse=True)[:1]
    )
    return {
        "general": process_tags(general_tags),
        "character": process_tags(character_tags, escape_parentheses=True),
    }


def _filter_img_tags_for_llm(img_tags: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, List[str]]]:
    """把识别结果中的置信度移除，转为纯标签列表，避免 LLM 按分数挑选。

    分数只用于代码侧的预过滤与排序，不传给扩写/组装 LLM；
    标签按置信度降序排列，画师标签保留 @ 前缀。"""
    filtered: Dict[str, Dict[str, List[str]]] = {}
    for image, groups in img_tags.items():
        entry: Dict[str, List[str]] = {}
        for group in ("general", "character", "artist"):
            raw = groups.get(group) or {}
            if isinstance(raw, dict):
                tags = [t for t, _ in sorted(raw.items(), key=lambda kv: kv[1], reverse=True)]
            elif isinstance(raw, (list, tuple)):
                tags = [str(t).strip() for t in raw if str(t).strip()]
            else:
                tags = []
            if tags:
                entry[group] = tags
        if entry:
            filtered[image] = entry
    return filtered


def _split_tags_by_language(tags: str) -> tuple[str, List[str]]:
    zh_tags = []
    en_tags = []
    for tag in re.split(r"[,，、]+", tags):
        tag = tag.strip()
        if not tag:
            continue
        if tag[:1] == '#':
            zh_tags.append(tag[1:])
        elif re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", tag):
            zh_tags.append(tag)
        else:
            en_tags.append(tag)
    return ", ".join(zh_tags), ", ".join(en_tags)


def _split_layers(prompt_content: str) -> tuple[str, str]:
    """按空行切分硬锚点层与空间叙事层。空间叙事层可能被模型折成多行，
    因此不能用 rsplit('\\n', 1)——那样会把叙事层的前几句并进标签里。"""
    content = (prompt_content or "").strip()
    content = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", content).strip()
    if not content:
        return "", "none"

    blocks = [b.strip() for b in re.split(r"\n\s*\n", content) if b.strip()]
    if len(blocks) >= 2:
        tags_block, nl_block = blocks[0], " ".join(blocks[1:])
    else:
        lines = [l.strip() for l in blocks[0].split("\n") if l.strip()]
        if len(lines) >= 2:
            tags_block, nl_block = lines[0], " ".join(lines[1:])
        else:
            return blocks[0], "none"

    nl_block = " ".join(nl_block.split())
    return tags_block, nl_block or "none"


def _normalize_artist_tags(raw_artist) -> dict[str, float]:
    """Normalize recognition output to ``{"@artist": confidence}``."""
    if isinstance(raw_artist, dict):
        entries = raw_artist.items()
    elif isinstance(raw_artist, (list, tuple)):
        if (
            len(raw_artist) == 2
            and not isinstance(raw_artist[0], (list, tuple, dict))
        ):
            entries = [raw_artist]
        else:
            entries = raw_artist
    else:
        entries = []

    normalized = {}
    for entry in entries:
        if isinstance(entry, dict):
            name = entry.get("artist") or entry.get("name")
            score = entry.get("score", 1.0)
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            name, score = entry[0], entry[1]
        else:
            name, score = entry, 1.0
        name = str(name or "").strip()
        if not name:
            continue
        name = f"@{name.lstrip('@')}"
        try:
            score = round(float(score), 2)
        except (TypeError, ValueError):
            score = 1.0
        normalized[name] = score
    return normalized


def _run_img_tag_batch(images: list) -> list:
    """Run the single-image tag recognizer for every image in one worker."""
    results = []
    for image in images:
        try:
            results.append(_get_img_tags(image))
        except Exception as exc:
            results.append(exc)
    return results


def _run_artist_batch(images: list) -> list:
    """Run the single-image artist recognizer for every image in one worker."""
    results = []
    for image in images:
        try:
            results.append(artist_recognition(image, top_k=1))
        except Exception as exc:
            results.append(exc)
    return results


def _build_recognized_image(
    image_index: int,
    tag_result,
    artist_result,
) -> tuple[str, dict]:
    if isinstance(tag_result, Exception):
        print("图像%d标签识别失败: %s", image_index, tag_result)
        tag_result = {}
    if isinstance(artist_result, Exception):
        print("图像%d画师识别失败: %s", image_index, artist_result)
        artist_result = {}

    if not isinstance(tag_result, dict):
        tag_result = {}
    return f"图像{image_index}", {
        "general": tag_result.get("general", {}),
        "character": tag_result.get("character", {}),
        "artist": _normalize_artist_tags(artist_result),
    }


async def _recognize_images(images: list[bytes]) -> dict[str, dict]:
    if not images:
        return {}

    tag_results, artist_results = await asyncio.gather(
        asyncio.to_thread(_run_img_tag_batch, images),
        asyncio.to_thread(_run_artist_batch, images),
        return_exceptions=True,
    )

    if isinstance(tag_results, Exception):
        print("图像标签批处理失败: %s", tag_results)
        tag_results = [tag_results] * len(images)
    if isinstance(artist_results, Exception):
        print("图像画师批处理失败: %s", artist_results)
        artist_results = [artist_results] * len(images)

    recognized = []
    for index in range(len(images)):
        tag_result = tag_results[index] if index < len(tag_results) else {}
        artist_result = artist_results[index] if index < len(artist_results) else {}
        recognized.append(
            _build_recognized_image(index + 1, tag_result, artist_result)
        )
    return dict(recognized)


def _parse_tag_categories(classify_tags: dict) -> dict[str, float]:
    """解析标签分类 LLM 的 JSON 输出，转换为 标签 -> 类别权重 的映射。

    支持两种形态：{"tags": [{"tag": ..., "category": ...}, ...]} 或直接
    {"tag": "category", ...}。未识别类别按中性权重 1.0 处理，避免
    分类遗漏导致画师关联强度被错误压低。"""

    entries: list = []
    if isinstance(classify_tags.get("tags"), list):
        entries = classify_tags["tags"]
    elif isinstance(classify_tags.get("results"), list):
        entries = classify_tags["results"]

    weights: dict[str, float] = {}
    if entries:
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            tag = str(entry.get("tag") or "").strip()
            category = str(entry.get("category") or "").strip().lower()
            if tag:
                weights[tag.lower()] = TAG_CATEGORY_WEIGHT.get(category, DEFAULT_TAG_CATEGORY_WEIGHT)
    else:
        for tag, category in classify_tags.items():
            if not isinstance(tag, str):
                continue
            category = str(category or "").strip().lower()
            weights[tag.strip().lower()] = TAG_CATEGORY_WEIGHT.get(category, DEFAULT_TAG_CATEGORY_WEIGHT)
    return weights