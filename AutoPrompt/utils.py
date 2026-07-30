import os
import re
import pandas as pd
import numpy as np

CSV_PATH = os.path.join(os.path.dirname(__file__), "tags_enhanced.csv")

_SCORE_RE = re.compile(r"^score_[1-9]$")
_WEIGHT_RE = re.compile(r"^\((?P<body>.+):(?P<weight>\d+(?:\.\d+)?)\)$")


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


_USER_WEIGHT_RE = re.compile(r"\(\s*([^():]+?)\s*:\s*(\d+(?:\.\d+)?)\s*\)")


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

def _artist_candidate_metrics(entry: dict) -> tuple[int, float, float]:
    coverage = len(entry.get("sources") or [])
    try:
        cooc = max(float(entry.get("cooc_count") or 0), 0.0)
        post_count = max(float(entry.get("post_count") or 0), 0.0)
    except (TypeError, ValueError):
        cooc = post_count = 0.0
    return coverage, cooc, post_count


def rank_artist_candidates(results: list[dict], top_n: int = 10) -> list[dict]:
    """画师候选预排序：sources 覆盖率优先，其次按 共现浓度 × log(共现量)。

    覆盖多个输入标签的画师（如同时命中角色与场景）比单标签高共现的画师
    更契合整体需求。次级得分不能只用浓度（cooc/post）——那会让百帖级
    小画师凭 100% 浓度挤掉千帖级的题材大手，而作品量大的画师风格
    被生成模型学习得更充分，故乘 log(cooc) 平衡专精度与证据量。
    排序后只保留 top_n，避免低相关候选进入最终采样池。"""
    def _score(entry: dict) -> tuple:
        coverage, cooc, post_count = _artist_candidate_metrics(entry)
        ratio = cooc / post_count if post_count > 0 else 0.0
        return (coverage, ratio * np.log10(cooc + 1))

    return sorted(results, key=_score, reverse=True)[:top_n]


def sample_artist_candidate(results: list[dict], top_n: int = 10) -> dict | None:
    """从高相关画师候选中按相关性加权随机抽取一个。

    候选先沿用 ``rank_artist_candidates`` 的排序逻辑限制在 top_n 内，
    再用覆盖率、共现浓度和共现量构成采样权重。这样高相关候选更常被选中，
    但不会像固定取第一名一样让结果失去随机性。
    """
    candidates = [
        result for result in results
        if isinstance(result, dict) and str(result.get("artist") or "").strip()
    ]
    candidates = rank_artist_candidates(candidates, top_n=top_n)
    if not candidates:
        return None

    weights = []
    for candidate in candidates:
        coverage, cooc, post_count = _artist_candidate_metrics(candidate)
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
        return data.replace('(', '\\(').replace(')', '\\)')
    elif isinstance(data, list):
        return [escape_parentheses(item) for item in data]
    elif isinstance(data, dict):
        return {escape_parentheses(key): escape_parentheses(value) for key, value in data.items()}
    elif isinstance(data, tuple):
        return tuple(escape_parentheses(item) for item in data)
    else:
        return data
