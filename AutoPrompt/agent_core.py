from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from openai import AsyncOpenAI
from typing import List, Dict, Any

from .agent_prompts import (_ANIMA_OUTPUT_FORMAT, _JAILBREAKER, _ANIMA_ASSEMBLY_DIRECTIVE, _LABEL_SYSTEM_PROMPT, _CLASSIFICATION_SYSTEM_PROMPT, _CHARACTER_SELECTION_SYSTEM_PROMPT,
                           _CHOOSE_ARTIST_SYSTEM_PROMPT, _EXPAND_TAGS_SYSTEM_PROMPT, _DRAWING_REQUEST_PARSER_PROMPT)
from .tools import execute_search_tags, execute_get_related_tags, execute_get_artist_recommendations
from .utils import sample_tags

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def _load_config() -> dict:
    cfg_path = "config.json"
    if not os.path.exists(cfg_path):
        return {}
    with open(cfg_path, encoding="utf-8") as f:
        raw = f.read().strip()
        return json.loads(raw) if raw else {}

cfg = _load_config()
client_cheap = AsyncOpenAI(
    api_key=cfg["cheap"]["api_key"],
    base_url=cfg["cheap"]["base_url"],
)
client_quality = AsyncOpenAI(
    api_key=cfg["quality"]["api_key"],
    base_url=cfg["quality"]["base_url"],
)


BATCH_SIZE = 4
SEARCH_RESULT_LIMIT = 40
RELATED_TAGS_LIMIT = 50
CHARACTER_RELATED_TAGS_LIMIT = 30
SAMPLE_TOP_K = 100
SAMPLE_MAX_THRESHOLD = 1000000
ARTIST_RECOMMEND_LIMIT = 30


async def search(zh_tags: str, user_description: str) -> List[Any]:
    resp = await client_cheap.chat.completions.create(
        model=cfg["cheap"]["model"],
        messages=[
            {"role": "system", "content": _CLASSIFICATION_SYSTEM_PROMPT},
            {"role": "user", "content": f"请对以下 tags 进行分类和合并：{zh_tags}\n\n请以 JSON 格式输出，格式为：{{\"results\": [{{\"query\": \"...\", \"category\": \"...\"}}]}}"}
        ],
        response_format={"type": "json_object"},
        extra_body={"thinking": {"type": "disabled"}},
        temperature=0.1,
    )
    
    parsed_response = json.loads(resp.choices[0].message.content)
    raw_search_queries = parsed_response["results"]

    search_queries = []
    for q in raw_search_queries:
        query_str = q.get("query", "").strip()
        category = q.get("category", "general")
        all_tags = [t.strip() for t in re.split(r'[,，、]+', query_str) if t.strip()]
        if len(all_tags) > BATCH_SIZE:
            for i in range(0, len(all_tags) // BATCH_SIZE * BATCH_SIZE, BATCH_SIZE):
                batch = all_tags[i:i + BATCH_SIZE]
                search_queries.append({
                    "query": ", ".join(batch),
                    "category": category
                })
            search_queries[-1]["query"] += ", " + ", ".join(all_tags[i+BATCH_SIZE:])
        else:
            if query_str:
                search_queries.append({
                    "query": query_str,
                    "category": category
                })

    logger.info("执行标签搜索:\n%s", json.dumps(search_queries, indent=2, ensure_ascii=False))

    tasks = [
        execute_search_tags(
            query=q["query"],
            search_mode="concept_explore",
            category=q["category"],
        )
        for q in search_queries
    ]
    search_results_raw = await asyncio.gather(*tasks)
    raw_search_results = [{"search_tags": q["query"], "results": json.loads(result)['results'][:SEARCH_RESULT_LIMIT]} for q, result in zip(search_queries, search_results_raw)]

    character_candidates: List[Dict[str, Any]] = []
    for query_item, raw_res in zip(search_queries, raw_search_results):
        if query_item["category"] == "character":
            raw_res = raw_res["results"]
            parsed_res = json.loads(raw_res) if isinstance(raw_res, str) else raw_res
            if parsed_res and "results" in parsed_res:
                character_candidates.extend(parsed_res["results"])

    if not character_candidates:
        # logger.info("搜索结果:\n%s", json.dumps(raw_search_results, indent=2, ensure_ascii=False))
        return raw_search_results

    selection_resp = await client_cheap.chat.completions.create(
        model=cfg["cheap"]["model"],
        messages=[
            {"role": "system", "content": _CHARACTER_SELECTION_SYSTEM_PROMPT},
            {
                "role": "user", 
                "content": (
                    f"【用户的原始描述】：\n{user_description}\n\n"
                    f"【候选角色列表】：\n{json.dumps(character_candidates, ensure_ascii=False, indent=2)}"
                )
            }
        ],
        response_format={"type": "json_object"},
        extra_body={"thinking": {"type": "disabled"}},
        temperature=0.1,
    )

    parsed_selection = json.loads(selection_resp.choices[0].message.content)
    selected_tags: List[str] = parsed_selection.get("selected_tags", [])
    selected_characters = []
    for character_candidate in character_candidates:
        if character_candidate["tag"] in selected_tags:
            selected_characters.append(character_candidate["cn_name"])
    logger.info("消歧后选中的角色: %s", selected_characters)

    resolved_character_results: List[str] = []
    
    if selected_tags:
        tasks = [
            execute_get_related_tags(
                tags=[selected_tag],
                limit=RELATED_TAGS_LIMIT
            )
            for selected_tag in selected_tags
        ]
        results = []
        for result in await asyncio.gather(*tasks):
            result = json.loads(result)["results"]
            new_result = []
            for entry in result:
                if '(' not in entry["tag"] and ')' not in entry["tag"]:
                    new_result.append(entry)
                if len(new_result) == CHARACTER_RELATED_TAGS_LIMIT:
                    break
            results.append(new_result)
        resolved_character_results = [{"character":selected_character, "tag":selected_tag, "related_tags":result} for selected_character, selected_tag, result in zip(selected_characters, selected_tags, results)]

    modified_search_results: List[Any] = []

    orig_characters = []
    for query_item, raw_res in zip(search_queries, raw_search_results):
        if query_item["category"] == "character":
            orig_characters.append(query_item["query"])
        else:
            modified_search_results.append(raw_res)

    modified_search_results.insert(0, {"search_tags":",".join(orig_characters), "results":resolved_character_results})

    # logger.info("搜索结果:\n%s", json.dumps(modified_search_results, indent=2, ensure_ascii=False))
    return modified_search_results

async def expand_zh_tags(user_description: str) -> list[str]:
    df_sampled = sample_tags(top_k=SAMPLE_TOP_K, max_threshold=SAMPLE_MAX_THRESHOLD)
    candidates = df_sampled.to_dict(orient="records")
    candidates = [candidate["cn_name"].split(',')[0] for candidate in candidates]
    logger.info("候选采样标签: %s", candidates)

    expand_context = (
        f"【用户原始描述 / User Description】:\n{user_description}\n\n"
        f"【候选采样标签 / Sampled Candidates】:\n{json.dumps(candidates, ensure_ascii=False, indent=2)}"
    )

    logger.info("正在尝试补充标签...")

    resp = await client_cheap.chat.completions.create(
        model=cfg["cheap"]["model"],
        messages=[
            {"role": "system", "content": _JAILBREAKER+"\n\n"+_EXPAND_TAGS_SYSTEM_PROMPT},
            {"role": "user", "content": expand_context},
        ],
        reasoning_effort="low",
        extra_body={"thinking": {"type": "enabled"}},
        temperature=1.0,
        top_p=0.9,
    )

    llm_output = resp.choices[0].message.content
    logger.info("LLM 标签补充分析响应内容:\n%s", llm_output)

    selected_tags_part = ""
    parts = llm_output.split("### 2. 候选筛选与画面补充")
    if len(parts) > 1:
        selected_tags_part = parts[1].strip()

    new_tags = re.sub(r"[`\s]", "", selected_tags_part)

    if new_tags and new_tags.lower() != "none":
        return new_tags

    return ""

async def agent(user_description: str) -> tuple[str, str, str]:
    resp = await client_cheap.chat.completions.create(
        model=cfg["cheap"]["model"],
        messages=[
            {"role": "system", "content": _LABEL_SYSTEM_PROMPT},
            {"role": "user", "content": user_description},
        ],
        extra_body={"thinking": {"type": "disabled"}},
        temperature=0.1,
    )
    resp_content = resp.choices[0].message.content
    
    tags = [tag.strip() for tag in resp_content.split(',')]
    zh_pattern = re.compile(r'[\u4e00-\u9fff]')
    zh_tags = ""
    en_tags = ""
    for tag in tags:
        if zh_pattern.search(tag):
            zh_tags += tag + ","
        else:
            en_tags += tag + ","
    
    logger.info("zh tags: %s | en tags: %s", zh_tags, en_tags)

    add_tags = await expand_zh_tags(user_description)
    if add_tags:
        user_description += "," + add_tags
        zh_tags += add_tags
    
    search_results = await search(zh_tags, user_description)

    user_context = (
        f"【输出格式要求 / Output Format】:\n{_ANIMA_OUTPUT_FORMAT}\n\n"
        f"【检索与关联标签结果 / Search Results】:\n{json.dumps(search_results, ensure_ascii=False, indent=2)}\n\n"
        f"【用户原始输入 / User Description】:\n{user_description}\n\n"
        f"{_ANIMA_ASSEMBLY_DIRECTIVE}"
    )

    logger.info("正在生成最终提示词...")

    resp = await client_quality.chat.completions.create(
        model=cfg["quality"]["model"],
        messages=[
            {"role": "system", "content": _JAILBREAKER},
            {"role": "user", "content": user_context},
        ],
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}},
        temperature=0.1,
    )
    
    final_output = resp.choices[0].message.content

    prompt_pattern = r"## Prompt\s*```([\s\S]*?)```"
    prompt_match = re.search(prompt_pattern, final_output)
    prompt_content = prompt_match.group(1).strip() if prompt_match else "未找到 Prompt 内容"

    chinese_pattern = r"## 中文解释\s*([\s\S]*?)(?=\n##|$)"
    chinese_match = re.search(chinese_pattern, final_output)
    chinese_content = chinese_match.group(1).strip() if chinese_match else "未找到中文解释内容"

    try:
        tags_prompt, natural_prompt = prompt_content.strip().rsplit('\n', 1)
    except:
        tags_prompt, natural_prompt = prompt_content, "none"

    has_artist_tag = any(tag.strip().startswith('@') for tag in tags_prompt.split(','))

    if not has_artist_tag and tags_prompt:
        logger.info("tags_prompt 中未发现画师标签，开始匹配画师...")
            
        recommendations_json = await execute_get_artist_recommendations(
            tags=tags_prompt.split(','),
            limit=ARTIST_RECOMMEND_LIMIT
        )
        recommendations_data = json.loads(recommendations_json)["results"]

        if len(recommendations_data) > 0:
            choose_context = (
                f"【用户原始描述】:\n{user_description}\n\n"
                f"【候选画师数据】:\n{json.dumps(recommendations_data, ensure_ascii=False, indent=2)}"
            )
            
            choose_resp = await client_cheap.chat.completions.create(
                model=cfg["cheap"]["model"],
                messages=[
                    {"role": "system", "content": _CHOOSE_ARTIST_SYSTEM_PROMPT},
                    {"role": "user", "content": choose_context},
                ],
                extra_body={"thinking": {"type": "disabled"}},
                temperature=1.0,
                top_p=0.9,
            )
            
            selected_artist = choose_resp.choices[0].message.content.strip()
            selected_artist = re.sub(r"^['\"`@\s]+|['\"`\s]+$", "", selected_artist)
            
            if selected_artist and selected_artist.lower() != "none":
                logger.info("匹配到最符合描述的画师: %s", selected_artist)
                tags_prompt_clean = tags_prompt.strip()
                if tags_prompt_clean.endswith(','):
                    tags_prompt = f"{tags_prompt_clean} @{selected_artist}"
                else:
                    tags_prompt = f"{tags_prompt_clean}, @{selected_artist}"
            else:
                logger.info("未筛选到契合的画师。")
        else:
            logger.info("未检索到相关联的画师数据。")
                
    else:
        if has_artist_tag:
            logger.info("tags_prompt 中已存在画师标签，跳过自动推荐。")

    return tags_prompt, natural_prompt, chinese_content

async def extract_prompt_params(text: str):
    messages = [
        {"role": "system", "content": _DRAWING_REQUEST_PARSER_PROMPT},
        {"role": "user", "content": text},
    ]

    response = await client_cheap.chat.completions.create(
        model=cfg["cheap"]["model"],
        messages=messages,
        extra_body={"thinking": {"type": "disabled"}},
        temperature=0.1,
    )

    raw = response.choices[0].message.content

    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if json_match:
        raw = json_match.group()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError(f"LLM 返回非 JSON 格式: {raw[:200]}")

    return data.get("prompt", text), int(data.get("width", 920)), int(data.get("height", 1536))