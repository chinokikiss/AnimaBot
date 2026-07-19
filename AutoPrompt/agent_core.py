from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from openai import AsyncOpenAI
from typing import List, Dict, Any

from .agent_prompts import (_ANIMA_OUTPUT_FORMAT, _ANIMA_ASSEMBLY_DIRECTIVE, _CLASSIFICATION_SYSTEM_PROMPT, _CHARACTER_SELECTION_SYSTEM_PROMPT,
                           _CHOOSE_ARTIST_SYSTEM_PROMPT, _EXPAND_TAGS_SYSTEM_PROMPT, _DRAWING_REQUEST_PARSER_PROMPT)
from .tools import execute_search_tags, execute_get_related_tags, execute_get_artist_recommendations
from .utils import sample_tags, replace_underscores

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


BATCH_SIZE = 5
SEARCH_RESULT_LIMIT = 40
RELATED_TAGS_LIMIT = 50
WEIGHTED_K = 100
RANDOM_K = 25
SAMPLE_MAX_THRESHOLD = 1000000
ARTIST_RECOMMEND_LIMIT = 30


async def search(zh_tags: str, user_description: str) -> List[Any]:
    resp = await client_cheap.chat.completions.create(
        model=cfg["cheap"]["model"],
        messages=[
            {"role": "system", "content": _CLASSIFICATION_SYSTEM_PROMPT},
            {
                "role": "user", 
                "content": (
                    f"原始用户描述（上下文参考）：{user_description}\n"
                    f"待分类的中文 tags：{zh_tags}\n\n"
                    f"请结合原始描述的语义上下文，对上述 tags 进行合理的分类、版权关联与合并。"
                    f"请以 JSON 格式输出，格式为：{{\"results\": [{{\"query\": \"...\", \"category\": \"...\"}}]}}"
                )
            }
        ],
        response_format={"type": "json_object"},
        extra_body={"thinking": {"type": "disabled"}},
        temperature=0.0,
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
    for query_item, parsed_res in zip(search_queries, raw_search_results):
        if query_item["category"] == "character":
            if parsed_res and "results" in parsed_res:
                character_candidates.append({"query":query_item["query"], "candidates":parsed_res["results"]})

    if not character_candidates:
        # logger.info("搜索结果:\n%s", json.dumps(raw_search_results, indent=2, ensure_ascii=False))
        return raw_search_results, []

    selection_resp = await client_cheap.chat.completions.create(
        model=cfg["cheap"]["model"],
        messages=[
            {"role": "system", "content": _CHARACTER_SELECTION_SYSTEM_PROMPT},
            {
                "role": "user", 
                "content": (
                    f"【用户的原始描述】：\n{user_description}\n\n"
                    f"【候选角色列表】：\n{character_candidates}"
                )
            }
        ],
        response_format={"type": "json_object"},
        extra_body={"thinking": {"type": "disabled"}},
        temperature=0.0,
    )

    parsed_selection = json.loads(selection_resp.choices[0].message.content)
    selected_tags: List[str] = parsed_selection.get("selected_tags", [])
    selected_characters = []
    for character_candidate in character_candidates:
        for candidate in character_candidate["candidates"]:
            if candidate["tag"] in selected_tags:
                selected_characters.append(candidate["cn_name"])
    logger.info("消歧后选中的角色: %s", selected_characters)

    resolved_character_results: List[str] = []

    if selected_tags:
        resolved_character_results = [{"character":selected_character, "tag":selected_tag, "related_tags":[]} for selected_character, selected_tag in zip(selected_characters, selected_tags)]
    
    # if selected_tags:
    #     tasks = [
    #         execute_get_related_tags(
    #             tags=[selected_tag],
    #             limit=RELATED_TAGS_LIMIT
    #         )
    #         for selected_tag in selected_tags
    #     ]
    #     results = []
    #     for result in await asyncio.gather(*tasks):
    #         result = json.loads(result)["results"]
    #         new_result = []
    #         for entry in result:
    #             if '(' not in entry["tag"] and ')' not in entry["tag"]:
    #                 new_result.append(entry)
    #         results.append(new_result)
    #     resolved_character_results = [{"character":selected_character, "tag":selected_tag, "related_tags":result} for selected_character, selected_tag, result in zip(selected_characters, selected_tags, results)]

    modified_search_results: List[Any] = []

    orig_characters = []
    for query_item, raw_res in zip(search_queries, raw_search_results):
        if query_item["category"] == "character":
            orig_characters.append(query_item["query"])
        else:
            modified_search_results.append(raw_res)

    modified_search_results.insert(0, {"search_tags":",".join(orig_characters), "results":resolved_character_results})

    modified_search_results = replace_underscores(modified_search_results)

    # logger.info("搜索结果:\n%s", json.dumps(modified_search_results, indent=2, ensure_ascii=False))
    return modified_search_results, selected_characters

async def expand_zh_tags(user_description: str) -> tuple[str, str]:
    df_sampled = sample_tags(weighted_k=WEIGHTED_K, random_k=RANDOM_K, max_threshold=SAMPLE_MAX_THRESHOLD)
    candidates = df_sampled.to_dict(orient="records")
    candidates = [candidate["cn_name"].split(',')[0] for candidate in candidates]
    logger.info("候选采样标签: %s", candidates)

    expand_context = (
        f"【用户原始描述 / User Description】:\n{user_description}\n\n"
        f"【候选采样标签 / Sampled Candidates】:\n{candidates}"
    )

    logger.info("正在尝试补充标签...")

    resp = await client_quality.chat.completions.create(
        model=cfg["quality"]["model"],
        messages=[
            {"role": "system", "content": _EXPAND_TAGS_SYSTEM_PROMPT},
            {"role": "user", "content": expand_context},
        ],
        reasoning_effort="medium",
        extra_body={"thinking": {"type": "enabled"}},
        temperature=1.0,
        top_p=0.9,
    )

    # print("-"*10)
    # print(resp.choices[0].message.reasoning_content)
    # print("-"*10)

    llm_output = resp.choices[0].message.content

    logger.info("LLM 标签补充分析响应内容:\n%s", llm_output)

    new_natural_prompt = ""
    new_tags_prompt = ""

    parts = llm_output.split("### 2. 最终输出")
    if len(parts) > 1:
        raw_tags = parts[1].strip().strip("`").strip()
        new_tags_prompt = re.sub(r"\s*,\s*", ", ", raw_tags)

        sub_parts = parts[0].split("### 1. 自然语言描述")
        if len(sub_parts) > 1:
            raw_natural = sub_parts[1].strip().strip("`").strip()
            new_natural_prompt = re.sub(r"\s+", " ", raw_natural)

    return new_tags_prompt, new_natural_prompt

async def agent(user_description: str) -> tuple[str, str, str]:
    logger.info("用户描述: %s", user_description)

    zh_tags, user_description = await expand_zh_tags(user_description)
            
    search_results, selected_characters = await search(zh_tags, user_description)

    user_context = (
        f"【检索与关联标签结果 / Search Results】:\n{search_results}\n\n"
        f"【用户原始输入 / User Description】:\n{user_description}\n\n"
        f"{_ANIMA_ASSEMBLY_DIRECTIVE}"
    )

    logger.info("正在生成最终提示词...")

    resp = await client_quality.chat.completions.create(
        model=cfg["quality"]["model"],
        messages=[
            {"role": "system", "content": _ANIMA_OUTPUT_FORMAT},
            {"role": "user", "content": user_context},
        ],
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}},
        temperature=0.0,
    )

    # print("-"*10)
    # print(resp.choices[0].message.reasoning_content)
    # print("-"*10)
    
    prompt_content = resp.choices[0].message.content

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
        recommendations_data = json.loads(recommendations_json).get("results", [])

        if len(recommendations_data) > 0:
            choose_context = (
                f"【用户原始描述】:\n{user_description}\n\n"
                f"【候选画师数据】:\n{recommendations_data}"
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

    return tags_prompt, natural_prompt, user_description, selected_characters

async def extract_prompt_params(text: str):
    messages = [
        {"role": "system", "content": _DRAWING_REQUEST_PARSER_PROMPT},
        {"role": "user", "content": text},
    ]

    response = await client_cheap.chat.completions.create(
        model=cfg["cheap"]["model"],
        messages=messages,
        extra_body={"thinking": {"type": "disabled"}},
        temperature=0.0,
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