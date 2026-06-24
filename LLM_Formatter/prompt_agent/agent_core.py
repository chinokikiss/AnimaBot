from __future__ import annotations

import asyncio
import json
import re
import sys
from openai import AsyncOpenAI

from prompt_agent.agent_prompts import (
    get_agent_system_prompt,
    get_format_tool_directive,
    LOW_ASSEMBLY_PROMPT,
    QUERY_REWRITE_PROMPT,
    _ANIMA_SELF_CHECK,
)
from prompt_agent.tools import (
    get_tools,
    REPLACE_PROMPT_TOOL,
    execute_search_tags,
    execute_get_related_tags,
    execute_get_artist_recommendations,
    execute_get_anima_format,
    execute_get_newbie_format,
)
from prompt_agent.cache import (
    get_baseline_store, compute_edit, normalize as normalize_prompt,
)
from prompt_agent import utils

MAX_ROUNDS = 10

_STAGNATION_MIN_NEW = 3
_STAGNATION_LIMIT = 2
_LOW_NOVELTY_RATIO = 0.34
_PROVIDED_TOPK = 3
_REVISION_MAX_ROUNDS = 3


def _sanitize_messages_for_gemini(messages):
    sanitized = []
    for m in messages:
        mc = dict(m)
        if mc.get("role") == "assistant" and mc.get("tool_calls"):
            mc.pop("content", None)
        if (mc.get("role") == "user"
                and isinstance(mc.get("content"), str)
                and sanitized and sanitized[-1].get("role") == "tool"):
            prev = sanitized[-1]
            prev_content = prev.get("content") or ""
            prev["content"] = (prev_content + "\n\n" + mc["content"]) if prev_content else mc["content"]
            continue
        sanitized.append(mc)

    result = []
    i = 0
    n = len(sanitized)
    while i < n:
        m = sanitized[i]
        tool_calls = m.get("tool_calls") if m.get("role") == "assistant" else None
        if tool_calls and len(tool_calls) > 1:
            j = i + 1
            resp_by_id = {}
            while j < n and sanitized[j].get("role") == "tool":
                resp_by_id[sanitized[j].get("tool_call_id")] = sanitized[j]
                j += 1
            for tc in tool_calls:
                single = dict(m)
                single["tool_calls"] = [tc]
                single.pop("content", None)
                result.append(single)
                resp = resp_by_id.get(tc.get("id"))
                if resp is not None:
                    result.append(resp)
                else:
                    result.append({"role": "tool", "tool_call_id": tc.get("id"),
                                   "content": "{}"})
            i = j
        else:
            result.append(m)
            i += 1
    return result


def _dump_request_debug(sanitized_messages, tools):
    import json as _json
    payload = {
        "model": "(see agent_core.py model_name)",
        "messages": sanitized_messages,
        "tools": tools,
    }
    try:
        blob = _json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    except Exception:
        blob = str(payload)
    _log_error("── 请求体 DEBUG DUMP ──")
    print(blob, file=sys.stderr, flush=True)
    _log_error("── DEBUG DUMP END ──")


class _C:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def _log(msg, color=""):
    prefix = f"{_C.BOLD}{_C.BLUE}[Agent]{_C.ENDC}"
    if color:
        print(f"{prefix} {color}{msg}{_C.ENDC}", file=sys.stderr, flush=True)
    else:
        print(f"{prefix} {msg}", file=sys.stderr, flush=True)

def _log_warn(msg):
    _log(f"⚠ {msg}", _C.WARNING)

def _log_error(msg):
    _log(f"✗ {msg}", _C.FAIL)

def _log_ok(msg):
    _log(f"✓ {msg}", _C.GREEN)

def _log_section(title):
    _log(f"── {title} " + "─" * max(0, 50 - len(title)))

def _log_round_header(round_num):
    _log(f"── Round {round_num} " + "─" * max(0, 50 - len(str(round_num)) - 7))

def _log_banner(msg):
    _log("═" * 55)
    _log(msg)
    _log("═" * 55)


def _repair_xml(xml_string):
    result = utils.repair_xml(xml_string)
    return result


def _clean_prompt(xml_content, gemma_prompt):
    result = utils.clean_prompt(xml_content, gemma_prompt)
    return result


def _split_by_language(text):
    return utils.split_by_language(text)


_EFFORT_CONFIG = {
    "Low":    {"search_mode": "full_scene", "related_limit": 50},
    "Medium": {"search_mode": "full_scene", "related_limit": 30, "max_rounds": 8},
    "High":   {"search_mode": "full_scene", "related_limit": 50, "max_rounds": 10, "include_wiki": True},
}


def _serialize_tool_calls(tool_calls):
    if not tool_calls:
        return []
    result = []
    for tc in tool_calls:
        result.append({
            "id": tc.id,
            "type": tc.type,
            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
        })
    return result


class PromptAgent:
    def __init__(self, api_key, api_url, model_name, mode, thinking, config, effort="Medium"):
        self.api_key = api_key
        self.api_url = api_url
        self.model_name = model_name
        self.mode = mode
        self.thinking = thinking
        self.config = config
        self.effort = effort
        self._effort_cfg = _EFFORT_CONFIG.get(effort, _EFFORT_CONFIG["Medium"])
        self.llm = AsyncOpenAI(api_key=api_key, base_url=api_url)
        from LLM_Node import get_platform_settings
        self._extra_body = get_platform_settings(self.api_url, self.model_name, False)
        self.total_cached_input = 0
        self.total_uncached_input = 0
        self.total_output = 0

    def _log_token_usage(self, usage):
        if not usage:
            return

        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens

        cached_tokens = 0
        if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
            cached_tokens = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0

        uncached_tokens = max(0, prompt_tokens - cached_tokens)

        self.total_cached_input += cached_tokens
        self.total_uncached_input += uncached_tokens
        self.total_output += completion_tokens

        _log(
            f"本轮 Token 详情: "
            f"输入 {prompt_tokens} (其中缓存命中 {cached_tokens}, 实际计算 {uncached_tokens}) "
            f"+ 输出 {completion_tokens}"
        )

    def _log_financial_summary(self, rounds=0):
        _log_banner(
            f"Agent 完成 | 总轮次: {rounds + 1}\n"
            f"  [账面统计] 累计物理交互 Token: {self.total_uncached_input + self.total_cached_input + self.total_output}\n"
            f"  [真实消耗] 缓存命中输入: {self.total_cached_input}\n"
            f"            实际计算输入: {self.total_uncached_input}\n"
            f"            实际生成输出: {self.total_output}\n"
        )

    async def _rewrite_query(self, question):
        _log_section("查询重写")
        prompt = QUERY_REWRITE_PROMPT.format(question=question)

        extra_body_list = [self._extra_body]
        if self._extra_body.get("reasoning"):
            extra_body_list.append({k: v for k, v in self._extra_body.items() if k != "reasoning"})

        for attempt, extra_body in enumerate(extra_body_list):
            raw = None
            try:
                resp = await self.llm.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=2048,
                    extra_body=extra_body,
                )
                raw = resp.choices[0].message.content
                if not raw or not raw.strip():
                    if attempt == 0 and len(extra_body_list) > 1:
                        _log_warn("查询重写返回空响应，去掉 reasoning 参数重试...")
                        continue
                    _log_warn("查询重写 LLM 返回空响应，跳过重写")
                    return "", []
                raw = raw.strip().strip("```json").strip("```").strip()
                variants = json.loads(raw)
                if isinstance(variants, list):
                    user_tags = ""
                    dimensions = []
                    for v in variants:
                        v = str(v).strip()
                        if not v:
                            continue
                        if v.startswith("[已有]"):
                            user_tags = v.replace("[已有]", "").strip()
                            _log(f"  [已有] 用户标签: {user_tags[:80]}...")
                        else:
                            dimensions.append(v)
                    _log(f"用户输入拆解为 {len(dimensions)} 个搜索维度 + {'已有标签' if user_tags else '无已有标签'}")
                    for i, q in enumerate(dimensions, 1):
                        _log(f"  {i}. {q}")
                    return user_tags, dimensions
            except Exception as e:
                if attempt == 0 and len(extra_body_list) > 1:
                    _log_warn(f"查询重写失败（{e}），去掉 reasoning 参数重试...")
                    continue
                _log_warn(f"查询重写失败（已跳过）: {e}")
                if raw is not None:
                    _log_warn(f"LLM 响应体: {raw[:500]}")
        return "", []

    async def _execute_tool(self, name, args):
        if name == "search_tags":
            default_mode = self._effort_cfg.get("search_mode", "full_scene")
            default_wiki = self._effort_cfg.get("include_wiki", False)
            return await execute_search_tags(
                query=str(args.get("query", "")),
                search_mode=str(args.get("search_mode", default_mode)),
                category=str(args.get("category", "all")),
                show_nsfw=bool(args.get("show_nsfw", True)),
                include_wiki=bool(args.get("include_wiki", default_wiki)),
            )
        elif name == "get_related_tags":
            args["limit"] = min(int(args.get("limit", 30)), self._effort_cfg["related_limit"])
            tags = args.get("tags", [])
            if isinstance(tags, str):
                try:
                    tags = json.loads(tags)
                except Exception:
                    tags = [t.strip() for t in tags.split(",") if t.strip()]
            return await execute_get_related_tags(
                tags=tags,
                limit=int(args.get("limit", 30)),
                show_nsfw=bool(args.get("show_nsfw", True)),
                include_wiki=bool(args.get("include_wiki", False)),
            )
        elif name == "get_artist_recommendations":
            tags = args.get("tags", [])
            if isinstance(tags, str):
                try:
                    tags = json.loads(tags)
                except Exception:
                    tags = [t.strip() for t in tags.split(",") if t.strip()]
            return await execute_get_artist_recommendations(
                tags=tags,
                limit=int(args.get("limit", 30)),
                min_cooc=int(args.get("min_cooc", 3)),
                show_nsfw=bool(args.get("show_nsfw", True)),
            )
        elif name == "get_anima_format":
            return await execute_get_anima_format()
        elif name == "get_newbie_format":
            return await execute_get_newbie_format()
        elif name == "replace_prompt":
            current = getattr(self, "_selfcheck_content", "")
            old_list = args.get("old_strings", [])
            new_list = args.get("new_strings", [])
            if old_list:
                modified = current
                changes = []
                for old, new in zip(old_list, new_list):
                    if old in modified:
                        modified = modified.replace(old, new)
                        changes.append((old, new))
                self._selfcheck_content = modified
                change_lines = [f"「{o}」→「{n}」" for o, n in changes]
                return json.dumps({
                    "status": "ok", "modified": len(changes) > 0,
                    "note": f"已完成 {len(changes)} 处替换",
                    "changes": change_lines,
                    "new_content": modified,
                }, ensure_ascii=False)
            else:
                new_content = new_list[0] if new_list else ""
                self._selfcheck_content = new_content
                return json.dumps({
                    "status": "ok", "modified": True,
                    "note": "已完全重写提示词内容",
                    "new_content": new_content,
                }, ensure_ascii=False)
        else:
            return json.dumps({"error": f"未知工具: {name}"}, ensure_ascii=False)

    def _log_tool_call(self, name, args):
        if name == "search_tags":
            query_str = args.get("query", "")
            mode = args.get("search_mode", "full_scene")
            _log(f"  > 搜索标签：{query_str}", _C.GREEN)
            _log(f"    [search_tags] mode={mode}, category={args.get('category', 'all')}")
        elif name == "get_related_tags":
            tags = args.get("tags", [])
            if isinstance(tags, str):
                try:
                    tags = json.loads(tags)
                except Exception:
                    tags = [t.strip() for t in tags.split(",") if t.strip()]
            _log(f"  > 关联推荐：{', '.join(tags[:5])}", _C.GREEN)
            _log(f"    [get_related_tags] tags={len(tags)}, limit={args.get('limit', 30)}")
        elif name == "get_artist_recommendations":
            tags = args.get("tags", [])
            if isinstance(tags, str):
                try:
                    tags = json.loads(tags)
                except Exception:
                    tags = [t.strip() for t in tags.split(",") if t.strip()]
            _log(f"  > 画师推荐：{', '.join(tags[:5])}", _C.GREEN)
            _log(f"    [get_artist_recommendations] tags={len(tags)}, limit={args.get('limit', 30)}")
        elif name == "get_anima_format":
            _log(f"  > 获取 Anima 格式规范", _C.GREEN)
        elif name == "get_newbie_format":
            _log(f"  > 获取 NewBie 格式规范", _C.GREEN)
        else:
            _log(f"  > 调用工具：{name}", _C.GREEN)

    def _log_tool_result(self, name, result_str):
        if name in ("get_anima_format", "get_newbie_format"):
            _log(f"    格式规范已获取 ({len(result_str)} chars)", _C.GREEN)
            return
        try:
            data = json.loads(result_str)
            results = data.get("results", [])
            if results:
                _log(f"    找到 {len(results)} 个标签", _C.GREEN)
            elif data.get("error"):
                _log_warn(f"    工具返回错误: {data['error']}")
            else:
                _log("    未找到标签", _C.WARNING)
        except Exception:
            pass

    @staticmethod
    def _extract_tag_list(result_str: str) -> list[str]:
        try:
            data = json.loads(result_str)
        except Exception:
            return []
        prompt = data.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            return [t.strip() for t in prompt.split(",") if t.strip()]
        return [
            (t.get("tag") or "").strip()
            for t in data.get("results", [])
            if (t.get("tag") or "").strip()
        ]

    @staticmethod
    def _extract_tag_names(result_str: str) -> set[str]:
        return set(PromptAgent._extract_tag_list(result_str))

    @staticmethod
    def _collect_cn_from_result(result_str: str) -> dict[str, str]:
        mapping = {}
        try:
            data = json.loads(result_str)
            for t in data.get("results", []):
                tag = (t.get("tag") or "").strip()
                cn = (t.get("cn_name") or "").strip()
                if tag and cn:
                    mapping[tag] = cn
        except Exception:
            pass
        return mapping

    def _get_output_format_section(self):
        from prompt_agent.agent_prompts import _NEWBIE_OUTPUT_FORMAT, _ANIMA_OUTPUT_FORMAT
        if self.mode == "Anima":
            return _ANIMA_OUTPUT_FORMAT
        return _NEWBIE_OUTPUT_FORMAT

    async def _fallback_normal(self, user_text, image):
        _log_warn("回退为普通模式（无工具调用）")
        from prompt_agent.agent_prompts import get_agent_system_prompt
        system_content, fu, fa = get_agent_system_prompt(self.mode, self.config)
        messages = [{"role": "system", "content": system_content}]
        if fu and fa:
            messages.append({"role": "user", "content": fu})
            messages.append({"role": "assistant", "content": fa})
        messages.append({"role": "user", "content": "<user_message>\n" + user_text + "\n</user_message>"})
        try:
            resp = await self.llm.chat.completions.create(
                model=self.model_name, messages=messages,
                temperature=0.7, max_tokens=10240, extra_body=self._extra_body,
            )
            content = resp.choices[0].message.content or ""
        except Exception as e:
            _log_error(f"回退模式 LLM 调用失败: {e}")
            raise
        _log_section("输出解析")
        xml_out, text_out = self._parse_output(content)
        return xml_out, text_out, content

    async def _batch_search_tags(self, dimensions):
        _log_section("批量搜索标签")
        all_tag_names = []
        tag_cn_map: dict[str, str] = {}
        for dim in dimensions:
            _log(f"  > 搜索：{dim}", _C.GREEN)
            result_str = await execute_search_tags(
                query=dim, search_mode="full_scene", show_nsfw=True,
            )
            try:
                data = json.loads(result_str)
                results = data.get("results", [])
                if results:
                    _log(f"    找到 {len(results)} 个标签", _C.GREEN)
                    for t in results:
                        tag = t.get("tag", "")
                        if tag:
                            all_tag_names.append(tag)
                        cn = (t.get("cn_name") or "").strip()
                        if tag and cn:
                            tag_cn_map[tag] = cn
                else:
                    _log("    未找到标签", _C.WARNING)
            except Exception:
                pass
        _log(f"共收集 {len(all_tag_names)} 个标签")
        return all_tag_names, tag_cn_map

    async def _explore_related_tags(self, all_tag_names, user_text):
        _log_section("标签关联探索")
        tools_related = [t for t in await get_tools() if t["function"]["name"] == "get_related_tags"]
        tags_preview = ", ".join(all_tag_names[:60])
        if len(all_tag_names) > 60:
            tags_preview += f", ... (共 {len(all_tag_names)} 个)"

        step3_system = (
            "你是提示词标签专家。以下是搜索到的标签集合。\n"
            "你可以调用 get_related_tags 工具来发现与这些标签共现的补充标签。\n"
            "选择你认为对画面构建最有价值的 5-10 个标签传入工具。调用一次即可。\n\n"
            "标签集合：" + tags_preview
        )
        step3_messages = [
            {"role": "system", "content": step3_system},
            {"role": "user", "content": user_text},
        ]

        try:
            resp = await self.llm.chat.completions.create(
                model=self.model_name, messages=step3_messages,
                tools=tools_related, tool_choice="auto",
                temperature=0.7, max_tokens=500,
                extra_body=self._extra_body,
            )
            msg = resp.choices[0].message
            if resp.usage:
                self._log_token_usage(resp.usage)

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    except json.JSONDecodeError:
                        args = {}
                    if name != "get_related_tags":
                        _log_warn(f"Low effort 不允许调用 {name}，跳过")
                        continue
                    args["limit"] = min(int(args.get("limit", 30)), 50)
                    self._log_tool_call(name, args)
                    result = await execute_get_related_tags(
                        tags=args.get("tags", []),
                        limit=int(args.get("limit", 30)),
                        show_nsfw=bool(args.get("show_nsfw", True)),
                        include_wiki=bool(args.get("include_wiki", False)),
                    )
                    self._log_tool_result(name, result)
                    try:
                        related_data = json.loads(result)
                        for t in related_data.get("results", []):
                            tag = t.get("tag", "")
                            if tag:
                                all_tag_names.append(tag)
                    except Exception:
                        pass
        except Exception as e:
            _log_warn(f"Step 3 LLM 调用失败（已跳过关联探索）: {e}")

        _log(f"最终标签集合: {len(all_tag_names)} 个")
        return all_tag_names

    async def _assemble_low_output(self, all_tag_names, tag_cn_map, user_text, user_tags, image):
        _log_section("组装最终 prompt")
        from prompt_agent.agent_prompts import get_agent_system_prompt
        output_format = LOW_ASSEMBLY_PROMPT.format(
            output_format_section=self._get_output_format_section(),
        )
        _, fewshot_user, fewshot_assistant = get_agent_system_prompt(self.mode, self.config)

        assembly_messages = [{"role": "system", "content": output_format}]
        if fewshot_user and fewshot_assistant:
            assembly_messages.append({"role": "user", "content": fewshot_user})
            assembly_messages.append({"role": "assistant", "content": fewshot_assistant})

        tags_str = ", ".join(all_tag_names)
        user_content = "<user_message>\n" + user_text + "\n</user_message>"
        if user_tags:
            user_content += "\n\n【用户已提供标签（直接信任，禁止检索）】\n" + user_tags
            user_content += "\n以上标签已由用户提供，直接使用，禁止检索这些标签或其变体。"
        user_content += "\n\n【预搜索标签集合】\n" + tags_str

        if image is not None:
            b64 = utils.bytes_to_base64(image)
            assembly_messages.append({"role": "user", "content": [
                {"type": "text", "text": user_content},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
            ]})
        else:
            assembly_messages.append({"role": "user", "content": user_content})

        try:
            resp = await self.llm.chat.completions.create(
                model=self.model_name, messages=assembly_messages,
                temperature=0.7, max_tokens=10240, extra_body=self._extra_body,
            )
            content = resp.choices[0].message.content or ""
            if resp.usage:
                self._log_token_usage(resp.usage)
        except Exception as e:
            _log_error(f"组装阶段 LLM 调用失败: {e}")
            raise

        _log_section("输出解析")
        xml_out, text_out = self._parse_output(content)

        _log_banner("Low effort 完成")
        return xml_out, text_out, content

    async def _run_low_effort(self, user_text, image=None):
        _log_banner("Low effort 流水线模式已启用")
        _log(f"模式: {self.mode} | Effort: Low | MCP: HF (主) / MS (备)")

        user_tags, dimensions = await self._rewrite_query(user_text)
        if not dimensions:
            dimensions = [user_text]
            _log("查询重写未返回结果，使用原始输入")
        provided_list, _ = self._collect_provided_tags(user_text, user_tags)
        user_tags = ", ".join(provided_list)
        if provided_list:
            _log(f"确定性抽取到用户已提供标签 {len(provided_list)} 个，将禁止检索")

        all_tag_names, tag_cn_map = await self._batch_search_tags(dimensions)
        if not all_tag_names:
            _log_warn("所有维度均未搜索到标签，回退为普通模式")
            return await self._fallback_normal(user_text, image)

        all_tag_names = await self._explore_related_tags(all_tag_names, user_text)

        result = await self._assemble_low_output(all_tag_names, tag_cn_map, user_text, user_tags, image)

        return result

    async def _force_final_output(self, messages):
        messages.append({
            "role": "user",
            "content": "请根据已收集到的标签信息直接输出最终 prompt。"
                       "禁止输出任何 XML 工具调用标签（如 <tool_call>、<invoke> 等），"
                       "只输出纯文本的最终 prompt。",
        })
        try:
            resp = await self.llm.chat.completions.create(
                model=self.model_name,
                messages=_sanitize_messages_for_gemini(messages),
                temperature=0.7, max_tokens=10240, extra_body=self._extra_body,
            )
            content = resp.choices[0].message.content or ""
            forced_tokens = resp.usage.total_tokens if resp.usage else 0
            if resp.usage:
                self._log_token_usage(resp.usage)
        except Exception as e:
            _log_error(f"强制输出 LLM 调用失败: {e}")
            raise
        return content, forced_tokens

    async def run(self, user_text, image=None):
        decision, edit, unique_id, baseline = self._decide_baseline(user_text, image)
        user_text = user_text.replace("重写", "")

        if decision == "reuse":
            return self._parse_output(baseline["output"])

        if self.effort == "Low":
            if decision == "continue":
                xml_out, text_out, content = await self._run_low_continuation(user_text, baseline, edit)
            else:
                xml_out, text_out, content = await self._run_low_effort(user_text, image)
            self._store_baseline(unique_id, user_text, content, image,
                                 baseline.get("format_spec") if baseline else None)
            self._log_financial_summary()
            return xml_out, text_out

        _log_banner("Agent 模式已启用，开始处理用户输入...")
        _log(f"模式: {self.mode} | Effort: {self.effort} | MCP: HF (主) / MS (备)")
        if decision == "continue":
            build = self._build_continuation(baseline, edit)
        else:
            build = await self._build_cold_run(user_text, image)
        messages, max_rounds, provided_norm = build

        content, rounds, total_tokens, captured_spec = await self._run_agent_loop(
            messages, max_rounds, provided_norm, user_text,
        )

        _log_section("输出解析")
        xml_out, text_out = self._parse_output(content)

        fmt_spec = captured_spec or (baseline.get("format_spec") if baseline else None)
        self._store_baseline(unique_id, user_text, content, image, fmt_spec)

        self._log_financial_summary(rounds)
        return xml_out, text_out

    def _decide_baseline(self, user_text, image):
        best_ratio = -1
        result = None
        unique_id = "0"
        for unique_id in get_baseline_store()._by_node:
            baseline = get_baseline_store().get(unique_id)
            if not (baseline and image is None
                    and baseline.get("mode") == self.mode
                    and not baseline.get("has_image")
                    and baseline.get("output")):
                continue
            if normalize_prompt(user_text) == baseline.get("norm_input"):
                best_ratio = 1.0
                result = ("reuse", None, unique_id, baseline)
                break
            edit = compute_edit(baseline["raw_input"], user_text)
            if edit["blocks"] == 0 or edit["continue"]:
                if edit['ratio'] > best_ratio:
                    best_ratio = edit['ratio']
                    if edit["blocks"] == 0:
                        result = ("reuse", None, unique_id, baseline)
                    if edit["continue"]:
                        result = ("continue", edit, unique_id, baseline)
        
        if result is None:
            result = ("cold", None, str(int(unique_id) + 1), None)
        elif "重写" in user_text:
            get_baseline_store().delete(unique_id)
            result = ("cold", None, unique_id, None)

        decision_label = result[0]
        _log(f"基线决策: {decision_label} (ratio={best_ratio:.3f}, unique_id={result[2]})")
        return result

    def _collect_provided_tags(self, user_text, rewrite_user_tags):
        provided_list = utils.extract_provided_tags(user_text)
        provided_norm = {utils.normalize_tag(t) for t in provided_list}
        if rewrite_user_tags:
            for t in rewrite_user_tags.split(","):
                t = t.strip()
                tn = utils.normalize_tag(t)
                if t and tn not in provided_norm:
                    provided_list.append(t)
                    provided_norm.add(tn)
        return provided_list, provided_norm

    async def _run_low_continuation(self, user_text, baseline, edit):
        _log_banner("Low 增量修订：在上一轮结果基础上修订")
        _log(f"改动：{edit['instruction']}")

        candidate_tags = []
        for term in edit.get("new_terms", []):
            result = await execute_search_tags(query=term, search_mode="full_scene", show_nsfw=True)
            names = self._extract_tag_list(result)
            if names:
                candidate_tags.extend(names)
                _log(f"  > 搜索变更词：{term} → {len(names)} 个候选", _C.GREEN)
            else:
                _log(f"  > 搜索变更词：{term} → 未找到", _C.WARNING)
        seen = set()
        candidate_tags = [t for t in candidate_tags if not (t in seen or seen.add(t))]

        if self.mode == "Anima":
            fmt_hint = "必须保留 `## Prompt` 和 `## 中文解释` 两个标题；`## 中文解释` 写完整设计说明。"
        else:
            fmt_hint = "保留同样的 `<img>` XML 代码块及其后的中文翻译。"
        revise_directive = "用户在上一轮提示词的基础上做了如下修改：\n" + edit["instruction"]
        if candidate_tags:
            revise_directive += "\n\n为本次改动检索到的候选标签（按需选用）：\n" + ", ".join(candidate_tags)
        revise_directive += (
            "\n\n请在上一轮输出的基础上进行**最小化修订**：只改动与本次修改直接相关的标签，"
            "其余标签与上一轮输出逐字保持一致。直接输出修订后的完整结果。"
            + fmt_hint
            + "禁止新增任何额外标题或说明段（如「改动说明」），禁止输出关于你做了哪些改动的解释。"
        )

        output_format = LOW_ASSEMBLY_PROMPT.format(
            output_format_section=self._get_output_format_section(),
        )
        messages = [
            {"role": "system", "content": output_format},
            {"role": "user", "content": "<user_message>\n" + baseline["raw_input"] + "\n</user_message>"},
            {"role": "assistant", "content": baseline["output"]},
            {"role": "user", "content": revise_directive},
        ]
        try:
            resp = await self.llm.chat.completions.create(
                model=self.model_name, messages=messages,
                temperature=0.7, max_tokens=10240, extra_body=self._extra_body,
            )
            content = resp.choices[0].message.content or ""
            if resp.usage:
                self._log_token_usage(resp.usage)
        except Exception as e:
            _log_error(f"Low 修订 LLM 调用失败: {e}")
            raise

        _log_section("输出解析")
        xml_out, text_out = self._parse_output(content)
        _log_banner("Low 增量修订完成")
        return xml_out, text_out, content

    async def _build_cold_run(self, user_text, image):
        max_rounds = self._effort_cfg["max_rounds"]

        rewrite_queries = []
        user_tags = ""
        if self._effort_cfg.get("rewrite", True) and len(user_text) > 10:
            user_tags, rewrite_queries = await self._rewrite_query(user_text)

        system_content, fewshot_user, fewshot_assistant = get_agent_system_prompt(
            self.mode, self.config, max_rounds=max_rounds,
        )
        messages = [{"role": "system", "content": system_content}]

        if fewshot_user and fewshot_assistant:
            messages.append({"role": "user", "content": fewshot_user})
            messages.append({"role": "assistant", "content": fewshot_assistant})
            _log("已注入 few-shot 示例")

        user_content = "<user_message>\n" + user_text + "\n</user_message>"

        provided_list, provided_norm = self._collect_provided_tags(user_text, user_tags)
        provided_str = ", ".join(provided_list)
        if provided_list:
            _log(f"确定性抽取到用户已提供标签 {len(provided_list)} 个，将禁止重复检索")

        if provided_str:
            user_content += "\n\n【用户已提供标签（直接信任，禁止检索）】\n" + provided_str
            if not rewrite_queries:
                user_content += (
                    "\n\n上述标签已覆盖全部要素，无需调用任何工具。"
                    "直接将上述标签标准化（空格→下划线、括号转义等）后按格式要求输出即可。"
                )
            else:
                user_content += (
                    "\n\n上述标签已覆盖部分维度（如人设、角色、服装等），直接信任、禁止检索；"
                    "你**只需要**检索下方【待搜索维度】中的内容。"
                )
            _log(f"已注入禁止检索的已提供标签: {len(provided_list)} 个")
        if rewrite_queries:
            user_content += "\n\n【待搜索维度（仅检索以下内容，禁止检索已覆盖概念）】\n" + "\n".join("- " + q for q in rewrite_queries)

        user_content += get_format_tool_directive(self.mode)

        if image is not None:
            b64 = utils.bytes_to_base64(image)
            messages.append({"role": "user", "content": [
                {"type": "text", "text": user_content},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
            ]})
            _log("已附加图片输入（多模态模式）")
        else:
            messages.append({"role": "user", "content": user_content})

        return messages, max_rounds, provided_norm

    def _build_continuation(self, baseline, edit):
        _log_banner("增量修订模式：在上一轮结果基础上续写")
        _log(f"改动：{edit['instruction']}")
        max_rounds = _REVISION_MAX_ROUNDS
        system_content, _, _ = get_agent_system_prompt(
            self.mode, self.config, max_rounds=max_rounds,
        )
        spec = baseline.get("format_spec")
        if spec:
            system_content += (
                "\n\n# 输出格式规范（权威参考，标题与整体结构必须严格遵守）\n\n" + spec
            )
        if self.mode == "Anima":
            fmt_hint = (
                "必须保留 `## Prompt` 和 `## 中文解释` 两个标题；"
                "`## 中文解释` 段照常写完整的中文设计说明（针对最终成品，不是改动清单）。"
            )
        else:
            fmt_hint = "保留同样的 `<img>` XML 代码块及其后的中文翻译。"
        revise_directive = (
            "用户在上一轮提示词的基础上做了如下修改：\n"
            + edit["instruction"]
            + "\n\n请在上一轮输出的基础上进行**最小化修订**："
            "只改动与本次修改直接相关的标签，其余标签与上一轮输出逐字保持一致。"
            "新出现的维度可调用工具检索；未改动、已确定的维度禁止改动、禁止重新检索。\n\n"
            "**输出要求（严格）**：直接输出修订后的**完整结果**，结构与标题必须与上一轮输出逐字一致。"
            + fmt_hint
            + "禁止新增任何额外标题或说明段（例如「改动说明」「修改说明」），"
            "禁止输出任何关于你做了哪些改动的解释。"
        )
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": "<user_message>\n" + baseline["raw_input"] + "\n</user_message>"},
            {"role": "assistant", "content": baseline["output"]},
            {"role": "user", "content": revise_directive},
        ]
        return messages, max_rounds, set()

    def _store_baseline(self, unique_id, user_text, content, image, format_spec=None):
        if not content or not content.strip():
            return
        try:
            get_baseline_store().put(unique_id, {
                "norm_input": normalize_prompt(user_text),
                "raw_input": user_text,
                "output": content,
                "mode": self.mode,
                "has_image": image is not None,
                "format_spec": format_spec,
            })
        except Exception:
            pass

    async def _run_agent_loop(self, messages, max_rounds, provided_norm, user_text=""):
        rounds = 0
        total_tokens = 0
        duplicate_tracker = {}
        tag_cn_map: dict[str, str] = {}
        seen_tags: set[str] = set()
        stagnant_rounds = 0
        stagnated = False
        content = ""
        captured_format = None

        while rounds < max_rounds:
            _log_round_header(rounds + 1)
            _tools = await get_tools()
            _log(f"LLM 请求: {len(_tools)} tools available, {len(messages)} messages")

            _messages = _sanitize_messages_for_gemini(messages)
            try:
                resp = await self.llm.chat.completions.create(
                    model=self.model_name, messages=_messages, tools=_tools,
                    tool_choice="auto", temperature=0.7, max_tokens=10240,
                    extra_body=self._extra_body,
                )
            except Exception as e:
                _log_error(f"LLM API 调用失败: {e}")
                _dump_request_debug(_messages, _tools)
                raise

            msg = resp.choices[0].message
            content = msg.content or ""
            tool_calls = _serialize_tool_calls(msg.tool_calls)
            finish_reason = resp.choices[0].finish_reason

            if resp.usage:
                total_tokens += resp.usage.total_tokens
                self._log_token_usage(resp.usage)

            if finish_reason == "tool_calls" and tool_calls:
                parsed = []
                skipped = []
                for tc in tool_calls:
                    name = tc["function"]["name"]
                    raw_args = tc["function"]["arguments"]
                    try:
                        args = json.loads(raw_args) if raw_args else {}
                    except json.JSONDecodeError:
                        args = {}
                    if name == "search_tags":
                        qn = utils.normalize_tag(str(args.get("query", "")))
                        if qn and qn in provided_norm:
                            _log_warn(f"搜索查询「{args.get('query', '')}」命中用户已提供标签，跳过执行")
                            skipped.append((tc, json.dumps(
                                {"skipped": "user_provided",
                                 "note": "该标签用户已提供，禁止重复搜索，直接使用用户提供的版本即可。"},
                                ensure_ascii=False)))
                            continue
                    call_key = name + ":" + json.dumps(args, sort_keys=True)
                    count = duplicate_tracker.get(call_key, 0) + 1
                    duplicate_tracker[call_key] = count
                    if count > 3:
                        _log_warn(f"检测到重复调用 {name}（第{count}次），跳过执行")
                        skipped.append((tc, json.dumps({"skipped": "duplicate"}, ensure_ascii=False)))
                        continue
                    parsed.append((tc, name, args))
                if not parsed:
                    messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
                    for tc in tool_calls:
                        messages.append({"role": "tool", "tool_call_id": tc["id"],
                                         "content": json.dumps({"skipped": "duplicate"}, ensure_ascii=False)})
                    _log_error("所有 tool_calls 均为重复调用，强制退出循环")
                    break
                messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
                try:
                    tasks = [self._execute_tool(name, args) for _, name, args in parsed]
                    results = await asyncio.gather(*tasks)
                except Exception as e:
                    _log_error(f"并行工具调用失败: {e}")
                    for tc, _, _ in parsed:
                        messages.append({"role": "tool", "tool_call_id": tc["id"],
                                         "content": json.dumps({"error": str(e)}, ensure_ascii=False)})
                    for tc, skip_content in skipped:
                        messages.append({"role": "tool", "tool_call_id": tc["id"],
                                         "content": skip_content})
                    break
                round_returned: set[str] = set()
                for (tc, name, args), result in zip(parsed, results):
                    self._log_tool_call(name, args)
                    self._log_tool_result(name, result)
                    tag_cn_map.update(self._collect_cn_from_result(result))
                    if name in ("get_anima_format", "get_newbie_format"):
                        captured_format = result
                    if name in ("search_tags", "get_related_tags"):
                        returned_list = self._extract_tag_list(result)
                        returned = set(returned_list)
                        if returned:
                            if provided_norm:
                                top_hit = [
                                    t for t in returned_list[:_PROVIDED_TOPK]
                                    if utils.normalize_tag(t) in provided_norm
                                ]
                                if top_hit:
                                    result = result + (
                                        f"\n\n[系统提示] 本次结果中排名最靠前的标签 "
                                        f"{', '.join(top_hit)} 属于用户已提供标签，"
                                        f"说明你正在搜索用户已覆盖的概念。用户已提供的标签禁止重复检索，"
                                        f"请勿再搜索该概念，转向尚未覆盖的维度或直接输出。"
                                    )
                            new_in_call = returned - seen_tags - round_returned
                            round_returned |= returned
                            if len(new_in_call) / len(returned) < _LOW_NOVELTY_RATIO:
                                result = result + (
                                    f"\n\n[系统提示] 本次返回 {len(returned)} 个标签，"
                                    f"其中仅 {len(new_in_call)} 个为新标签，其余均已在先前轮次出现。"
                                    f"该主题/维度已充分覆盖，请勿换措辞重复搜索同一主题，"
                                    f"转向尚未覆盖的维度，或直接输出最终结果。"
                                )
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
                for tc, skip_content in skipped:
                    messages.append({"role": "tool", "tool_call_id": tc["id"],
                                     "content": skip_content})

                round_new = len(round_returned - seen_tags)
                round_had_search = len(round_returned) > 0
                seen_tags |= round_returned
                if round_had_search and round_new < _STAGNATION_MIN_NEW:
                    stagnant_rounds += 1
                    _log_warn(
                        f"低信息增量轮次（本轮新增 {round_new} 个标签），"
                        f"停滞计数 {stagnant_rounds}/{_STAGNATION_LIMIT}"
                    )
                else:
                    stagnant_rounds = 0

                rounds += 1

                if stagnant_rounds >= _STAGNATION_LIMIT:
                    _log_warn("连续低信息增量，提前结束探索，进入收尾输出")
                    stagnated = True
                    break

                remaining = max_rounds - rounds
                progress_msg = f"【轮次进度】第 {rounds}/{max_rounds} 轮，剩余 {remaining} 轮。"
                messages.append({"role": "user", "content": progress_msg})
                continue

            _log(f"LLM 输出最终回答 (finish_reason={finish_reason})")
            content, total_tokens = await self._run_self_check(content, total_tokens, user_text)
            break
        else:
            _log_error(f"Agent 循环超过最大轮次 ({max_rounds})，强制输出")
            content, forced_tokens = await self._force_final_output(messages)
            total_tokens += forced_tokens
            content, total_tokens = await self._run_self_check(content, total_tokens, user_text)

        if stagnated:
            content, forced_tokens = await self._force_final_output(messages)
            total_tokens += forced_tokens
            content, total_tokens = await self._run_self_check(content, total_tokens, user_text)

        return content, rounds, total_tokens, captured_format

    async def _run_self_check(self, content, total_tokens, user_text=""):
        """Anima 模式自检：调用 LLM 检查最终提示词是否符合要求。"""
        if self.mode != "Anima" or not content.strip():
            return content, total_tokens

        _log("进入自检阶段：LLM 检查提示词是否符合要求")
        self._selfcheck_content = content
        selfcheck_prompt = (
            f"【最终自检】请对下方输出的提示词进行逐项检查。\n\n"
            f"=== 用户要求 ===\n{user_text}\n\n"
            f"=== 当前输出 ===\n{content}\n\n"
            f"{_ANIMA_SELF_CHECK}\n\n"
            f"如果发现不符合清单要求或用户要求的问题，请调用 replace_prompt 工具修正。\n"
            f"- 要替换多处内容：提供 old_strings 和 new_strings 列表，两列表一一对应\n"
            f"- 要完全重写：仅提供 new_strings（不提供 old_strings），new_strings[0] 作为全新内容\n"
            f"如全部通过，回复「自检通过」即可。\n\n"
            f"注意：仅可使用 replace_prompt 工具，不要调用其他工具。"
        )
        check_messages = [
            {"role": "user", "content": selfcheck_prompt}
        ]
        check_tools = [REPLACE_PROMPT_TOOL]

        try:
            from LLM_Node import get_platform_settings
            extra_body = get_platform_settings(self.api_url, self.model_name, True)

            check_resp = await self.llm.chat.completions.create(
                model=self.model_name, messages=check_messages,
                tools=check_tools, tool_choice="auto",
                temperature=0.3, max_tokens=10240,
                extra_body=extra_body,
            )
            if check_resp.usage:
                total_tokens += check_resp.usage.total_tokens
                self._log_token_usage(check_resp.usage)

            check_msg = check_resp.choices[0].message
            if check_msg.tool_calls:
                _log("自检触发了修改操作")
                for tc in check_msg.tool_calls:
                    if tc.function.name == "replace_prompt":
                        raw_args = tc.function.arguments
                        try:
                            args = json.loads(raw_args) if raw_args else {}
                        except json.JSONDecodeError:
                            args = {}
                        result_str = await self._execute_tool("replace_prompt", args)
                        result = json.loads(result_str)
                        content = result.get("new_content", content)
                        changes = result.get("changes")
                        if changes:
                            for c in changes:
                                _log_ok(f"  {c}")
                        else:
                            _log_ok(f"  完全重写提示词")
            else:
                _log("自检通过，无需修改")
        except Exception as e:
            _log_warn(f"自检阶段异常（不影响最终输出）: {e}")

        return content, total_tokens

    def _parse_output(self, content):
        if self.mode == "Anima":
            return self._parse_anima_output(content)
        return self._parse_newbie_output(content)

    def _parse_anima_output(self, content):
        _log("Anima 模式: 按 Markdown 标题分割输出 (取最后匹配)")
        prompt_matches = list(re.finditer(r'#{2,}\s*Prompt\s*\n(.*?)(?=\n#{2,}|\Z)', content, re.DOTALL))
        prompt_match = prompt_matches[-1] if prompt_matches else None
        explanation_matches = list(re.finditer(r'#{2,}\s*中文解释\s*\n(.*)', content, re.DOTALL))
        explanation_match = explanation_matches[-1] if explanation_matches else None
        expl_text = explanation_match.group(1) if explanation_match else None
        if expl_text is None:
            headings = list(re.finditer(r'(?m)^#{2,}[^\n]*\n', content))
            if len(headings) >= 2:
                expl_text = content[headings[1].end():]
                _log_warn("第二个标题非「中文解释」，已按位置回退提取解释段")

        if prompt_match:
            prompt_content = utils.strip_code_fences(prompt_match.group(1)).strip()
            lines = prompt_content.split('\n')
            tag_str = '\n'.join(lines[:-1]).strip()
            nl_str = lines[-1].strip()
            xml_out = [tag_str, nl_str]
            text_out = utils.strip_code_fences(expl_text) if expl_text is not None else ""
            _log_ok(f"成功按标题分割: Tags={len(tag_str)} chars, NL={len(nl_str)} chars, 解释={len(text_out)} chars")
            if not expl_text:
                _log_warn("未找到解释段标题，仅提取 Prompt 部分")
        else:
            _log_warn("未找到 ## Prompt 标题，回退到按行分离中英文")
            en_part, zh_part = _split_by_language(content)
            en_part = utils.strip_code_fences(en_part)
            if not en_part:
                _log_warn("Anima 模式未检测到英文内容，返回完整响应")
                en_part = content
            xml_out = [en_part, ""]
            text_out = zh_part
        return xml_out, text_out

    def _parse_newbie_output(self, content):
        _log("NewBie 模式: 提取 XML 代码块")
        xml_content, text_content = utils.parse_newbie_content(content)
        if not re.search(r"", content, re.DOTALL):
            if "<img>" in content and "</img>" in content:
                pass
            elif "<img>" in content:
                _log_warn("回复可能被截断")
            else:
                _log_warn("未检测到 <img> 标签")

        gemma_prompt = self.config.get(
            "gemma_prompt",
            "You are an assistant designed to generate high-quality anime images with the highest degree of image-text alignment based on xml format textual prompts. <Prompt Start>\n",
        )
        xml_content = _clean_prompt(xml_content, gemma_prompt)
        return xml_content, text_content
