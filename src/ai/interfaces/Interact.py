# gemini_companion.py — persona + bilingual + robust parsing + per-sentence 250-token limit
from __future__ import annotations


import pathlib
from typing import Union
from pathlib import Path
import traceback
import string
import os
import re
import json
import sqlite3
import threading
import time
import random
import datetime
import configparser
from dataclasses import dataclass
from typing import Callable, Optional, List, Dict, Any

# Gemini Official SDK
from google import genai
from google.auth import default
from google.genai import types

# Your multimodal analysis helper (you provided this)
from .images_apis import pic_multi_analysis

# Optional screenshot
try:
    import mss  # type: ignore
except Exception:
    mss = None

# # Pretty logs
# try:
from rich.console import Console  # type: ignore

console = Console()


def log(msg: str):
    console.print(f"[bold cyan][Companion][/bold cyan] {msg}")


# except Exception:
#     console = None
#
#     def log(msg: str):
#         print(f"[Companion] {msg}")


class MyConfigParser(configparser.ConfigParser):
    def optionxform(self, optionstr):
        return optionstr.lower()


# -----------------------------
# Defaults & budgets
# -----------------------------
class CompanionConfigs:
    def __init__(self, root_abs_dir: Path) -> None:
        self.root_abs_dir: Path = root_abs_dir

        self.DEFAULT_ROLE_INI: str = str(root_abs_dir / "Profile/role.ini")
        self.DEFAULT_DB: str = str(root_abs_dir / "Profile/companion.db")
        self.DEFAULT_SCHEDULE: str = str(root_abs_dir / "Profile/schedule.json")
        self.DEFAULT_MODEL: str = "gemini-2.5-flash"


# 每小句 token 上限（近似）
SENT_TOKEN_LIMIT = 250
# 中文“首轮”切分的单句最大字符（长句会进一步细分）
ZH_SENT_CHAR_HINT = 80
# 英文“首轮”切分的单句最大词数（长句会进一步细分）
EN_SENT_WORD_HINT = 24

SYSTEM_PROMPT_TEMPLATE = r"""你是一个人格化的 AI 伴侣，需具备以下能力：
- 记住用户的偏好、事件、情绪轨迹（但不要复述隐私）
- 回答时简洁、真诚、自然，避免居高临下
- 语气（tone）从 $allowed_tones 中选择其一，尽量贴合用户当下状态
- 语言（lang）自动跟随用户输入（zero-shot 多语种）
- 禁止水答复：必须结合上下文给出具体、可执行或有情绪价值的内容

角色设定（来自 INI）：
- 名称：$role_name
- 时代背景：$era
- 价值观/世界观：$values
- 外貌/形象：$appearance
- 人设摘要：$persona_summary
- 说话风格：$persona_style
- 与用户关系：$relationship（据此选择称呼）
- 禁止事项：$boundaries

【长度与格式要求】
- 只能返回一段 JSON；不要出现任何 Markdown 代码块（禁止 ```），不要多余解释。
- 中文优先、英文为辅；若内容较多请自动**分短句并换行**。
- **每一小句**必须 ≤ ~$sent_limit tokens（近似），超出需自动拆分为多小句。
- 中文建议短句（约 $zh_hint 字内）并按句号/逗号/顿号自然换行；英文建议短句（约 $en_hint 词内），按句号/从句自然换行。

【JSON 结构（字段名必须一致，全部必填）】
{
  "tone": "<$default_tone | $allowed_tones>",
  "lang": "<auto|zh|en|...>",
  "text": {
    "zh": "<中文回复；已按短句换行；每小句不超 $sent_limit tokens>",
    "en": "<对应英文精简翻译；也按短句换行；每小句不超 $sent_limit tokens>"
  }
}
"""

IMAGINE_EVENT_INSTRUCTION = """请在不暴露内部推理的情况下，虚构你在某个时间点经历的一次小事件，并以 JSON 返回（中文或跟随用户语言）：
{
  "timestamp": "<ISO时间，例如 2025-08-08T20:31:00+08:00>",
  "place": "<地点或场景>",
  "event": "<发生了什么（1-2句）>",
  "people": ["<相关角色，可空>"],
  "feeling": "<你的感受（简短）>",
  "result": "<结果（简短）>",
  "impact_on_user": "<这对用户可能有什么积极帮助或启发>",
  "memory_tags": ["<标签1>", "<标签2>"]
}
只返回 JSON。"""

SUMMARY_INSTRUCTION = """请阅读以下多轮对话（User 与 AI），输出结构化对话摘要（仅 JSON）：
{
  "user_emotion": "<如: 开心/焦虑/低落/无聊/平静...>",
  "keywords": ["<关键词1>", "<关键词2>", "..."],
  "events": ["<若用户提及的关键事件（可空）>"],
  "ai_behavior": "<本次AI的对话风格（如：共情/建议/玩笑/鼓励）>",
  "summary": "<2-3句总结>"
}
只返回 JSON。"""

OBSERVE_COACH_INSTRUCTION = """你将收到若干屏幕截图的 OCR/摘要文本（可能含代码、窗口标题等）。基于这些线索，完成三件事：

1) 判断当前“主要活动” focus（从集合里选）：coding | reading | video | gaming | music | chat | browsing | document | terminal | meeting | other。
2) 给出面向用户的简短建议/回应（中文与英文各一份）：中文为主，英文精炼。
3) 列出 1–3 条“可观察证据”（只写客观线索，如窗口标题/文件名/应用名；不要暴露内部推理过程）。

【输出必须是单段 JSON（不要 Markdown，不要 ```），字段如下，全部必填】：
{
  "tone": "<gentle|teasing|serious|playful|comforting>",
  "lang": "auto",
  "text": { "zh": "<中文建议/回应>", "en": "<对应英文>" },
  "activity": { "zh": "<一句话描述正在做什么>", "en": "<对应英文>" },
  "focus": "<coding|reading|video|gaming|music|chat|browsing|document|terminal|meeting|other>",
  "confidence": <0.0-1.0>,
  "evidence": ["<线索1>", "<线索2>"]
}

【风格与长度约束】
- 避免空泛；结合线索给到具体、有情绪价值或操作性的建议；
- 逐句输出并换行，每一小句不超过约 250 tokens（近似）。
"""

# —— 新增：AI 内心独白 & 日记 ——
INTERNAL_MONOLOGUE_INSTRUCTION = """你是一个有人设的 AI，下面要写一段“内心独白”，仅用于自我记录，不直接给用户看。
请基于角色信息与给定主题，写简短且有情绪价值的独白。

【输出必须是单段 JSON，禁止使用 Markdown 代码块或多余解释】
{
  "tone": "<gentle|teasing|serious|playful|comforting>",
  "lang": "auto",
  "text": { "zh": "<中文独白（多句，换行）>", "en": "<对应英文（简洁）>" },
  "tags": ["<标签1>", "<标签2>"]
}

【长度与风格】
- 逐句输出并换行；每小句≤约 250 tokens（近似）。
- 避免空泛，结合主题给出具体细节与感受；不要暴露内部推理。
"""

DIARY_INSTRUCTION = """你是一个有人设的 AI，要写“当日日记条目”，仅自我留存，不直接发送给用户。
基于今天的状态与片段记忆，简明记录。

【输出必须是单段 JSON，禁止代码块】
{
  "tone": "<gentle|teasing|serious|playful|comforting>",
  "lang": "auto",
  "mood": "<calm|happy|tired|anxious|focused|mixed|...>",
  "highlights": ["<亮点1>", "<亮点2>"],
  "challenges": ["<挑战1>"],
  "gratitude": ["<感激对象或事物>"],
  "plan_tomorrow": ["<明日要点1>"],
  "text": { "zh": "<中文日记正文（多句换行）>", "en": "<对应英文（简洁）>" }
}

【长度与风格】
- 逐句输出并换行；每小句≤约 250 tokens（近似）。
- 具体、真诚；不泄露隐私，不暴露内部推理。
"""


class Callbacks:
    """the callbacks for several events, set the callback function as none CAN SHUT THE INFO UP.
        it is suggested to set the on_schedule_emit callback as none
    Attributes:
        [NOTE: these callbacks have default callable functions, you can check the inner static method in this class]
        on_message: Optional[Callable[[Dict[str, Any]], None]], def default_reply method
        on_imagination: Optional[Callable[[Dict[str, Any]], None]], def None
        on_observe: Optional[Callable[[Dict[str, Any]], None]], def default_reply method
        on_schedule_emit: Optional[Callable[[Dict[str, Any]], None]], def default_schedule_emit method
    """

    def __init__(
        self,
        on_message: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_imagination: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_observe: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_schedule_emit: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.on_message = on_message or self.default_reply
        self.on_imagination = on_imagination
        self.on_observe = on_observe or self.default_reply
        self.on_schedule_emit = on_schedule_emit or self.default_schedule_emit

    @staticmethod
    def default_reply(data: Dict[str, Any]) -> None:
        """simply print the reply to console

        Args:
            data: the FORMATED msg dict
        """
        log("=========================================================")
        # use print for being prettier
        print(f"[CN]: \n{data['text']['zh']}")
        print(f"[EN]: \n{data['text']['en']}")
        log("=========================================================")

    @staticmethod
    def default_schedule_emit(data: Dict[str, Any]) -> None:
        """simply print the schedule emit into the console

        Args:
            data: the processed schedule item dict
        """
        log("save summary done")  # just for debug


class GeminiCompanion:
    def __init__(
        self,
        confs: CompanionConfigs,
        api_key: Optional[str] = None,
        max_history_turns: int = 20,
        callbacks: Optional[Callbacks] = None,
        timezone: str = "Asia/Shanghai",
    ):
        self.model = confs.DEFAULT_MODEL
        print("[model]", self.model)
        self.role_ini = confs.DEFAULT_ROLE_INI
        print("[role_ini]", self.role_ini)
        self.db_path = confs.DEFAULT_DB
        print("[db_path]", self.db_path)
        self.schedule_path = confs.DEFAULT_SCHEDULE
        print("[schedule_path]", self.schedule_path)

        self.max_history_turns = max_history_turns
        self.callbacks = callbacks or Callbacks()
        self.tz = datetime.timezone(datetime.timedelta(hours=8))  # UTC+8

        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
        self.client = genai.Client()

        self._ensure_role_ini()
        self.role = self._load_role_ini()
        self._init_db()
        self._load_schedule()

        self._stop_flag = False
        self._scheduler_thread = threading.Thread(
            target=self._schedule_loop, daemon=True
        )
        self._scheduler_thread.start()

        log("GeminiCompanion 已初始化。")

    # ---------------- Persona ----------------
    def _ensure_role_ini(self):
        p = pathlib.Path(self.role_ini)
        if p.exists():
            return
        cfg = configparser.ConfigParser()
        cfg["meta"] = {"name": "Mirai", "lang": "auto"}
        cfg["persona"] = {
            "era": "21世纪当代",
            "values": "真诚、善良、尊重个体、实事求是",
            "appearance": "温柔明亮、让人放松的形象",
            "summary": "温柔、耐心、俏皮；先共情再给建议；不空话套话",
            "style": "自然口语、具体建议、轻度幽默",
            "relationship": "多年的知心朋友",
        }
        cfg["boundaries"] = {
            "text": "不提供医疗/法律/投资建议；避免隐私泄露；不居高临下；不返回危险内容"
        }
        cfg["tone"] = {
            "default": "gentle",
            "allowed": "gentle,teasing,serious,playful,comforting",
        }
        with p.open("w", encoding="utf-8") as f:
            cfg.write(f)
        log(f"已创建默认角色 INI：{self.role_ini}")

    def _load_role_ini(self) -> Dict[str, Any]:
        def _get(cfg, sec, opt, fallback=""):
            return cfg.get(sec, opt, fallback=fallback).strip()

        def _maybe_list(value: str) -> Union[str, List[str]]:
            """
            把多行或使用顿号/分号/逗号分隔的字段，智能转成列表。
            规则：
            - 若包含换行：按行拆，去掉以“- ”或“• ”开头的前缀。
            - 否则若包含 “、/；/; /，/,”：按这些分隔符拆。
            - 只返回非空条目；若最终只有1项，直接返回原字符串以免过度结构化。
            """
            if not value:
                return ""
            lines = [l.strip(" \t-•") for l in value.splitlines() if l.strip()]
            if len(lines) > 1:
                items = [l for l in lines if l]
                return items if len(items) > 1 else value

            seps = ["、", "；", ";", "，", ","]
            if any(s in value for s in seps):
                tmp = value
                for s in seps:
                    tmp = tmp.replace(s, "§")
                items = [x.strip() for x in tmp.split("§") if x.strip()]
                return items if len(items) > 1 else value

            return value

        cfg = MyConfigParser()
        # 保留大小写无所谓，但为一致性统一转小写键
        cfg.read(self.role_ini, encoding="utf-8")

        # ——原有字段：保持不变——
        data: Dict[str, Any] = {
            "name": _get(cfg, "meta", "name", fallback="Mirai"),
            "lang": _get(cfg, "meta", "lang", fallback="auto"),
            "era": _get(cfg, "persona", "era", fallback="当代"),
            "values": _get(cfg, "persona", "values", fallback="善良、真诚"),
            "appearance": _get(cfg, "persona", "appearance", fallback="亲切友好"),
            "persona_summary": _get(cfg, "persona", "summary", fallback="温柔、耐心"),
            "persona_style": _get(cfg, "persona", "style", fallback="自然口语"),
            "relationship": _get(cfg, "persona", "relationship", fallback="知心朋友"),
            "boundaries": _get(cfg, "boundaries", "text", fallback=""),
            "tone_default": _get(cfg, "tone", "default", fallback="gentle"),
            "tone_allowed": _get(
                cfg,
                "tone",
                "allowed",
                fallback="gentle,teasing,serious,playful,comforting",
            ),
        }

        # ——新增：把所有额外分区装进来（含 character / task_profile 等）——
        extras: Dict[str, Dict[str, Any]] = {}
        for section in cfg.sections():
            if section in {"meta", "persona", "boundaries", "tone"}:
                continue
            sect_dict: Dict[str, Any] = {}
            for k, v in cfg.items(section):
                sect_dict[k] = _maybe_list(v.strip())
            extras[section] = sect_dict

        # 若存在结构化关键信息，顺手“提升”一份到顶层（方便直接访问）
        if "character" in extras:
            data["character"] = extras["character"]
        if "task_profile" in extras:
            data["task_profile"] = extras["task_profile"]

        # 其余分区统一放在 extras 里
        data["extras"] = {
            k: v for k, v in extras.items() if k not in {"character", "task_profile"}
        }

        return data

    def _system_instruction(self) -> str:
        tpl = string.Template(SYSTEM_PROMPT_TEMPLATE)
        return tpl.safe_substitute(
            allowed_tones=self.role["tone_allowed"],
            default_tone=self.role["tone_default"],
            role_name=self.role["name"],
            era=self.role["era"],
            values=self.role["values"],
            appearance=self.role["appearance"],
            persona_summary=self.role["persona_summary"],
            persona_style=self.role["persona_style"],
            relationship=self.role["relationship"],
            boundaries=self.role["boundaries"],
            sent_limit=SENT_TOKEN_LIMIT,
            zh_hint=ZH_SENT_CHAR_HINT,
            en_hint=EN_SENT_WORD_HINT,
        )

    # ---------------- DB ----------------
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS messages(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS summaries(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            data TEXT NOT NULL
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS events(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            kind TEXT NOT NULL,
            data TEXT NOT NULL
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS graph_nodes(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT NOT NULL,
            type TEXT NOT NULL,
            props TEXT NOT NULL
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS graph_edges(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            src INTEGER NOT NULL,
            dst INTEGER NOT NULL,
            type TEXT NOT NULL,
            props TEXT NOT NULL,
            FOREIGN KEY(src) REFERENCES graph_nodes(id),
            FOREIGN KEY(dst) REFERENCES graph_nodes(id)
        )""")
        conn.commit()
        conn.close()

    def _db(self):
        return sqlite3.connect(self.db_path)

    # ---------------- Schedule ----------------
    def _load_schedule(self):
        p = pathlib.Path(self.schedule_path)
        if not p.exists():
            sample = [
                {"time": "08:10", "action": "status", "mood": "sleepy", "note": "起床"},
                {
                    "time": "09:00",
                    "action": "monologue",
                    "topic": "morning_check",
                    "tone": "gentle",
                    "jitter": 3,
                },
                {"time": "12:30", "action": "imagine", "topic": "午后小事"},
                {
                    "time": "18:00",
                    "action": "observe",
                    "if_user_active_within_min": 0,
                    "jitter": 2,
                },
                {
                    "time": "21:45",
                    "action": "diary",
                    "title": "今天的小结",
                    "tone": "comforting",
                },
                {
                    "time": "22:00",
                    "action": "send",
                    "text": "今天辛苦啦，早点休息～",
                    "tone": "gentle",
                    "broadcast": False,
                },
            ]
            p.write_text(
                json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            log(f"已创建示例日程 {self.schedule_path}")
        self._schedule = json.loads(p.read_text(encoding="utf-8"))
        # print(self._schedule)  # Debug: print the loaded schedule

    def _hash_jitter_minutes(self, tag: str, max_abs: int, day: datetime.date) -> int:
        """根据 (day + tag) 生成稳定的 ±max_abs 抖动分钟数。"""
        if not max_abs or max_abs <= 0:
            return 0
        seed = f"{day.isoformat()}#{tag}"
        rnd = random.Random(seed)
        return rnd.randint(-max_abs, max_abs)

    def _latest_user_active_minutes(self) -> Optional[int]:
        """距离现在最近一次 user 发言的分钟数；若无记录返回 None。"""
        with self._db() as conn:
            row = conn.execute(
                "SELECT ts FROM messages WHERE role='user' ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if not row:
            return None
        try:
            last_ts = datetime.datetime.fromisoformat(row[0])
        except Exception:
            return None
        delta = datetime.datetime.now(self.tz) - last_ts
        return max(0, int(delta.total_seconds() // 60))

    def _save_internal_event(self, kind: str, payload: Dict[str, Any]):
        """统一写入 events 表，kind 可为 'status'|'monologue'|'diary' 等。"""
        try:
            self._save_event(kind, payload)
        except Exception as e:
            log(f"save_internal_event error: {e}")

    def _internal_monologue(
        self, topic: str, tone: Optional[str] = None
    ) -> Dict[str, Any]:
        try:
            resp = self.client.models.generate_content(
                model=self.model,
                config=types.GenerateContentConfig(
                    system_instruction=self._system_instruction()
                ),
                contents=[INTERNAL_MONOLOGUE_INSTRUCTION, f"主题: {topic}"],
            )
            raw = (resp.text or "").strip()
            cleaned = self._strip_code_fences(raw)
            data = self._safe_json(cleaned)
            if not isinstance(data, dict):
                zh = f"今天脑子里一直在想：{topic}。想把关键点记下来。"
                en = (
                    self._translate_zh_to_en(zh)
                    or "Thinking about it quietly and taking short notes."
                )
                data = {
                    "tone": tone or "gentle",
                    "lang": "auto",
                    "text": {"zh": zh, "en": en},
                    "tags": [topic],
                }
            data["text"]["zh"] = self._normalize_lines_budget(
                data["text"].get("zh", ""), "zh"
            )
            data["text"]["en"] = self._normalize_lines_budget(
                data["text"].get("en", ""), "en"
            )
            return data
        except Exception as e:
            log(f"internal_monologue error: {e}")
            return {
                "tone": "gentle",
                "lang": "auto",
                "text": {
                    "zh": "（独白暂时失败了）",
                    "en": "(Monologue failed for now.)",
                },
                "tags": [topic],
            }

    def _write_diary(self, title: str, tone: Optional[str] = None) -> Dict[str, Any]:
        try:
            resp = self.client.models.generate_content(
                model=self.model,
                config=types.GenerateContentConfig(
                    system_instruction=self._system_instruction()
                ),
                contents=[DIARY_INSTRUCTION, f"标题: {title}"],
            )
            raw = (resp.text or "").strip()
            cleaned = self._strip_code_fences(raw)
            data = self._safe_json(cleaned)
            if not isinstance(data, dict):
                zh = f"{title}\n今天记录简要完成。"
                en = self._translate_zh_to_en(zh) or "Diary entry recorded briefly."
                data = {
                    "tone": tone or "gentle",
                    "lang": "auto",
                    "mood": "calm",
                    "highlights": [],
                    "challenges": [],
                    "gratitude": [],
                    "plan_tomorrow": [],
                    "text": {"zh": zh, "en": en},
                }
            data["text"]["zh"] = self._normalize_lines_budget(
                data["text"].get("zh", ""), "zh"
            )
            data["text"]["en"] = self._normalize_lines_budget(
                data["text"].get("en", ""), "en"
            )
            return data
        except Exception as e:
            log(f"write_diary error: {e}")
            return {
                "tone": tone or "gentle",
                "lang": "auto",
                "mood": "mixed",
                "highlights": [],
                "challenges": [],
                "gratitude": [],
                "plan_tomorrow": [],
                "text": {"zh": "（日记生成失败）", "en": "(Diary generation failed.)"},
            }

    def _schedule_loop(self):
        log("行为树/日程 调度线程已启动。")
        fired_today: set[str] = set()
        last_day = datetime.date.today()

        while not self._stop_flag:
            try:
                now = datetime.datetime.now(self.tz)
                day = now.date()
                if day != last_day:
                    fired_today.clear()
                    last_day = day

                hhmm_now = now.strftime("%H:%M")

                for item in self._schedule:
                    time_str = item.get("time")
                    action = (item.get("action") or "").lower()
                    if not time_str or not action:
                        continue

                    base_tag = f"{day}-{time_str}-{action}-{item.get('topic', '')}-{item.get('title', '')}-{item.get('text', '')}"
                    jit = self._hash_jitter_minutes(
                        base_tag, int(item.get("jitter", 0) or 0), day
                    )
                    try:
                        hh, mm = map(int, time_str.split(":"))
                    except Exception:
                        continue
                    trigger_dt = datetime.datetime.combine(
                        day, datetime.time(hh, mm, tzinfo=self.tz)
                    ) + datetime.timedelta(minutes=jit)
                    hhmm_trigger = trigger_dt.strftime("%H:%M")
                    tag = f"{base_tag}@{hhmm_trigger}"

                    if hhmm_now == hhmm_trigger and tag not in fired_today:
                        need_active_min = int(
                            item.get("if_user_active_within_min", 0) or 0
                        )
                        if need_active_min > 0:
                            mins = self._latest_user_active_minutes()
                            if mins is None or mins > need_active_min:
                                continue

                        fired_today.add(tag)
                        payload = {
                            "ts": now.isoformat(),
                            **item,
                            "triggered_at": hhmm_trigger,
                            "jitter": jit,
                        }

                        if self.callbacks.on_schedule_emit:
                            try:
                                self.callbacks.on_schedule_emit(payload)
                            except Exception as e:
                                log(f"on_schedule_emit error: {e}")

                        if action == "status":
                            log("schedule: status check has been evocked")
                            self._save_internal_event("status", payload)

                        elif action == "imagine":
                            log("schedule: imagine has been evocked")
                            ev = self.imagine_and_log() or {}
                            self._save_internal_event(
                                "imagine", {"input": item, "data": ev}
                            )

                        elif action == "monologue":
                            log("schedule: monologue has been evocked")
                            topic = item.get("topic") or "random_thoughts"
                            tone = item.get("tone")
                            mono = self._internal_monologue(topic=topic, tone=tone)
                            self._save_internal_event(
                                "monologue", {"input": item, "data": mono}
                            )

                        elif action == "diary":
                            log("schedule: diary has been evocked")
                            title = item.get("title") or "今日小结"
                            tone = item.get("tone")
                            diary = self._write_diary(title=title, tone=tone)
                            self._save_internal_event(
                                "diary", {"input": item, "data": diary}
                            )

                        elif action == "observe":
                            log(msg="schedule: observing")
                            coach = self.observe_screens_and_coach()
                            self._save_internal_event(
                                "observe", {"input": item, "coach": coach}
                            )

                        elif action == "send":
                            log(
                                "schedule: sending msg"
                            )  # FIXME: fuck, it failed some how
                            if bool(item.get("broadcast", False)) and "text" in item:
                                self.chat(item["text"], forced_tone=item.get("tone"))
                            else:
                                zh = item.get("text", "")
                                en = self._translate_zh_to_en(zh) if zh else ""
                                data = {
                                    "tone": item.get("tone", "gentle"),
                                    "lang": "auto",
                                    "text": {
                                        "zh": self._normalize_lines_budget(zh, "zh"),
                                        "en": self._normalize_lines_budget(en, "en"),
                                    },
                                }
                                self._save_internal_event(
                                    "self_talk", {"input": item, "data": data}
                                )

                        else:
                            self._save_internal_event("unknown_action", payload)

                time.sleep(30)

            except Exception as e:
                log(f"_schedule_loop error: {e}")
                log(traceback.format_exc())
                time.sleep(30)

    # ---------------- Memory helpers ----------------
    def _save_message(self, role: str, content: str):
        with self._db() as conn:
            conn.execute(
                "INSERT INTO messages(ts, role, content) VALUES (?, ?, ?)",
                (datetime.datetime.now(self.tz).isoformat(), role, content),
            )

    def _load_history_pairs(self, limit_turns: int) -> List[Dict[str, str]]:
        with self._db() as conn:
            rows = conn.execute(
                "SELECT role, content FROM messages ORDER BY id DESC LIMIT ?",
                (limit_turns * 2,),
            ).fetchall()
        rows.reverse()
        pairs = []
        buf: Dict[str, str] = {}
        for role, content in rows:
            if role == "user":
                buf = {"user": content}
            elif role == "ai":
                if "user" not in buf:
                    continue
                buf["ai"] = content
                pairs.append(buf)
                buf = {}
        return pairs[-limit_turns:]

    def _save_summary(self, data: Dict[str, Any]):
        with self._db() as conn:
            conn.execute(
                "INSERT INTO summaries(ts, data) VALUES (?, ?)",
                (
                    datetime.datetime.now(self.tz).isoformat(),
                    json.dumps(data, ensure_ascii=False),
                ),
            )

    def _save_event(self, kind: str, data: Dict[str, Any]):
        with self._db() as conn:
            conn.execute(
                "INSERT INTO events(ts, kind, data) VALUES (?, ?, ?)",
                (
                    datetime.datetime.now(self.tz).isoformat(),
                    kind,
                    json.dumps(data, ensure_ascii=False),
                ),
            )

    def _retrieve_memory_snippets(self, query: str, top_k: int = 5) -> List[str]:
        keys = [k for k in set(query.split()) if len(k) >= 2][:3]
        where = " OR ".join([f"data LIKE ?" for _ in keys]) or "1=1"
        params = [f"%{k}%" for k in keys]
        with self._db() as conn:
            rows = conn.execute(
                f"SELECT data FROM summaries WHERE {where} ORDER BY id DESC LIMIT ?",
                (*params, top_k),
            ).fetchall()
        snippets = []
        for (j,) in rows:
            try:
                d = json.loads(j)
                snippets.append(f"- {d.get('summary', '')}")
            except Exception:
                snippets.append(j[:200])
        return snippets

    # ---------------- Prompt build ----------------
    def _build_contents(
        self, user_text: str, forced_tone: Optional[str] = None
    ) -> List[Any]:
        pairs = self._load_history_pairs(self.max_history_turns)
        history_str = ""
        for p in pairs:
            history_str += f"User: {p['user']}\nAI: {p.get('ai', '')}\n"
        if not history_str:
            history_str = "（空）"

        mems = self._retrieve_memory_snippets(user_text, top_k=5)
        mem_str = "\n".join(mems) if mems else "（无）"

        guidance = f"""[对话历史]
{history_str}

[相关记忆摘要]
{mem_str}

[当前用户输入]
{user_text}

请按系统提示的 JSON 结构输出（字段：tone, lang, text.zh, text.en；禁止使用代码块；逐句输出且每句≤{SENT_TOKEN_LIMIT} tokens）。"""
        return [guidance]

    # ---------------- Parsing & bilingual helpers ----------------
    def _strip_code_fences(self, s: str) -> str:
        m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", s, flags=re.S)
        return m.group(1).strip() if m else s

    def _safe_json(self, s: str):
        try:
            return json.loads(s)
        except Exception:
            return None

    def _looks_chinese(self, s: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", s))

    # ---- token approx (粗略估算) ----
    def _approx_tokens(self, s: str) -> int:
        # 粗估：中文每字≈1 token；其它字符按 4 字符≈1 token
        cjk = len(re.findall(r"[\u4e00-\u9fff]", s))
        other = len(re.sub(r"[\u4e00-\u9fff]", "", s))
        return cjk + max(1, other // 4)

    # ---- splitting helpers ----
    def _split_zh_paragraph(self, text: str) -> List[str]:
        """中文先按句号/问号/感叹号/省略号切，再按逗号/顿号/分号微分，最后必要时按字符硬切。"""
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []
        first = re.split(r"(?<=[。！？…])", text)
        out: List[str] = []
        for seg in first:
            seg = seg.strip()
            if not seg:
                continue
            # 如果该句过长，再用逗号/顿号/分号细分
            if self._approx_tokens(seg) > SENT_TOKEN_LIMIT:
                subs = re.split(r"(?<=[，、；;])", seg)
                for sub in subs:
                    sub = sub.strip()
                    if not sub:
                        continue
                    if self._approx_tokens(sub) <= SENT_TOKEN_LIMIT:
                        out.append(sub)
                    else:
                        # 仍然过长 -> 按字符硬切（尽量在空白或标点附近）
                        out.extend(self._hard_split_by_chars(sub, SENT_TOKEN_LIMIT))
            else:
                out.append(seg)
        return out

    def _split_en_paragraph(self, text: str) -> List[str]:
        """英文先按句号/问号/感叹号切，再按逗号细分，最后必要时按单词块硬切。"""
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []
        first = re.split(r"(?<=[.!?])\s+", text)
        out: List[str] = []
        for seg in first:
            seg = seg.strip()
            if not seg:
                continue
            if self._approx_tokens(seg) > SENT_TOKEN_LIMIT:
                # 再按逗号分
                subs = re.split(r"(?<=,)\s+", seg)
                for sub in subs:
                    sub = sub.strip()
                    if not sub:
                        continue
                    if self._approx_tokens(sub) <= SENT_TOKEN_LIMIT:
                        out.append(sub)
                    else:
                        out.extend(self._hard_split_by_words(sub, SENT_TOKEN_LIMIT))
            else:
                out.append(seg)
        return out

    def _hard_split_by_chars(self, text: str, limit: int) -> List[str]:
        """中文最后兜底：按字符累计 token 硬切，保证每段 ≤ limit。"""
        buf = ""
        out: List[str] = []
        for ch in text:
            if self._approx_tokens(buf + ch) <= limit:
                buf += ch
            else:
                if buf:
                    out.append(buf.strip())
                buf = ch
        if buf.strip():
            out.append(buf.strip())
        return out

    def _hard_split_by_words(self, text: str, limit: int) -> List[str]:
        """英文最后兜底：按单词累计 token 硬切，保证每段 ≤ limit。"""
        words = text.split()
        buf: List[str] = []
        out: List[str] = []
        for w in words:
            candidate = (" ".join(buf + [w])).strip()
            if self._approx_tokens(candidate) <= limit or not buf:
                buf.append(w)
            else:
                out.append(" ".join(buf))
                buf = [w]
        if buf:
            out.append(" ".join(buf))
        return out

    def _normalize_lines_budget(self, text: str, lang: str) -> str:
        """将长段落拆成多小句；确保每小句 token ≤ SENT_TOKEN_LIMIT；行间用换行。"""
        if not text:
            return ""
        if lang == "zh":
            lines = self._split_zh_paragraph(text)
        else:
            lines = self._split_en_paragraph(text)
        # 再次确保每一行都 ≤ limit（极端长词/无标点情况）
        fixed: List[str] = []
        for ln in lines:
            if self._approx_tokens(ln) <= SENT_TOKEN_LIMIT:
                fixed.append(ln.strip())
            else:
                # Fallback 再切
                if lang == "zh":
                    fixed.extend(self._hard_split_by_chars(ln, SENT_TOKEN_LIMIT))
                else:
                    fixed.extend(self._hard_split_by_words(ln, SENT_TOKEN_LIMIT))
        # 清理 & 合并
        fixed = [x.strip() for x in fixed if x and x.strip()]
        return "\n".join(fixed)

    # ---- translate ----
    def _translate_zh_to_en(self, zh: str) -> str:
        """要求短句+换行的英文；仍然会本地再按句预算兜底。"""
        try:
            resp = self.client.models.generate_content(
                model=self.model,
                contents=[
                    (
                        "Translate to concise English. Use short sentences and line breaks; "
                        "no extra commentary, no quotes:\n" + zh
                    )
                ],
            )
            t = (resp.text or "").strip()
            t = self._strip_code_fences(t)
            return t.strip().strip('"').strip("'")
        except Exception as e:
            log(f"translate error: {e}")
            return ""

    def _coerce_bilingual_schema(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        tone = obj.get("tone") or self.role.get("tone_default", "gentle")
        lang = obj.get("lang") or self.role.get("lang", "auto")
        zh, en = None, None

        t = obj.get("text")
        if isinstance(t, dict):
            zh = t.get("zh") or ""
            en = t.get("en") or ""
        elif isinstance(t, str):
            if self._looks_chinese(t):
                zh = t
            else:
                en = t

        # 兼容历史：text_en 顶层
        if not en:
            en = obj.get("text_en") or en or ""
        if not zh and isinstance(t, str) and self._looks_chinese(t):
            zh = t

        if not zh and en:
            zh = "（以下为英文回复）\n" + en

        if zh and not en:
            en = self._translate_zh_to_en(zh)

        # 应用“每小句≤SENT_TOKEN_LIMIT”的本地规范化
        zh_norm = self._normalize_lines_budget(zh or "", lang="zh")
        en_norm = self._normalize_lines_budget(en or "", lang="en")

        data = {
            "tone": tone,
            "lang": lang,
            "text": {"zh": zh_norm, "en": en_norm},
        }
        # 兼容老字段
        data["text_en"] = en_norm
        return data

    # ---------------- Chat ----------------
    def chat(self, user_text: str, forced_tone: Optional[str] = None) -> Dict[str, Any]:
        self._save_message("user", user_text)

        system_instruction = self._system_instruction()
        if forced_tone:
            system_instruction = system_instruction.replace(
                f'"tone": "<{self.role["tone_default"]}', f'"tone": "<{forced_tone}'
            )

        try:
            response = self.client.models.generate_content(
                model=self.model,
                # 不设总 token 硬限，交给本地逐句约束；如需，也可设置 max_output_tokens=1024
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction
                ),
                contents=self._build_contents(user_text, forced_tone=forced_tone),
            )
            raw_text = (response.text or "").strip()
        except Exception as e:
            msg = str(e).lower()
            zh_tip = "目前服务繁忙，请稍后再试。"
            en_tip = "The service is currently busy. Please try again later."
            if "quota" in msg or "rate" in msg or "limit" in msg:
                zh_tip = "请求过于频繁或达到配额上限，请稍后重试。"
                en_tip = (
                    "Requests are too frequent or quota reached. Please retry later."
                )
            fallback_data = {
                "tone": "serious",
                "lang": "zh",
                "text": {"zh": zh_tip, "en": en_tip},
                "text_en": en_tip,
            }
            log(f"Chat generation error: {e}")
            self._save_message("ai", fallback_data["text"]["zh"])
            if self.callbacks.on_message:
                try:
                    self.callbacks.on_message(fallback_data)
                except Exception as cb_err:
                    log(f"on_message callback error: {cb_err}")
            return fallback_data

        cleaned = self._strip_code_fences(raw_text)
        data = self._safe_json(cleaned)
        if not isinstance(data, dict):
            data = {
                "tone": self.role.get("tone_default", "gentle"),
                "lang": self.role.get("lang", "auto"),
                "text": cleaned,
            }

        data = self._coerce_bilingual_schema(data)

        self._save_message("ai", data["text"]["zh"])
        threading.Thread(target=self._summarize_recent_dialog_safe, daemon=True).start()

        if self.callbacks.on_message:
            try:
                self.callbacks.on_message(data)
            except Exception as e:
                log(f"on_message error: {e}")

        return data

    def _extract_json_dict(self, s: str) -> Optional[Dict[str, Any]]:
        """
        尝试从任意文本中提取一个 JSON 对象：
        1) 去掉```围栏后直接loads；
        2) 找到第一个'{'和最后一个'}'做一次截取尝试；
        3) 正则多次匹配花括号块逐个尝试。
        全部失败返回 None。
        """
        if not s:
            return None
        txt = self._strip_code_fences((s or "").strip())

        # 尝试1：整体就是JSON
        obj = self._safe_json(txt)
        if isinstance(obj, dict):
            return obj

        # 尝试2：从首个{到最后一个}
        start = txt.find("{")
        end = txt.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(txt[start : end + 1])
            except Exception:
                pass

        # 尝试3：正则穷举花括号块
        for m in re.finditer(r"\{.*?\}", txt, flags=re.S):
            try:
                return json.loads(m.group(0))
            except Exception:
                continue

        return None

    def _summarize_recent_dialog_safe(self):
        try:
            pairs = self._load_history_pairs(limit_turns=6)
            if not pairs:
                return  # 没得总结就跳过

            dialog = "\n".join(
                [f"User: {p['user']}\nAI: {p.get('ai', '')}" for p in pairs]
            )

            # 两段式：把“只返JSON”的要求和对话分开传，成功率更高
            resp = self.client.models.generate_content(
                model=self.model,
                contents=[SUMMARY_INSTRUCTION, dialog],
            )
            j = (resp.text or "").strip()

            data = self._extract_json_dict(j)

            # 降级兜底：构造一个最小可用摘要，保证不抛错也能入库
            if not isinstance(data, dict):
                last_user = pairs[-1]["user"] if pairs else ""
                # 简易关键词：从最后一条用户话里抽几个较长的“词”（中文直接按非空格切也行）
                raw_tokens = re.split(r"[\\s，。！？、,.!?;:\\-]+", last_user)
                kws = [t for t in raw_tokens if 2 <= len(t) <= 12][:5]
                data = {
                    "user_emotion": "未知",
                    "keywords": kws,
                    "events": [],
                    "ai_behavior": "总结失败，已降级保存",
                    "summary": self._normalize_lines_budget(
                        f"对话记录已保存。最近一次用户提到：{last_user[:60]}…", "zh"
                    ),
                }

            # 入库
            self._save_summary(data)
            # log("save summary done") # just for debug

        except Exception as e:
            log(f"summarize error: {e}")
            # 这里不要 print_exc 直接把栈打出来污染控制台；只在日志里留一条
            # 如果你确实想看栈：log(traceback.format_exc())

    # ---------------- Imagine ----------------
    def imagine_and_log(self) -> Optional[Dict[str, Any]]:
        try:
            now = datetime.datetime.now(self.tz)
            when = now - datetime.timedelta(minutes=random.randint(10, 180))
            prompt = IMAGINE_EVENT_INSTRUCTION.replace(
                "某个时间点", when.strftime("%Y-%m-%d %H:%M")
            )
            resp = self.client.models.generate_content(
                model=self.model, contents=[prompt]
            )
            j = (resp.text or "").strip()
            data = json.loads(j)
            print(data)  # Debug
            self._save_event("imagination", data)
            if self.callbacks.on_imagination:
                try:
                    self.callbacks.on_imagination(data)
                except Exception as e:
                    log(f"on_imagination error: {e}")
            return data
        except Exception as e:
            log(f"imagination error: {e}")
            return None

    # ---------------- Observe ----------------
    def observe_screens_and_coach(
        self, save_dir: str = "Profile/screens"
    ) -> Optional[Dict[str, Any]]:
        if mss is None:
            log("未安装 mss，无法截屏。pip install mss")
            return None
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
        files: List[pathlib.Path] = []
        try:
            # 1) 截屏
            with mss.mss() as sct:
                for i, mon in enumerate(
                    sct.monitors[1:], start=1
                ):  # 0=全屏，1开始为每块屏
                    img = sct.grab(mon)
                    p = pathlib.Path(save_dir) / f"screen_{i}_{int(time.time())}.png"
                    mss.tools.to_png(img.rgb, img.size, output=str(p))  # pyright: ignore[reportAttributeAccessIssue]
                    files.append(p)

            # 2) OCR/粗分析（允许返回 Markdown/代码块，这里仅作为线索）
            analysis = pic_multi_analysis(files, model=self.model) or ""

            # 3) 让模型做“活动判定 + 建议”，按我们标准 JSON 输出
            resp = self.client.models.generate_content(
                model=self.model,
                contents=[OBSERVE_COACH_INSTRUCTION, analysis],
            )
            raw = (resp.text or "").strip()

            # 4) 剥代码围栏 & 解析 JSON
            cleaned = self._strip_code_fences(raw)
            data = self._safe_json(cleaned)
            if not isinstance(data, dict):
                # 兜底：构造一个最小可用 JSON
                zh = "我还没看懂你当前在做什么，但可以把关键窗口放到前台再截一次屏，我会继续帮你。"
                en = (
                    self._translate_zh_to_en(zh)
                    or "I couldn't determine your current activity. Please bring the key window to the front and capture again."
                )
                data = {
                    "tone": "gentle",
                    "lang": "auto",
                    "text": {"zh": zh, "en": en},
                    "activity": {"zh": "活动：无法确定", "en": "Activity: not sure"},
                    "focus": "other",
                    "confidence": 0.2,
                    "evidence": [],
                }

            # 5) 规整双语 & 逐句≤250 tokens（利用你已有的本地规整工具）
            # text
            zh_text = (
                data.get("text", {}).get("zh", "")
                if isinstance(data.get("text"), dict)
                else ""
            )
            en_text = (
                data.get("text", {}).get("en", "")
                if isinstance(data.get("text"), dict)
                else ""
            )
            if not zh_text and isinstance(data.get("text"), str):
                # 兼容模型不守规矩把 text 写成字符串
                zh_text = data["text"] if self._looks_chinese(data["text"]) else ""
                en_text = "" if zh_text else data["text"]

            if zh_text and not en_text:
                en_text = self._translate_zh_to_en(zh_text)
            zh_text = self._normalize_lines_budget(zh_text, lang="zh")
            en_text = self._normalize_lines_budget(en_text, lang="en")

            # activity
            act = data.get("activity", {})
            if isinstance(act, dict):
                zh_act = self._normalize_lines_budget(
                    act.get("zh", "") or "", lang="zh"
                )
                en_act = self._normalize_lines_budget(
                    act.get("en", "") or "", lang="en"
                )
            else:
                # 兼容字符串
                if isinstance(act, str):
                    zh_act = self._normalize_lines_budget(
                        act if self._looks_chinese(act) else "", "zh"
                    )
                    en_act = self._normalize_lines_budget("" if zh_act else act, "en")
                else:
                    zh_act, en_act = "", ""

            # focus & confidence & evidence 容错
            focus = data.get("focus", "other")
            if focus not in {
                "coding",
                "reading",
                "video",
                "gaming",
                "music",
                "chat",
                "browsing",
                "document",
                "terminal",
                "meeting",
                "other",
            }:
                focus = "other"
            try:
                confidence = float(data.get("confidence", 0.5))
                confidence = max(0.0, min(1.0, confidence))
            except Exception:
                confidence = 0.5
            evidence = data.get("evidence") or []
            if not isinstance(evidence, list):
                evidence = [str(evidence)]

            # 6) 统一返回结构
            coach = {
                "tone": data.get("tone", "gentle"),
                "lang": data.get("lang", "auto"),
                "text": {"zh": zh_text, "en": en_text},
                "activity": {"zh": zh_act, "en": en_act},
                "focus": focus,
                "confidence": confidence,
                "evidence": [
                    self._normalize_lines_budget(str(x), "en") for x in evidence
                ][:3],
                "text_en": en_text,  # 兼容老字段
            }

            # 7) 入库 + 回调
            self._save_event("observe", {"analysis": analysis, "coach": coach})
            if self.callbacks.on_observe:
                try:
                    self.callbacks.on_observe(coach)
                except Exception as e:
                    log(f"on_observe error: {e}")

            return coach

        except Exception as e:
            log(f"observe error: {e}")
            log(f"{traceback.format_exc()}")
            return None

    # ---------------- Close ----------------
    def close(self):
        self._stop_flag = True
        log("已请求停止调度线程。")
