# gemini_companion.py — persona + bilingual + robust parsing + per-sentence 250-token limit
from __future__ import annotations

import os
import re
import json
import sqlite3
import threading
import time
import random
import datetime
import configparser
import pathlib
from dataclasses import dataclass
from typing import Callable, Optional, List, Dict, Any

# Gemini Official SDK
from google import genai
from google.genai import types

# Your multimodal analysis helper (you provided this)
from images_apis import pic_multi_analysis

# Optional screenshot
try:
    import mss  # type: ignore
except Exception:
    mss = None

# Pretty logs
try:
    from rich.console import Console  # type: ignore

    console = Console()

    def log(msg: str):
        console.print(f"[bold cyan][Companion][/bold cyan] {msg}")
except Exception:
    console = None

    def log(msg: str):
        print(f"[Companion] {msg}")


# -----------------------------
# Defaults & budgets
# -----------------------------
DEFAULT_ROLE_INI = "role.ini"
DEFAULT_DB = "companion.db"
DEFAULT_SCHEDULE = "schedule.json"
DEFAULT_MODEL = "gemini-2.5-flash"

# 每小句 token 上限（近似）
SENT_TOKEN_LIMIT = 250
# 中文“首轮”切分的单句最大字符（长句会进一步细分）
ZH_SENT_CHAR_HINT = 80
# 英文“首轮”切分的单句最大词数（长句会进一步细分）
EN_SENT_WORD_HINT = 24

SYSTEM_PROMPT_TEMPLATE = """你是一个人格化的 AI 伴侣，需具备以下能力：
- 记住用户的偏好、事件、情绪轨迹（但不要复述隐私）
- 回答时简洁、真诚、自然，避免居高临下
- 语气（tone）从 {allowed_tones} 中选择其一，尽量贴合用户当下状态
- 语言（lang）自动跟随用户输入（zero-shot 多语种）
- 禁止水答复：必须结合上下文给出具体、可执行或有情绪价值的内容

角色设定（来自 INI）：
- 名称：{role_name}
- 时代背景：{era}
- 价值观/世界观：{values}
- 外貌/形象：{appearance}
- 人设摘要：{persona_summary}
- 说话风格：{persona_style}
- 与用户关系：{relationship}（据此选择称呼）
- 禁止事项：{boundaries}

【长度与格式要求】
- 只能返回一段 JSON；不要出现任何 Markdown 代码块（禁止 ```），不要多余解释。
- 中文优先、英文为辅；若内容较多请自动**分短句并换行**。
- **每一小句**必须 ≤ ~{sent_limit} tokens（近似），超出需自动拆分为多小句。
- 中文建议短句（约 {zh_hint} 字内）并按句号/逗号/顿号自然换行；英文建议短句（约 {en_hint} 词内），按句号/从句自然换行。

【JSON 结构（字段名必须一致，全部必填）】
{{
  "tone": "<{default_tone} | {allowed_tones}>",
  "lang": "<auto|zh|en|...>",
  "text": {{
    "zh": "<中文回复；已按短句换行；每小句不超 {sent_limit} tokens>",
    "en": "<对应英文精简翻译；也按短句换行；每小句不超 {sent_limit} tokens>"
  }}
}}
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

OBSERVE_COACH_INSTRUCTION = """以下是对屏幕截图的文字/代码/活动提取结果。请用轻松幽默的方式做两件事：
1) 用一句话判断我大致在干嘛（如果不确定就直说）。
2) 给出一句温柔的建议或玩笑。
输出 JSON：
{
  "tone": "<gentle|teasing|serious|playful|comforting>",
  "lang": "auto",
  "text": "<一句话建议或玩笑>",
  "activity": "<一句话你猜测我在干嘛>"
}
只返回 JSON。"""


@dataclass
class Callbacks:
    on_message: Optional[Callable[[Dict[str, Any]], None]] = None
    on_imagination: Optional[Callable[[Dict[str, Any]], None]] = None
    on_observe: Optional[Callable[[Dict[str, Any]], None]] = None
    on_schedule_emit: Optional[Callable[[Dict[str, Any]], None]] = None


class GeminiCompanion:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        role_ini: str = DEFAULT_ROLE_INI,
        db_path: str = DEFAULT_DB,
        schedule_path: str = DEFAULT_SCHEDULE,
        max_history_turns: int = 20,
        callbacks: Optional[Callbacks] = None,
        timezone: str = "Asia/Shanghai",
    ):
        self.model = model
        self.role_ini = role_ini
        self.db_path = db_path
        self.schedule_path = schedule_path
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
        cfg = configparser.ConfigParser()
        cfg.read(self.role_ini, encoding="utf-8")
        return {
            "name": cfg.get("meta", "name", fallback="Mirai"),
            "lang": cfg.get("meta", "lang", fallback="auto"),
            "era": cfg.get("persona", "era", fallback="当代"),
            "values": cfg.get("persona", "values", fallback="善良、真诚"),
            "appearance": cfg.get("persona", "appearance", fallback="亲切友好"),
            "persona_summary": cfg.get("persona", "summary", fallback="温柔、耐心"),
            "persona_style": cfg.get("persona", "style", fallback="自然口语"),
            "relationship": cfg.get("persona", "relationship", fallback="知心朋友"),
            "boundaries": cfg.get("boundaries", "text", fallback=""),
            "tone_default": cfg.get("tone", "default", fallback="gentle"),
            "tone_allowed": cfg.get(
                "tone", "allowed", fallback="gentle,teasing,serious,playful,comforting"
            ),
        }

    def _system_instruction(self) -> str:
        return SYSTEM_PROMPT_TEMPLATE.format(
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
                {
                    "time": "08:30",
                    "action": "send",
                    "text": "早呀～起床喝水，今天也要加油鸭！",
                    "tone": "gentle",
                },
                {
                    "time": "22:45",
                    "action": "send",
                    "text": "要不要准备睡觉啦？我可以给你讲个小故事。",
                    "tone": "playful",
                },
            ]
            p.write_text(
                json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            log(f"已创建示例日程 {self.schedule_path}")
        self._schedule = json.loads(p.read_text(encoding="utf-8"))

    def _schedule_loop(self):
        log("行为树/日程 调度线程已启动。")
        fired_today = set()
        last_day = datetime.date.today()
        while not self._stop_flag:
            now = datetime.datetime.now(self.tz)
            day = now.date()
            if day != last_day:
                fired_today.clear()
                last_day = day
            hhmm = now.strftime("%H:%M")
            for item in self._schedule:
                tag = f"{day}-{item.get('time')}-{item.get('text', '')}"
                if item.get("time") == hhmm and tag not in fired_today:
                    fired_today.add(tag)
                    payload = {"ts": now.isoformat(), **item}
                    if self.callbacks.on_schedule_emit:
                        try:
                            self.callbacks.on_schedule_emit(payload)
                        except Exception as e:
                            log(f"on_schedule_emit error: {e}")
                    if item.get("action") == "send" and "text" in item:
                        self.chat(item["text"], forced_tone=item.get("tone"))
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

    def _summarize_recent_dialog_safe(self):
        try:
            pairs = self._load_history_pairs(limit_turns=6)
            dialog = "\n".join(
                [f"User: {p['user']}\nAI: {p.get('ai', '')}" for p in pairs]
            )
            prompt = f"{SUMMARY_INSTRUCTION}\n\n---\n{dialog}"
            resp = self.client.models.generate_content(
                model=self.model, contents=[prompt]
            )
            j = (resp.text or "").strip()
            data = json.loads(j)
            self._save_summary(data)
        except Exception as e:
            log(f"summarize error: {e}")

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
        self, save_dir: str = "screens"
    ) -> Optional[Dict[str, Any]]:
        if mss is None:
            log("未安装 mss，无法截屏。pip install mss")
            return None
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
        files: List[pathlib.Path] = []
        try:
            with mss.mss() as sct:
                for i, mon in enumerate(sct.monitors[1:], start=1):
                    img = sct.grab(mon)
                    p = pathlib.Path(save_dir) / f"screen_{i}_{int(time.time())}.png"
                    mss.tools.to_png(img.rgb, img.size, output=str(p))
                    files.append(p)
            analysis = pic_multi_analysis(files, model=self.model)
            resp = self.client.models.generate_content(
                model=self.model,
                contents=[OBSERVE_COACH_INSTRUCTION, analysis or ""],
            )
            j = (resp.text or "").strip()
            data = json.loads(j)
            self._save_event("observe", {"analysis": analysis, "coach": data})
            if self.callbacks.on_observe:
                try:
                    self.callbacks.on_observe(data)
                except Exception as e:
                    log(f"on_observe error: {e}")
            return data
        except Exception as e:
            log(f"observe error: {e}")
            return None

    # ---------------- Close ----------------
    def close(self):
        self._stop_flag = True
        log("已请求停止调度线程。")


# Demo (optional)
if __name__ == "__main__":
    cb = Callbacks(on_message=lambda m: log(f"AI 回复：{m}"))
    bot = GeminiCompanion(callbacks=cb, api_key=os.getenv("GOOGLE_API_KEY"))
    print(bot.chat("我今晚有点焦虑，明天考试怕挂。"))
    print(bot.chat("你是谁？"))
    time.sleep(1)
    bot.close()
