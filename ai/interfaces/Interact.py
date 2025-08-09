# gemini_companion.py
# --------------------------------------------------------------
# pip install google-genai mss rich
# 需要你同目录下有 images.py（你已提供，内含 pic_multi_analysis）
# --------------------------------------------------------------

from __future__ import annotations

import os
import json
import sqlite3
import threading
import time
import random
import datetime
import configparser
import pathlib
import platform

from dataclasses import dataclass
from typing import Callable, Optional, List, Dict, Any

# Gemini 官方 SDK（与你给的示例一致：google.genai）
from google import genai
from google.genai import types

# 你上传的多模态截图分析
from images_apis import pic_multi_analysis

# 截屏：跨平台
try:
    import mss  # type: ignore
except Exception:
    mss = None

# 颜色日志（可选）
try:
    from rich.console import Console  # type: ignore

    console = Console()

    def log(msg: str):
        if console is not None:
            console.print(f"[bold cyan][Companion][/bold cyan] {msg}")
        else:
            raise ImportError("rich 库未安装，请运行 pip install rich")

except Exception:
    console = None

    def log(msg: str):
        print(f"[Companion] {msg}")


# -----------------------------
# 常量与默认文件
# -----------------------------
DEFAULT_ROLE_INI = "role.ini"
DEFAULT_DB = "companion.db"
DEFAULT_SCHEDULE = "schedule.json"

DEFAULT_MODEL = "gemini-2.5-flash"  # 你也可以换成 1.5-pro 之类的

# 系统提示词骨架（你可以按需改）
SYSTEM_PROMPT_TEMPLATE = """你是一个人格化的 AI 伴侣，需具备以下能力：
- 记住用户的偏好、事件、情绪轨迹（但不要复述隐私）
- 回答时简洁、真诚、自然，避免居高临下
- 语气（tone）从 {allowed_tones} 中选择其一，尽量贴合用户当下状态
- 语言（lang）自动跟随用户输入（zero-shot 多语种）

角色人设（来自INI）：
- 名称：{role_name}
- 人设摘要：{persona_summary}
- 说话风格：{persona_style}
- 禁止事项：{boundaries}

输出格式（必须返回 JSON，严格字段名，不要多余解释）：
{{
  "tone": "<{default_tone} | {allowed_tones}>",
  "lang": "<auto|zh|en|...>",
  "text": "<最终给用户的回复文本>"
}}
"""

# 想象事件的结构化模板
IMAGINE_EVENT_INSTRUCTION = """请在不暴露内部推理的情况下，虚构你在某个时间点经历的一次小事件，并以 JSON 返回（中文或跟随用户语言）：
必须字段：
{{
  "timestamp": "<ISO时间，例如 2025-08-08T20:31:00+08:00>",
  "place": "<地点或场景>",
  "event": "<发生了什么（1-2句）>",
  "people": ["<相关角色，可空>"],
  "feeling": "<你的感受（简短）>",
  "result": "<结果（简短）>",
  "impact_on_user": "<这对用户可能有什么积极帮助或启发>",
  "memory_tags": ["<标签1>", "<标签2>"]
}}
只返回 JSON。"""

# 从对话生成摘要的提示
SUMMARY_INSTRUCTION = """请阅读以下多轮对话（User 与 AI），输出结构化对话摘要（仅 JSON）：
字段：
{{
  "user_emotion": "<如: 开心/焦虑/低落/无聊/平静...>",
  "keywords": ["<关键词1>", "<关键词2>", "..."],
  "events": ["<若用户提及的关键事件（可空）>"],
  "ai_behavior": "<本次AI的对话风格（如：共情/建议/玩笑/鼓励）>",
  "summary": "<2-3句总结>"
}}
只返回 JSON。"""

# 屏幕观察后的建议（基于 pic_multi_analysis 的文本结果再次提炼）
OBSERVE_COACH_INSTRUCTION = """以下是对屏幕截图的文字/代码/活动提取结果。请用轻松幽默的方式做两件事：
1) 用一句话判断我大致在干嘛（如果不确定就直说）。
2) 给出一句温柔的小建议或玩笑。
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
    """用户可注入的回调"""

    on_message: Optional[Callable[[Dict[str, Any]], None]] = (
        None  # 模型最终回复（JSON）
    )
    on_imagination: Optional[Callable[[Dict[str, Any]], None]] = (
        None  # 想象事件（JSON）
    )
    on_observe: Optional[Callable[[Dict[str, Any]], None]] = (
        None  # 截屏观察建议（JSON）
    )
    on_schedule_emit: Optional[Callable[[Dict[str, Any]], None]] = (
        None  # 定时行为树触发（字典）
    )


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
        """
        - 角色配置：INI（不存在会自动创建）
        - 记忆：SQLite（消息、摘要、事件、图谱节点/边）
        - 行为树：schedule.json（每天按时间触发）
        """
        self.model = model
        self.role_ini = role_ini
        self.db_path = db_path
        self.schedule_path = schedule_path
        self.max_history_turns = max_history_turns
        self.callbacks = callbacks or Callbacks()
        self.tz = datetime.timezone(datetime.timedelta(hours=8))  # 简单处理东八区

        # Gemini 客户端
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
        self.client = genai.Client()

        # 文件与表
        self._ensure_role_ini()
        self.role = self._load_role_ini()
        self._init_db()
        self._load_schedule()

        # 后台线程：行为树与观察
        self._stop_flag = False
        self._scheduler_thread = threading.Thread(
            target=self._schedule_loop, daemon=True
        )
        self._scheduler_thread.start()

        log("GeminiCompanion 已初始化。")

    # --------------- 角色系统 ---------------
    def _ensure_role_ini(self):
        p = pathlib.Path(self.role_ini)
        if p.exists():
            return
        cfg = configparser.ConfigParser()
        cfg["meta"] = {"name": "Mirai", "lang": "auto"}
        cfg["persona"] = {
            "summary": "温柔、耐心、俏皮、会恰到好处地开玩笑",
            "style": "自然口语、简洁；先共情再给建议；偶尔撒娇但不过度",
        }
        cfg["boundaries"] = {
            "text": "不提供医疗/法律/投资建议；避免泄露隐私；不过度依赖权威语气"
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
        role = {
            "name": cfg.get("meta", "name", fallback="Mirai"),
            "lang": cfg.get("meta", "lang", fallback="auto"),
            "persona_summary": cfg.get("persona", "summary", fallback="温柔、耐心"),
            "persona_style": cfg.get("persona", "style", fallback="自然口语"),
            "boundaries": cfg.get("boundaries", "text", fallback=""),
            "tone_default": cfg.get("tone", "default", fallback="gentle"),
            "tone_allowed": cfg.get(
                "tone", "allowed", fallback="gentle,teasing,serious,playful,comforting"
            ),
        }
        return role

    def _system_instruction(self) -> str:
        return SYSTEM_PROMPT_TEMPLATE.format(
            allowed_tones=self.role["tone_allowed"],
            default_tone=self.role["tone_default"],
            role_name=self.role["name"],
            persona_summary=self.role["persona_summary"],
            persona_style=self.role["persona_style"],
            boundaries=self.role["boundaries"],
        )

    # --------------- 数据库 ---------------
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS messages(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            role TEXT NOT NULL,  -- 'user' | 'ai'
            content TEXT NOT NULL
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS summaries(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            data TEXT NOT NULL -- JSON
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS events(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            kind TEXT NOT NULL,   -- 'imagination' | 'observe'
            data TEXT NOT NULL    -- JSON
        )""")
        # 简易图谱（可选）
        c.execute("""CREATE TABLE IF NOT EXISTS graph_nodes(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT NOT NULL,
            type TEXT NOT NULL,
            props TEXT NOT NULL   -- JSON
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS graph_edges(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            src INTEGER NOT NULL,
            dst INTEGER NOT NULL,
            type TEXT NOT NULL,
            props TEXT NOT NULL,  -- JSON
            FOREIGN KEY(src) REFERENCES graph_nodes(id),
            FOREIGN KEY(dst) REFERENCES graph_nodes(id)
        )""")
        conn.commit()
        conn.close()

    def _db(self):
        return sqlite3.connect(self.db_path)

    # --------------- 行为树（日程调度） ---------------
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
        """每分钟轮询一次，触发 schedule.json 中的事件"""
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
                    # 回调
                    if self.callbacks.on_schedule_emit:
                        try:
                            self.callbacks.on_schedule_emit(payload)
                        except Exception as e:
                            log(f"on_schedule_emit error: {e}")
                    # 如果是自动发话
                    if item.get("action") == "send" and "text" in item:
                        # 直接把日程文本当作用户输入，用指定 tone
                        self.chat(item["text"], forced_tone=item.get("tone"))
            time.sleep(30)  # 半分钟检查一次

    # --------------- 历史与记忆 ---------------
    def _save_message(self, role: str, content: str):
        with self._db() as conn:
            conn.execute(
                "INSERT INTO messages(ts, role, content) VALUES (?, ?, ?)",
                (datetime.datetime.now(self.tz).isoformat(), role, content),
            )

    def _load_history_pairs(self, limit_turns: int) -> List[Dict[str, str]]:
        """最近 N 轮（user, ai）对"""
        with self._db() as conn:
            rows = conn.execute(
                "SELECT role, content FROM messages ORDER BY id DESC LIMIT ?",
                (limit_turns * 2,),
            ).fetchall()
        rows.reverse()
        pairs = []
        buf = {}
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
        """
        简易检索：从 summaries 里按关键词做 LIKE + 按时间降序
        （如果你要向量召回，后续可接 FAISS，这里先给最小可用）
        """
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

    # --------------- Prompt 构造 ---------------
    def _build_contents(
        self, user_text: str, forced_tone: Optional[str] = None
    ) -> List[Any]:
        # 最近历史
        pairs = self._load_history_pairs(self.max_history_turns)
        history_str = ""
        for p in pairs:
            history_str += f"User: {p['user']}\nAI: {p.get('ai', '')}\n"

        # 召回记忆摘要
        mems = self._retrieve_memory_snippets(user_text, top_k=5)
        mem_str = "\n".join(mems) if mems else "（无）"

        # 拼装
        guidance = f"""[对话历史]
{history_str or "（空）"}

[相关记忆摘要]
{mem_str}

[当前用户输入]
{user_text}

请按系统要求的 JSON 输出（包含 tone/lang/text）。"""
        return [guidance]

    # --------------- 与模型对话 ---------------
    def chat(self, user_text: str, forced_tone: Optional[str] = None) -> Dict[str, Any]:
        """主入口：发话 → 结构化回复（JSON）"""
        self._save_message("user", user_text)

        system_instruction = self._system_instruction()
        if forced_tone:
            # 小小钩子：把默认 tone 换掉
            system_instruction = system_instruction.replace(
                '"tone": "<{default_tone}', f'"tone": "<{forced_tone}'
            )

        resp = self.client.models.generate_content(
            model=self.model,
            config=types.GenerateContentConfig(system_instruction=system_instruction),
            contents=self._build_contents(user_text, forced_tone=forced_tone),
        )

        text = (resp.text or "").strip()
        # 期望纯 JSON
        try:
            data = json.loads(text)
        except Exception:
            data = {
                "tone": self.role["tone_default"],
                "lang": self.role["lang"],
                "text": text,
            }

        # 保存 AI 消息
        self._save_message("ai", data.get("text", ""))

        # 对话完成后 → 生成摘要入库（异步）
        threading.Thread(target=self._summarize_recent_dialog_safe, daemon=True).start()

        # 回调
        if self.callbacks.on_message:
            try:
                self.callbacks.on_message(data)
            except Exception as e:
                log(f"on_message error: {e}")

        return data

    def _summarize_recent_dialog_safe(self):
        try:
            pairs = self._load_history_pairs(limit_turns=6)
            # 拼成多轮
            dialog = "\n".join(
                [f"User: {p['user']}\nAI: {p.get('ai', '')}" for p in pairs]
            )
            prompt = f"{SUMMARY_INSTRUCTION}\n\n---\n{dialog}"
            resp = self.client.models.generate_content(
                model=self.model,
                contents=[prompt],
            )
            j = (resp.text or "").strip()
            data = json.loads(j)
            self._save_summary(data)
        except Exception as e:
            log(f"summarize error: {e}")

    # --------------- 想象事件 ---------------
    def imagine_and_log(self) -> Optional[Dict[str, Any]]:
        try:
            now = datetime.datetime.now(self.tz)
            when = now - datetime.timedelta(minutes=random.randint(10, 180))
            prompt = IMAGINE_EVENT_INSTRUCTION.replace(
                "某个时间点", when.strftime("%Y-%m-%d %H:%M")
            )
            resp = self.client.models.generate_content(
                model=self.model,
                contents=[prompt],
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

    # --------------- 截屏观察 ---------------
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
                for i, mon in enumerate(
                    sct.monitors[1:], start=1
                ):  # 0是“全部屏幕”，1开始是每个
                    img = sct.grab(mon)
                    p = pathlib.Path(save_dir) / f"screen_{i}_{int(time.time())}.png"
                    mss.tools.to_png(img.rgb, img.size, output=str(p))
                    files.append(p)
            # 用你给的 images.py → Gemini 多模态分析
            analysis = pic_multi_analysis(files, model=self.model)

            # 再喂一个小提示，让它给建议/玩笑
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

    # --------------- 关闭 ---------------
    def close(self):
        self._stop_flag = True
        log("已请求停止调度线程。")


# -----------------------------
# 简单用法示例（你可以删掉这段）
# -----------------------------
if __name__ == "__main__":
    # 1) 先 export GOOGLE_API_KEY=xxx 或在构造时传 api_key
    cb = Callbacks(
        on_message=lambda m: log(f"AI 回复：{m}"),
        on_imagination=lambda e: log(f"想象事件：{e}"),
        on_observe=lambda o: log(f"观察建议：{o}"),
        on_schedule_emit=lambda s: log(f"日程触发：{s}"),
    )
    bot = GeminiCompanion(
        callbacks=cb, api_key="AIzaSyBky3d2cfB6_hHiEktVwhAN-68UaR2G7mk"
    )

    # 主动聊一句
    print(bot.chat("我今晚有点焦虑，明天考试怕挂。"))
    print(bot.chat("有时候我会在睡前幻想怎么杀死我认识的每个人，感觉很爽"))

    # 想象一条事件
    print(bot.imagine_and_log())

    # 定时看看屏幕（如果要手动触发一次）
    # print(bot.observe_screens_and_coach())

    # 运行若干秒后退出（演示）
    time.sleep(5)
    bot.close()
