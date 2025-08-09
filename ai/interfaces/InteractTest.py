#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, time, argparse, datetime, importlib.util, sqlite3, pathlib
from typing import Any, Dict
from Interact import GeminiCompanion, Callbacks

HERE = pathlib.Path(__file__).parent.resolve()


def try_load_api_key_from_system_config() -> str | None:
    """
    可选：如果同目录有 system_config.py，且定义了 GOOGLE_API_KEY / API_KEY，就用它。
    """
    cfg_path = HERE / "system_config.py"
    if not cfg_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("system_config", str(cfg_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return getattr(mod, "GOOGLE_API_KEY", None) or getattr(mod, "API_KEY", None)


def ensure_next_minute_schedule(
    schedule_path: str,
    text: str = "（自动日程测试）现在是我来打个招呼～",
    tone: str = "playful",
):
    """
    写一个 schedule.json，让它在下一分钟触发一次 'send'。
    注意：GeminiCompanion 初始化时会加载 schedule.json，
    所以需要在实例化之前写入。
    """
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
    nxt = now + datetime.timedelta(minutes=1)
    hhmm = nxt.strftime("%H:%M")
    data = [{"time": hhmm, "action": "send", "text": text, "tone": tone}]
    pathlib.Path(schedule_path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[test] 已写入 {schedule_path}，将在 {hhmm} 触发。")


def db_counts(db_path: str) -> Dict[str, int]:
    q = {
        "messages": "SELECT COUNT(*) FROM messages",
        "summaries": "SELECT COUNT(*) FROM summaries",
        "events": "SELECT COUNT(*) FROM events",
    }
    out: Dict[str, int] = {}
    with sqlite3.connect(db_path) as conn:
        for k, sql in q.items():
            (n,) = conn.execute(sql).fetchone()
            out[k] = int(n)
    return out


def db_tail(db_path: str, table: str, n: int = 3):
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            f"SELECT ts, * FROM {table} ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
    return rows


def main():
    ap = argparse.ArgumentParser(description="GeminiCompanion smoke test")
    ap.add_argument(
        "--api-key",
        default=None,
        help="Google Gemini API key（或用环境变量 GOOGLE_API_KEY）",
    )
    ap.add_argument(
        "--say", action="append", default=[], help="对话文本，可多次传入以形成多轮"
    )
    ap.add_argument("--imagine", action="store_true", help="触发一次想象事件")
    ap.add_argument(
        "--observe", action="store_true", help="截屏分析一次（需要 mss 权限）"
    )
    ap.add_argument(
        "--schedule", action="store_true", help="创建下一分钟触发的日程并等待"
    )
    ap.add_argument(
        "--run-mins", type=int, default=0, help="保持运行 N 分钟（等待定时任务）"
    )
    ap.add_argument(
        "--db", default="companion.db", help="数据库路径（默认 companion.db）"
    )
    ap.add_argument("--role-ini", default="role.ini", help="角色 INI（默认 role.ini）")
    ap.add_argument(
        "--schedule-file",
        default="schedule.json",
        help="日程文件（默认 schedule.json）",
    )
    args = ap.parse_args()

    # API Key 优先级：CLI > ENV > system_config.py
    api_key = (
        args.api_key
        or os.getenv("GOOGLE_API_KEY")
        or try_load_api_key_from_system_config()
    )
    if not api_key:
        print(
            "❌ 请提供 API Key：--api-key 或环境变量 GOOGLE_API_KEY 或 system_config.py"
        )
        sys.exit(1)

    # 如果需要测试日程，在实例化前写入 schedule.json（下一分钟触发）
    if args.schedule:
        ensure_next_minute_schedule(args.schedule_file)

    # 回调打印
    def on_message(msg: Dict[str, Any]):
        print("\n[回调] AI 回复(JSON):")
        print(json.dumps(msg, ensure_ascii=False, indent=2))

    def on_imagination(ev: Dict[str, Any]):
        print("\n[回调] 想象事件(JSON):")
        print(json.dumps(ev, ensure_ascii=False, indent=2))

    def on_observe(res: Dict[str, Any]):
        print("\n[回调] 观察建议(JSON):")
        print(json.dumps(res, ensure_ascii=False, indent=2))

    def on_schedule_emit(s: Dict[str, Any]):
        print("\n[回调] 日程触发:")
        print(json.dumps(s, ensure_ascii=False, indent=2))

    cb = Callbacks(
        on_message=on_message,
        on_imagination=on_imagination,
        on_observe=on_observe,
        on_schedule_emit=on_schedule_emit,
    )

    # 实例化
    bot = GeminiCompanion(
        api_key=api_key,
        model="gemini-2.5-flash",  # 你也可以改成 1.5-pro 做更强推理
        role_ini=args.role_ini,
        db_path=args.db,
        schedule_path=args.schedule_file,
        max_history_turns=8,
        callbacks=cb,
        timezone="Asia/Shanghai",
    )

    # 如果没有传 --say，给一条默认消息
    says = args.say or ["我们来做一个端到端测试：先从自我介绍开始吧。"]
    for i, utter in enumerate(says, 1):
        print(f"\n=== [对话 {i}] USER: {utter}")
        reply = bot.chat(utter)
        print("\n=== [对话 {i}] AI(JSON):")
        print(json.dumps(reply, ensure_ascii=False, indent=2))

    # 想象事件
    if args.imagine:
        ev = bot.imagine_and_log()
        if ev:
            print("\n=== 想象事件(JSON) ===")
            print(json.dumps(ev, ensure_ascii=False, indent=2))

    # 观察屏幕
    if args.observe:
        obs = bot.observe_screens_and_coach()
        if obs:
            print("\n=== 观察建议(JSON) ===")
            print(json.dumps(obs, ensure_ascii=False, indent=2))
            print(
                "（提示：macOS/Wayland 可能需要截屏权限；Linux 需有 Xorg 或兼容终端）"
            )

    # 等待 N 分钟（用于 schedule 触发 & 异步摘要）
    wait_secs = max(args.run_mins * 60, 5 if (args.schedule or args.imagine) else 0)
    if wait_secs:
        print(f"\n[test] 等待 {wait_secs} 秒以便定时/异步任务完成…")
        time.sleep(wait_secs)

    # 打印 DB 摘要
    print("\n=== 数据库统计 ===")
    counts = db_counts(args.db)
    for k, v in counts.items():
        print(f"{k}: {v}")

    # 各表最近记录
    try:
        print("\n=== summaries 最近 3 条 ===")
        for row in db_tail(args.db, "summaries", 3):
            # row: (ts, id, ts, data) 因 SELECT ts,*，前两个 ts 重复，简单打印后两项
            print(row)

        print("\n=== events 最近 3 条 ===")
        for row in db_tail(args.db, "events", 3):
            print(row)
    except Exception as e:
        print(f"[warn] 读取 DB 细节失败：{e}")

    # 结束
    bot.close()
    print("\n✅ 测试结束。")


if __name__ == "__main__":
    main()

# python InteractTest.py --api-key AIzaSyBky3d2cfB6_hHiEktVwhAN-68UaR2G7mk --observe
