# voidmate_ui.py
import time
import threading
import PySimpleGUI as sg
from typing import Callable, Optional


class VoidMateUI:
    """
    轻量 GUI：日志窗口 + 输入框 + 托盘
    - Multiline 接管 stdout/stderr（print 直接显示）
    - 输入回车或点“发送”调用 on_send（后台线程）
    - 托盘菜单：显示/隐藏、退出
    """

    LOG_KEY = "-LOG-"
    IN_KEY = "-IN-"
    SEND_KEY = "-SEND-"
    LOG_EVT = "-LOG-EVT-"
    REPLY_EVT = "-AI-REPLY-"

    def __init__(
        self,
        title: str = "VoidMate",
        icon: Optional[str] = None,
        on_send: Optional[Callable[[str], Optional[str]]] = None,
        width: int = 90,
        height: int = 24,
        enable_tray: bool = True,
        minimize_to_tray_on_close: bool = True,
        reroute_print: bool = True,
    ):
        """
        :param on_send: 回调函数，接收用户输入文本，返回回复字符串（可为 None）。
                        在后台线程执行，结果通过事件推回 UI。
        """
        self.title = title
        self.icon = icon
        self.on_send = on_send or (lambda msg: f"(echo) {msg}")
        self.enable_tray = enable_tray
        self.minimize_to_tray_on_close = minimize_to_tray_on_close

        layout = [
            [
                sg.Multiline(
                    "",
                    size=(width, height),
                    key=self.LOG_KEY,
                    autoscroll=True,
                    write_only=True,
                    reroute_stdout=reroute_print,
                    reroute_stderr=reroute_print,
                    disabled=True,
                    expand_x=True,
                    expand_y=True,
                )
            ],
            [
                sg.Input(
                    "",
                    key=self.IN_KEY,
                    expand_x=True,
                    do_not_clear=False,
                    enable_events=False,
                ),
                sg.Button(
                    "发送", key=self.SEND_KEY
                ),  # 这里先不管回车，下面用 bind 处理
            ],
        ]

        self.window = sg.Window(
            self.title,
            layout,
            icon=self.icon,
            finalize=True,
            resizable=True,
        )

        self.tray = None
        if self.enable_tray:
            tray_menu = sg.Menu([["", ["显示/隐藏", "退出"]]])
            self.tray = sg.SystemTray(menu=tray_menu, filename=self.icon)

        # 欢迎行
        print(f"[{self.title}] 已启动。输入框回车或点“发送”开始对话。")

        # 关闭拦截：改为最小化到托盘
        if self.minimize_to_tray_on_close:
            self.window.TKroot.protocol("WM_DELETE_WINDOW", self._on_close_clicked)

    # ---------------- 公共 API ----------------
    def run(self):
        """主事件循环（阻塞）"""
        while True:
            event, values = self.window.read(timeout=100)

            # 托盘事件
            if self.tray:
                tray_event = self.tray.read(timeout=0)
                if tray_event == "显示/隐藏":
                    self.toggle_visible()
                elif tray_event == "退出":
                    self.close()
                    break

            if event in (sg.WIN_CLOSED, None):
                if self.minimize_to_tray_on_close and self.tray:
                    # 改为隐藏到托盘
                    self.hide()
                    self._tray_balloon("VoidMate", "已最小化到托盘")
                    continue
                else:
                    self.close()
                    break

            if event == self.SEND_KEY or event == self.IN_KEY:
                # Input 绑定了 Return，触发的是 SEND 或 IN_KEY（取决于 bind_return_key）
                msg = values.get(self.IN_KEY, "").strip()
                if msg:
                    self._append_line(f"你: {msg}")
                    # 后台线程调用回调
                    threading.Thread(
                        target=self._do_send, args=(msg,), daemon=True
                    ).start()
                    self.window[self.IN_KEY].update("")

            elif event == self.LOG_EVT:
                self._append_line(values[self.LOG_EVT])

            elif event == self.REPLY_EVT:
                self._append_line(f"AI: {values[self.REPLY_EVT]}")

    def log(self, text: str):
        """从后台线程安全写日志到界面"""
        self.window.write_event_value(self.LOG_EVT, text)

    def reply(self, text: str):
        """从后台线程把 AI 回复推回界面"""
        self.window.write_event_value(self.REPLY_EVT, text)

    def show(self):
        self.window.un_hide()
        self.window.bring_to_front()

    def hide(self):
        self.window.hide()

    def toggle_visible(self):
        if self.window.TKroot.state() == "withdrawn":
            self.show()
        else:
            self.hide()

    def close(self):
        if self.tray:
            try:
                self.tray.close()
            except Exception:
                pass
        self.window.close()

    # ---------------- 内部实现 ----------------
    def _append_line(self, s: str):
        ml: sg.Multiline = self.window[self.LOG_KEY]
        ml.update(disabled=False)
        ml.print(s)
        ml.update(disabled=True)

    def _do_send(self, msg: str):
        try:
            ret = self.on_send(msg)
        except Exception as e:
            ret = f"[错误] {e}"
        if ret is not None:
            self.reply(ret)

    def _tray_balloon(self, title: str, msg: str, millis: int = 1500):
        if self.tray:
            try:
                self.tray.show_message(title, msg, time=millis)
            except Exception:
                pass

    def _on_close_clicked(self):
        if self.tray and self.minimize_to_tray_on_close:
            self.hide()
            self._tray_balloon("VoidMate", "已最小化到托盘")
        else:
            self.close()


def on_send(msg: str):
    # 这里写你的后端调用逻辑；演示用假延迟
    time.sleep(0.3)
    # 示例：调用 FastAPI
    # r = httpx.post("http://127.0.0.1:8000/chat", json={"text": msg}, timeout=10)
    # return r.json().get("reply", "")
    return f"收到啦：{msg}"


if __name__ == "__main__":
    ui = VoidMateUI(
        title="VoidMate",
        icon=None,  # 可换成 'icon.ico'
        on_send=on_send,
        enable_tray=True,
        minimize_to_tray_on_close=True,
        reroute_print=True,  # print/异常 会直接显示在日志区
    )
    ui.run()
