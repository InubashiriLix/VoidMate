import sys
import subprocess
import ctypes

import pystray
from pystray import Menu, MenuItem
from PIL import Image

import win32con
import win32gui

APP_TITLE = "VoidMate"

# 正确拼写
GetConsoleWindow = ctypes.windll.kernel32.GetConsoleWindow


def _hwnd():
    h = GetConsoleWindow()
    return h if h else None


def console_visible():
    h = _hwnd()
    return bool(h and win32gui.IsWindowVisible(h))


def show_console():
    h = _hwnd()
    if h:
        win32gui.ShowWindow(h, win32con.SW_SHOW)
        try:
            win32gui.SetForegroundWindow(h)
        except win32gui.error:
            pass


def hide_console():
    h = _hwnd()
    if h:
        win32gui.ShowWindow(h, win32con.SW_HIDE)


def on_quit(icon, item):
    icon.visible = False
    icon.stop()


def on_open_cmd(icon, item):
    # 打开新的经典 CMD 窗口
    subprocess.Popen("start cmd", shell=True)


def on_toggle_console(icon, item):
    if console_visible():
        hide_console()
    else:
        show_console()
    # 切换后刷新菜单文本
    icon.update_menu()


# ✅ 动态菜单文本回调需要接收一个参数（MenuItem）
def label_toggle(item):
    return "隐藏控制台" if console_visible() else "显示控制台"


def main():
    if "--minimized" in sys.argv and _hwnd():
        hide_console()

    image = Image.open("icon.ico")  # 建议用包含 16/32/48 多尺寸的 .ico

    menu = Menu(
        MenuItem(label_toggle, on_toggle_console, default=True),
        MenuItem("打开命令行", on_open_cmd),
        Menu.SEPARATOR,
        MenuItem("退出", on_quit),
    )

    icon = pystray.Icon("voidmate_tray", image, APP_TITLE, menu)
    icon.run()


if __name__ == "__main__":
    # 如果在 Windows Terminal 运行，可能拿不到控制台句柄；此时“显示/隐藏控制台”会失效，但“打开命令行”可用
    if not _hwnd():
        sys.stderr.write(
            "[提示] 未检测到经典控制台窗口句柄（可能在 Windows Terminal 中运行）。\n"
            "      显示/隐藏当前控制台可能不可用，请从 cmd.exe/经典 PowerShell 启动以获得完整功能。\n"
        )
    main()
