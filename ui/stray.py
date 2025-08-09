import pystray
import subprocess
from PIL import Image


def on_quit_clicked(icon):
    icon.stop()


def on_icon_clicked(icon):
    print("clicked")
    subprocess.Popen("start cmd", shell=True)


image = Image.open("icon.ico")
menu = (
    pystray.MenuItem(text="退出", action=on_quit_clicked),
    pystray.MenuItem(text="打开命令行", action=on_icon_clicked),
)
icon = pystray.Icon("name", image, "托盘名称", menu)

icon.run()
