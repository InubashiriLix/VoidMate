from typing import Iterable, List, Union
from pathlib import Path

from google import genai
from google.genai import types


_pic_analysis_origin_prompt = """
请执行三件事：

 **提取代码**  
   - 检测截图里所有编辑器 / 终端 / IDE 区域。  
   - 将完整代码原样放进 Markdown 代码块 ```lang\n...\n```，保留缩进、空行与语法高亮语言（若能判断）。  
   - 多个代码片段请用多个独立代码块依次给出，别合并。

 **提取其它可读文字**  
   - 对非代码区域，逐行 OCR，保留原有换行。  
   - 只要能帮我看出在干嘛的文字就行（窗口标题、按钮、歌词、视频标题、文档标题、游戏 HUD 等均可）。  
   - 不要重复的 UI 杂项（如导航栏同一词反复出现）。

 **推断主要活动 & 简述细节**  
   - 根据截图推断我当时主要在做什么：编程 / 看视频 / 玩游戏 / 读文档 / 听音乐 / 其他。  
   - 用一句话描述关键信息，例如：  
     - `编程：正在 VSCode 中编写 Python，文件 demo.py`  
     - `看视频：B 站正在播放 <进击的巨人 S03E12>`  
     - `玩游戏：Genshin Impact，角色在须弥城`  
   - 若实在判断不了，写 `活动：无法确定`。

 **输出格式严格如下**（除 Markdown 代码块外，不要有任何多余文字或注释）：

代码:

<lang>
复制
编辑
...代码1...
<lang>
复制
编辑
...代码2...
文字:
<行1>
<行2>
...

活动:
<一句话描述>

markdown
复制
编辑

> 若截图中文字总计少于 10 行，可省略“文字”段，只返回“活动”段和可能存在的代码块。
"""


def pic_analaysis(
    path: Path,
    model: str = "gemini-2.5-flash",
    prompt: str = _pic_analysis_origin_prompt,
):
    """
    if the main content in the pic is the text, then extract the text, else give both text and description
    args:
        path: Path to the image file, the existens should be checked before calling this function.
        model: the model name
        prompt: the prompt to use for text extraction
    """
    client = genai.Client()

    # 读图片 → bytes
    with path.open("rb") as f:
        img_bytes = f.read()

    # 把图片和提示一起丢进去
    resp = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
            prompt,
        ],
    )
    if resp is None or not resp.text:
        raise ValueError("No text extracted from the image.")

    return resp.text.strip()


__all__ = ["pic_multi_analysis"]

MAX_INLINE_BYTES = 20 * 1024 * 1024  # 20 MB


def _to_part(client: genai.Client, p: Path) -> types.Part:
    """
    将 Path 转成 Gemini Part：小文件用 bytes，大文件先上传再引用。
    修正图片 MIME 类型。
    """
    mime_type = p.suffix.lstrip(".").lower()  # 获取文件扩展名
    if mime_type not in ["jpeg", "png"]:
        raise ValueError(f"Unsupported image type: {mime_type}")

    if p.stat().st_size <= MAX_INLINE_BYTES:
        with p.open("rb") as f:
            return types.Part.from_bytes(data=f.read(), mime_type=f"image/{mime_type}")
    else:
        # 大文件用 files.upload
        uploaded = client.files.upload(file=str(p))
        return uploaded  # type: ignore[return-value]


def pic_multi_analysis(
    paths: Union[Path, Iterable[Path]],
    model: str = "gemini-2.5-flash",
    prompt: str = _pic_analysis_origin_prompt,
) -> str | None:
    """
    对一张或多张图片做内容+OCR 综合分析。

    Parameters
    ----------
    paths : Path | Iterable[Path]
        单张或多张图片路径；请自行确保文件存在。
    model : str
        Gemini 模型名称，默认 "gemini-2.5-flash"。
    prompt : str
        提示词，可覆盖默认多模态 OCR & 活动识别提示。

    Returns
    -------
    str | None
        模型返回的文本；空字符串或 None 代表调用失败或无文本。
    """
    client = genai.Client()

    # --- 1. 统一成 list[Path] ---
    if isinstance(paths, Path):
        path_list: List[Path] = [paths]
    else:
        path_list = list(paths)

    if not path_list:
        raise ValueError("`paths` 不能为空")

    # --- 2. 构造 contents ---
    contents: List[Union[types.Part, str]] = [prompt]
    for p in path_list:
        contents.append(_to_part(client, p))

    # --- 3. 调用模型 ---
    response = client.models.generate_content(
        model=model,
        contents=contents,  # type: ignore
    )

    return (response.text or "").strip() if response else None


if __name__ == "__main__":
    print(
        pic_multi_analysis(
            [
                Path("C:/Users/lixin/Desktop/Inubashiri/images/music.png"),
                Path("C:/Users/lixin/Desktop/Inubashiri/images/programming_kalman.png"),
            ]
        )
    )
