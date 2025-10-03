# clone.py
from __future__ import annotations
import pathlib
import numpy as np
import inspect
from typing import Any, Optional

__all__ = ["MODEL_NAME", "clone_and_save", "load_latents"]


MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"


def _get(obj: Any, dotted: str) -> Optional[Any]:
    cur = obj
    for part in dotted.split("."):
        if cur is None:
            return None
        cur = getattr(cur, part, None)
    return cur


def _resolve_xtts_model(tts_obj: Any) -> Any:
    """
    在不同版本的 Coqui TTS 里尽力拿到 XTTS 模型对象。
    """
    candidates = [
        _get(tts_obj, "_model"),
        _get(tts_obj, "model"),
        _get(tts_obj, "synthesizer"),
        _get(tts_obj, "synthesizer.tts_model"),
        _get(tts_obj, "synthesizer.model"),
    ]
    syn = _get(tts_obj, "synthesizer")
    if syn is not None:
        tts_models = getattr(syn, "tts_models", None)
        if isinstance(tts_models, (list, tuple)) and tts_models:
            candidates.append(tts_models[0])
        elif isinstance(tts_models, dict) and tts_models:
            candidates.append(next(iter(tts_models.values())))
    expanded = []
    for c in candidates:
        if c is None:
            continue
        expanded.append(c)
        m2 = getattr(c, "model", None)
        if m2 is not None:
            expanded.append(m2)
    for obj in expanded:
        for mname in ("get_conditioning_latents", "get_conditioning", "encode_speaker"):
            if hasattr(obj, mname):
                return obj
    return None


def _call_get_latents(xtts_model: Any, wavs: list[str], language: str | None):
    """
    兼容多版本签名：
    - 新版：get_conditioning_latents(audio_path=..., lang='zh') / (language='zh')
    - 旧版(0.22.0)：get_conditioning_latents(audio_path=..., max_ref_length=..., ...)
    我们只把“该版本支持的参数”塞进去，避免传错触发切片报错。
    """
    last_err = None
    for mname in ("get_conditioning_latents", "get_conditioning", "encode_speaker"):
        fn = getattr(xtts_model, mname, None)
        if fn is None:
            continue
        try:
            sig = inspect.signature(fn)
            params = sig.parameters
            # args/kwargs 组合，尽最大兼容
            if "audio_path" in params:
                args = ()
                kwargs = {"audio_path": wavs}
            else:
                # 老版本把第一个位置参数当成音频列表
                args = (wavs,)
                kwargs: dict = {}

            # 语言参数：只有当函数确实支持时才传，老版本就不传
            if language:
                if "lang" in params:
                    kwargs["lang"] = language
                elif "language" in params:
                    kwargs["language"] = language

            # 常见数值超参（只有存在才传）
            if "gpt_cond_len" in params:
                kwargs["gpt_cond_len"] = 6
            if "gpt_cond_chunk_len" in params:
                kwargs["gpt_cond_chunk_len"] = 6
            if "max_ref_length" in params:
                kwargs["max_ref_length"] = 30
            if "sound_norm_refs" in params:
                kwargs["sound_norm_refs"] = False
            if "load_sr" in params:
                kwargs["load_sr"] = 24000

            res = fn(*args, **kwargs)

            # 解析返回
            if isinstance(res, (tuple, list)) and len(res) >= 2:
                gpt, spk = res[0], res[1]
            elif isinstance(res, dict):
                gpt, spk = res.get("gpt_cond_latent"), res.get("speaker_embedding")
            else:
                last_err = RuntimeError(f"未知返回类型: {type(res)}")
                continue

            if gpt is None or spk is None:
                last_err = RuntimeError("返回缺少 gpt_cond_latent 或 speaker_embedding")
                continue

            return np.array(gpt, dtype=np.float32), np.array(spk, dtype=np.float32)

        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(
        f"未能调用潜变量提取接口（请检查 TTS 版本）。最后错误: {last_err}"
    )


# ---------- 你要的两个公开函数 ----------
def clone_and_save(
    name: str,
    wav_paths: list[str] | str,
    language: str = "zh",  # 对外保留 language；旧版会自动忽略
    save_dir: str = "tts/voice_profiles",
    model_name: str | None = MODEL_NAME,
) -> str:
    """
    the function tath clone a voice in the upploadde wav file
    and return the saved path

    Args:
        name: the voive profile name, it can be the owner of that voice
        wav_paths: str / list[str] the source wav file paths
        language: the lang used in the soruce wav file
        save_dir: the saved dir for voice profile, it used to be in the 'projectroot/tts/voice_profiles' only if you use the default value and the script is run in the project root dir
        model_name: the pretrained model name

    Returns:
        [TODO:return]

    Raises:
        ValueError: [TODO:throw]
        FileNotFoundError: [TODO:throw]
        RuntimeError: [TODO:throw]
    """
    """
    提取 XTTS 说话人潜变量并保存为 .npz
    返回保存路径，如 voice_profiles/{name}/latents.npz
    """
    if model_name is None:
        model_name = MODEL_NAME

    if isinstance(wav_paths, str):
        wav_paths = [wav_paths]
    if not wav_paths:
        raise ValueError("至少提供一段参考音频路径")

    wavs = []
    for p in wav_paths:
        q = pathlib.Path(p).expanduser().resolve()
        if not q.exists():
            raise FileNotFoundError(str(q))
        wavs.append(str(q))

    from TTS.api import TTS

    tts = TTS(model_name=model_name, progress_bar=True)

    xtts_model = _resolve_xtts_model(tts)
    if xtts_model is None:
        raise RuntimeError(
            "XTTS model not found, please check the tts package version, notes that the versio author used is 0.22.0"
        )

    gpt, spk = _call_get_latents(xtts_model, wavs, language=language)

    out_dir = pathlib.Path(save_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "latents.npz"
    np.savez_compressed(out_path, gpt_cond_latent=gpt, speaker_embedding=spk)
    return str(out_path)


def load_latents(name: str, save_dir: str = "tts/voice_profiles"):
    """

    Args:
        name: the profile name
        save_dir: the dir taht contains the voice profiles
    Returns:
        tuple("gpt_cond_latent", "speaker_embedding") of np.ndarray

    Raises:
        FileNotFoundError: if the profile does not exists
    """
    p = pathlib.Path(save_dir) / name / "latents.npz"
    if not p.exists():
        raise FileNotFoundError(f"profile file not extist: {p}")
    d = np.load(p)
    return d["gpt_cond_latent"], d["speaker_embedding"]


# 自测
if __name__ == "__main__":
    path = clone_and_save(
        name="test",
        wav_paths=["tts/cloned_samples/output_merged.wav"],
        language="jp",  # 新版会用上，TTS 0.22.0 会被自动忽略
    )
    print(f"saved: {path}")
    gpt_lat, spk_lat = load_latents("test")
    print(gpt_lat.shape, spk_lat.shape)
