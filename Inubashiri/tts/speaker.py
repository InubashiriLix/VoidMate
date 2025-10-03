from typing import Any
import os
import json
import time
import numpy as np
import torch
import pathlib
from pathlib import Path

import winsound
import soundfile as sf

from TTS.api import TTS
from TTS.utils.manage import ModelManager
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# WARNING: these may work in the torch > 2.6 versions, not tested on torch >= 2.5
# it is not recommanded to use torch > 2.6, and you'd better follow the requirements in the readme file
try:
    from torch.serialization import add_safe_globals
    from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
    from TTS.config.shared_configs import BaseDatasetConfig

    add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig])
except Exception:
    pass

from emotion_presets import EMOTION_PRESETS

from clone import *  # noqa


import soundfile as sf


def _nowstr():
    return time.strftime("%Y%m%d-%H%M%S")


def _to_tensor(x, device, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, dtype=dtype, device=device)


def _norm_lang(lang: str | None) -> str:
    if not lang:
        return "zh-cn"
    lang = lang.lower()
    if lang in ("jp", "jpn", "ja-jp"):
        return "ja"
    if lang in ("zh", "cn", "zh_cn", "zhcn"):
        return "zh-cn"
    return lang


class Speaker:
    presets = EMOTION_PRESETS

    profile_name: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    pretrained_model_name: str = MODEL_NAME  # noqa
    lang: str = "zh-cn"

    inference_queue: list[dict] = []
    audio_play_queue: list[Any] = []

    sample_rate: int = 24000

    def __init__(
        self,
        profile_name: str,
        output_lang: str | None = None,
        pretrained_model_name: str | None = None,
        wav_path: list[str] | str | None = None,
        source_wav_language: str | None = None,
        out_dir: str = "tts/tts_out",
    ):
        print(f"[speaker] [init]: initing speaker: {profile_name}")

        os.environ.setdefault("COQUI_TOS_AGREED", "1")

        print(f"[speaker] [init]: source_wav_language: {source_wav_language}")
        self.lang = _norm_lang(output_lang) or self.lang
        print(f"[speaker] [init]: output lang: {self.lang}")

        self.profile_name = profile_name
        self.pretrained_model_name = pretrained_model_name or MODEL_NAME  # noqa
        self.gpt_cond_latenet: np.ndarray | torch.Tensor
        self.speaker_embedding: np.ndarray | torch.Tensor

        self.out_dir = Path(out_dir)
        if not Path(self.out_dir).exists():
            Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        print(f"[speaker] [init]: output dir: {Path(out_dir).resolve()}")

        try:
            self._load_voice_profile()
            print("[speaker] [init]: load voice done")
        except FileNotFoundError as fe:
            print(f"error: {fe}")
            print("[speaker]: profile not found, clone voice first")

            if profile_name and wav_path and source_wav_language:
                clone_and_save(  # noqa
                    name=self.profile_name,
                    wav_paths=wav_path,
                    language=source_wav_language,
                    model_name=pretrained_model_name,
                )

                try:
                    self._load_voice_profile()
                    print("[speaker] [init]: load voice done")
                except FileNotFoundError as fe2:
                    print("[load]: profile not found, check your profile name settings")
                    print(fe)
                    print(fe2)
            else:
                raise RuntimeError(
                    "Arguement not enough to clone voice, check the init method of Speaker class"
                )

        print(f"[speaker] [init]: using device: {self.device}")

        print("[speaker] [init]: loading xtts (low-level) for latent inference")
        self.xtts = self._load_xtts_model(self.pretrained_model_name)

        # NOTE: the api is the backup method, for common cases, use self.xtts + latenet
        print("[speaker] [init]: loading TTS api (fallback)")
        self.tts_api = TTS(self.pretrained_model_name).to(self.device)

        print("[speaker] [init]: speaker init done")

    def _load_voice_profile(self):
        self.gpt_cond_latenet, self.speaker_embedding = load_latents(  # noqa
            self.profile_name
        )
        self.gpt_cond_latenet = _to_tensor(self.gpt_cond_latenet, self.device)
        self.speaker_embedding = _to_tensor(self.speaker_embedding, self.device)

    def _load_xtts_model(self, model_name: str) -> Xtts:
        # 让 ModelManager 负责下载/定位模型
        from TTS.utils.manage import ModelManager
        from pathlib import Path

        manager = ModelManager()
        model_path, config_path, _ = manager.download_model(model_name)

        # —— 关键：如果 config_path 为空或不存在，自动在 model_path 附近搜 config.json ——
        def _auto_find_config(mp: str | None, cp: str | None) -> str:
            cands = []
            if cp:
                cands.append(Path(cp))
            if mp:
                p = Path(mp)
                if p.is_dir():
                    cands.append(p / "config.json")  # 常见位置
                    # 兜底：递归找（只取第一个命中）
                    try:
                        for sub in p.rglob("config.json"):
                            cands.append(sub)
                            break
                    except Exception:
                        pass
                else:
                    cands.append(p.parent / "config.json")
            for c in cands:
                if c and c.exists():
                    return str(c)
            raise FileNotFoundError(
                f"config.json not found near: model_path={mp!r}, config_path={cp!r}"
            )

        config_path = _auto_find_config(model_path, config_path)

        # 正常加载 Xtts（低层接口，配合你的潜变量）
        cfg = XttsConfig()
        cfg.load_json(config_path)
        model = Xtts.init_from_config(cfg)

        ckpt_dir = str(Path(config_path).parent)
        model.load_checkpoint(cfg, checkpoint_dir=ckpt_dir, eval=True)
        if self.device.startswith("cuda"):
            model.cuda()
        return model

    def inference(self, sentence: dict):
        """
        sentence 结构建议：
        {
          "text": "...",                 # 必填
          "emotion": "soft"/"happy"/..., # 选填，不给则默认 neutral
          "override": {...},             # 选填，覆盖预设的解码参数
          "pad_ms": 120,                 # 选填，播放拼接时前后留白（只影响 play()）
        }
        """
        assert "text" in sentence, "sentence must contain 'text' key"

        text = sentence["text"]
        emotion = emotion = sentence.get("emotion", "neutral")
        override = sentence.get("override", {}) or {}
        pad_ms = int(sentence.get("pad_ms", 120))

        preset = self.presets.get(emotion, self.presets["neutral"])
        params = {**preset["xtts_params"], **override}

        out = self.xtts.inference(
            text=text,
            language=self.lang,
            gpt_cond_latent=self.gpt_cond_latenet,
            speaker_embedding=self.speaker_embedding,
            **params,
        )

        wav = np.asarray(out["wav"], dtype=np.float32).flatten()
        dur = round(len(wav) / self.sample_rate, 3)

        item = {
            "wav": wav,
            "meta": {
                "text": text,
                "emotion": emotion,
                "va": preset["va"],
                "params": params,
                "duration_s": dur,
                "pad_ms": pad_ms,
            },
        }
        self.audio_play_queue.append(item)
        return item

    def play(
        self,
        out_wav: str | None = None,
        out_json: str | None = None,
        clear_after: bool = True,
    ):
        """
        把 audio_play_queue 里的段落拼接导出：
        - out_wav: 输出 WAV 文件名（默认 tts_out/{profile}-{timestamp}.wav）
        - out_json: 同名 JSON（时间线与参数）
        - clear_after: 导出后是否清空播放队列
        """
        if not self.audio_play_queue:
            print("[speaker] [play]: queue is empty, nothing to output.")
            return None, None

        pads = []
        audio = []
        timeline = []
        cursor = 0.0
        for seg in self.audio_play_queue:
            wav = seg["wav"]
            meta = dict(seg["meta"])  # copy

            # concat pad_ms
            pad_ms = int(meta.get("pad_ms", 120))
            pad = np.zeros(int(self.sample_rate * pad_ms / 1000), dtype=np.float32)

            # timeline
            meta["start_s"] = round(cursor, 3)
            cursor += meta["duration_s"] + pad_ms / 1000.0
            timeline.append(meta)

            audio.append(wav)
            pads.append(pad)

        audio_concat = []
        for w, p in zip(audio, pads):
            audio_concat.append(w)
            audio_concat.append(p)
        audio_concat = np.concatenate(audio_concat, dtype=np.float32)

        stamp = _nowstr()
        base = f"{self.profile_name}-{stamp}"
        out_wav = str(self.out_dir / (out_wav or (base + ".wav")))
        out_json = str(self.out_dir / (out_json or (base + ".json")))

        sf.write(out_wav, audio_concat, self.sample_rate, subtype="PCM_16")

        info = {
            "profile": self.profile_name,
            "language": self.lang,
            "sample_rate": self.sample_rate,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "segments": timeline,
            "total_duration_s": round(len(audio_concat) / self.sample_rate, 3),
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

        print(f"[speaker] [play]: wrote {out_wav}")
        print(f"[speaker] [play]: wrote {out_json}")

        if clear_after:
            self.audio_play_queue.clear()

        return out_wav, out_json


if __name__ == "__main__":
    import time, shutil

    print("=== SMOKE TEST: XTTS low-level + saved latents ===")

    # 可按需改：参考音色素材路径、语言（jp->ja 映射已在类里处理）
    spk = Speaker(
        profile_name="test",
        pretrained_model_name=MODEL_NAME,  # noqa
        wav_path=["tts/cloned_samples/output_merged.wav"],
        source_wav_language="jp",
        output_lang="en-us",
        # output_lang="zh-cn",
    )

    t0 = time.time()
    # spk.inference(
    #     {"text": "おはようございます。テストを始めます。", "emotion": "neutral"}
    # )
    # spk.inference(
    #     {"text": "おはようございます。テストを始めます。", "emotion": "whisper"}
    # )
    # spk.inference(
    #     {"text": "おはようございます。テストを始めます。", "emotion": "happy"}
    # )
    # spk.inference(
    #     {"text": "おはようございます。テストを始めます。", "emotion": "comforting"}
    # )
    # spk.inference(
    #     {"text": "おはようございます。テストを始めます。", "emotion": "lullaby"}
    # )
    # spk.inference(
    #     {"text": "おはようございます。テストを始めます。", "emotion": "flirty"}
    # )

    # spk.inference({"text": "测试开始", "emotion": "neutral", "pad_ms": 150})
    # spk.inference(
    #     {"text": "这里是犬走椛，开始测试", "emotion": "neutral", "pad_ms": 150}
    # )
    # spk.inference({"text": "杀了他们", "emotion": "angry", "pad_ms": 150})

    # spk.inference({"text": "测试开始", "emotion": "neutral", "pad_ms": 150})
    # spk.inference(
    #     {
    #         "text": "盖伦,发发！盖伦,发发！盖伦,发发！盖伦,发发！盖伦,发发！盖伦,发发！盖伦,发发！盖伦,发发！盖伦,发发！盖伦,发发！盖伦,发发！",
    #         "emotion": "neutral",
    #         "pad_ms": 150,
    #     }
    # )

    out_wav, out_json = spk.play(out_wav="smoketest.wav")  # -> tts_out/smoketest.wav
    dt = time.time() - t0

    winsound.PlaySound(
        "E:/0-19_VoidMate/Inubashiri/tts/tts_out/smoketest.wav", winsound.SND_FILENAME
    )

    while True:
        test_input: str = input()
        test_emotion: str = input()

        if test_emotion not in spk.presets.keys():
            test_emotion = "neutral"

        print(spk.presets.keys())

        if test_input.lower() == "exit":
            print("[SMOKE] exiting...")
            break

        spk.inference({"text": test_input, "emotion": test_emotion, "pad_ms": 150})
        spk.play(out_wav="smoketest.wav")  # -> tts_out/smoketest.wav
        winsound.PlaySound(
            "E:/0-19_VoidMate/Inubashiri/tts/tts_out/smoketest.wav",
            winsound.SND_FILENAME,
        )

    # 打印结果与简单校验
    print(f"[SMOKE] done in {dt:.2f}s")
    if out_wav and os.path.exists(out_wav):
        print(f"[SMOKE] WAV:  {out_wav}  ({shutil.get_terminal_size().columns} cols)")
    if out_json and os.path.exists(out_json):
        with open(out_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
        print(f"[SMOKE] JSON: {out_json}")
        print(
            f"[SMOKE] segments={len(meta['segments'])}, total_duration_s={meta['total_duration_s']}"
        )
