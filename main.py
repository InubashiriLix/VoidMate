import time
import os
import pathlib
from pathlib import Path

from queue import Queue, Empty, Full
from threading import Thread

from src.ai.interfaces.Interact import GeminiCompanion, Callbacks, log, CompanionConfigs
from src.tts.speaker import Speaker


def _detect_profile_file(prompt: str, cwd: Path, endwith: str) -> list[Path]:
    print(prompt)
    rtn_list = [
        f
        for f in (cwd / "Profile").rglob("*")
        if f.is_file() and f.name.endswith(endwith)
    ]
    print("done")
    return rtn_list


def _select_file_from_list(
    file_list: list[Path], prompt: str, default_value: str, input_func=input
) -> str:
    for i, fp in enumerate(file_list):
        print(f"{i}: {fp.name}")
    sele_idx = input_func(prompt)
    if sele_idx.isdigit() and int(sele_idx) in range(len(file_list)):
        return str(file_list[int(sele_idx)])
    else:
        return default_value


def setup() -> CompanionConfigs:
    print("======================== setup =================================")
    current_dir: Path = pathlib.Path(__file__).parent
    print("current_dir: ", current_dir)

    # ensure Profile directory exists
    pathlib.Path(current_dir / "Profile").mkdir(parents=True, exist_ok=True)

    conf = CompanionConfigs(current_dir)

    ini_f_list: list[Path] = _detect_profile_file(
        "detecting ini files: ", current_dir, ".ini"
    )
    db_f_list = _detect_profile_file("detecting db files: ", current_dir, ".db")
    sche_json_f_list = _detect_profile_file(
        "detecting schedule json files: ", current_dir, ".json"
    )

    is_skip_sele_config = input("Skip select config files? [Y/n]\n>>")
    if not (is_skip_sele_config.lower() in ["y", "yes", ""]):
        print("===================== SELECT CONFIG ============================")
        # setting up ini config
        conf.DEFAULT_ROLE_INI = _select_file_from_list(
            ini_f_list,
            "enter index to select ini file. Enter only to use default:\n>>",
            conf.DEFAULT_ROLE_INI,
        )
        print("using ini file: ", conf.DEFAULT_ROLE_INI, "\n")

        # setting up database
        conf.DEFAULT_DB = _select_file_from_list(
            db_f_list,
            "enter index to select database file. Enter only to use default:\n>>",
            conf.DEFAULT_DB,
        )
        print("using database file: ", conf.DEFAULT_DB, "\n")

        conf.DEFAULT_SCHEDULE = _select_file_from_list(
            sche_json_f_list,
            "enter index to select schedule json file. Enter only to use default:\n>>",
            conf.DEFAULT_SCHEDULE,
        )
        print("using schedule json file: ", conf.DEFAULT_SCHEDULE, "\n")
    else:
        print("using ini file: ", conf.DEFAULT_ROLE_INI)
        print("using database file: ", conf.DEFAULT_DB)
        print("using schedule json file: ", conf.DEFAULT_SCHEDULE)

    print("======================= setup done =============================")

    return conf


def setup_speaker() -> Speaker:
    # TODO: add systematic profile ini config, and add the speaker model name into it
    # 1. add the speaker model name
    # 2. add output language into it
    # 3. add the source wav and its lang into profile
    return Speaker(
        profile_name="test",
        pretrained_model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        wav_path=["Profile/cloned_samples/output_merged.wav"],
        source_wav_language="jp",
        output_lang="en-us",
        out_dir="RuntimeCache/tts_output",
    )


def setup_campanion(confs: CompanionConfigs) -> GeminiCompanion:
    cb = Callbacks(on_message=lambda m: log(f"AI reply：{m}"))
    bot = GeminiCompanion(
        callbacks=cb, confs=confs, api_key=os.getenv("GOOGLE_API_KEY")
    )
    return bot


if __name__ == "__main__":
    # === runtime ===
    tts_queue: Queue = Queue(maxsize=256)
    SENTINEL = object()
    cwd = Path(__file__).parent

    try:
        # init companion & speaker
        confs = setup()
        bot = setup_campanion(confs)
        spk = setup_speaker()

        # quick self-test
        spk.say(
            "Your best grab a blade to slap upon your belt. You gonna need it.", cwd
        )
        print(bot.chat("I'm back again")["text"]["en"])

        # workers
        def producer():
            try:
                while True:
                    enter = input("enter your next msg (type 'exit' to quit): ")
                    if enter.strip().lower() in {"exit", "quit", "/q", "/quit"}:
                        tts_queue.put(SENTINEL)
                        break

                    reply = bot.chat(enter)
                    text = reply.get("text", {}).get("en", "")
                    text = (text or "").strip()
                    if not text:
                        log("Empty reply text; skip.")
                        continue

                    lines = text.splitlines()
                    tts_queue.put(lines)

            except KeyboardInterrupt:
                # put the SENTINEL on quit
                tts_queue.put(SENTINEL)
            except Exception as e:
                log(f"Producer error: {e}")
                tts_queue.put(SENTINEL)

        def consumer():
            try:
                while True:
                    try:
                        item = tts_queue.get(timeout=2)
                    except Empty:
                        continue

                    if item is SENTINEL:
                        break

                    # 兼容 list[str] 或 str
                    lines = item if isinstance(item, list) else [str(item)]
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        spk.say(line, cwd, verbose=False)

            except Exception as e:
                log(f"TTS consumer error: {e}")

        # START
        prod_t = Thread(target=producer, name="producer", daemon=True)
        cons_t = Thread(target=consumer, name="consumer", daemon=True)
        prod_t.start()
        cons_t.start()

        # QUIT
        prod_t.join()  # quit / exit input
        cons_t.join()  # waiting for notification of SENTINEL

    except KeyboardInterrupt:
        try:
            tts_queue.put(SENTINEL)
        except Exception:
            pass
    finally:
        try:
            bot.close()
        except Exception:
            pass
