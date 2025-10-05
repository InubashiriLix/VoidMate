import time
import os
import pathlib
from pathlib import Path

from queue import Queue, Empty, Full
from threading import Thread

from src.ai.interfaces.Interact import GeminiCompanion, Callbacks, log, CompanionConfigs
from src.tts.speaker import Speaker


def _detect_profile_file(prompt: str, root_proj_path: Path, endwith: str) -> list[Path]:
    """detect the profile files in the profile directory

    Args:
        prompt: the input prompt to show
        root_proj_path: the root project path
        endwith: the extension to filter the files

    Returns:
        return the list of detected files, and it might be empty, check pls.
    """
    print(prompt)
    rtn_list = [
        f
        for f in (root_proj_path / "Profile").rglob("*")
        if f.is_file() and f.name.endswith(endwith)
    ]
    print("done")
    return rtn_list


def _select_file_from_list(
    file_list: list[Path], prompt: str, default_value: str, input_func=input
) -> str:
    """the user-interactive file selector from the given file list

    Args:
        input_func the input function, default to built-in input(), can be replaced for test
        file_list: the file list to seelct from
        prompt: the input prompt to show
        default_value: the default value to return if the input is invalid

    Returns:
        the selected file path as str, or the default value if input is invalid
    """
    for i, fp in enumerate(file_list):
        print(f"{i}: {fp.name}")
    sele_idx = input_func(prompt)
    if sele_idx.isdigit() and int(sele_idx) in range(len(file_list)):
        return str(file_list[int(sele_idx)])
    else:
        return default_value


def setup() -> CompanionConfigs:
    """setup the companion configs interactively

    Returns:
        return the CompanionConfigs object
    """
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
    """setup the TTS speaker

    Returns:
        the TTS Speaker object
    """
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
    """setup the Gemini companion bot

    Args:
        confs: CompanionConfig object, which should be initialized by setup() function

    Returns:
        companion bot object
    """
    cb = Callbacks()
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
        spk.say("Welcome back.", cwd)

        # workers
        def producer():
            """the producer thread, use the Gemini bot to chat and push the reply the the tts queue"""
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
            """pop the tts queue and use the speaker to say the text"""
            try:
                while True:
                    try:
                        item = tts_queue.get(timeout=2)
                    except Empty:
                        continue

                    if item is SENTINEL:
                        break

                    # support list[str] or str
                    lines = item if isinstance(item, list) else [str(item)]
                    spk.say_batch(words=lines, root_abs_path=cwd, verbose=False)

                    # for line in lines:
                    #     line = line.strip()
                    #     if not line:
                    #         continue
                    #     spk.say(line, cwd, verbose=False)

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
