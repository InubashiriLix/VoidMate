EMOTION_PRESETS = {
    "neutral": {
        "va": {"valence": 0.0, "arousal": 0.2, "dominance": 0.5},
        "xtts_params": {
            "temperature": 0.65,
            "top_p": 0.7,
            "top_k": 30,
            "length_penalty": 1.0,
            "repetition_penalty": 2.0,
            "speed": 1.0,
        },
    },
    "happy": {
        "va": {"valence": 0.8, "arousal": 0.7, "dominance": 0.6},
        "xtts_params": {
            "temperature": 0.8,
            "top_p": 0.85,
            "top_k": 50,
            "length_penalty": 1.1,
            "repetition_penalty": 2.0,
            "speed": 1.05,
        },
    },
    "excited": {
        "va": {"valence": 0.9, "arousal": 0.9, "dominance": 0.7},
        "xtts_params": {
            "temperature": 0.95,
            "top_p": 0.9,
            "top_k": 80,
            "length_penalty": 1.2,
            "repetition_penalty": 2.2,
            "speed": 1.1,
        },
    },
    "sad": {
        "va": {"valence": -0.6, "arousal": 0.25, "dominance": 0.4},
        "xtts_params": {
            "temperature": 0.6,
            "top_p": 0.6,
            "top_k": 20,
            "length_penalty": 0.95,
            "repetition_penalty": 2.0,
            "speed": 0.95,
        },
    },
    "angry": {
        "va": {"valence": -0.7, "arousal": 0.85, "dominance": 0.8},
        "xtts_params": {
            "temperature": 0.9,
            "top_p": 0.85,
            "top_k": 60,
            "length_penalty": 1.15,
            "repetition_penalty": 2.3,
            "speed": 1.05,
        },
    },
    "soft": {  # 轻声/温柔
        "va": {"valence": 0.4, "arousal": 0.15, "dominance": 0.4},
        "xtts_params": {
            "temperature": 0.55,
            "top_p": 0.6,
            "top_k": 20,
            "length_penalty": 0.95,
            "repetition_penalty": 2.0,
            "speed": 0.92,
        },
    },
    # —— 轻声、亲密、安抚类 ——
    "whisper": {  # 低声耳语
        "va": {"valence": 0.3, "arousal": 0.10, "dominance": 0.35},
        "xtts_params": {
            "temperature": 0.50,
            "top_p": 0.55,
            "top_k": 20,
            "length_penalty": 0.95,
            "repetition_penalty": 2.0,
            "speed": 0.90,
        },
    },
    "comforting": {  # 安抚、抚慰
        "va": {"valence": 0.5, "arousal": 0.20, "dominance": 0.45},
        "xtts_params": {
            "temperature": 0.60,
            "top_p": 0.65,
            "top_k": 30,
            "length_penalty": 1.00,
            "repetition_penalty": 2.0,
            "speed": 0.96,
        },
    },
    "reassuring": {  # 让人安心
        "va": {"valence": 0.55, "arousal": 0.25, "dominance": 0.55},
        "xtts_params": {
            "temperature": 0.60,
            "top_p": 0.65,
            "top_k": 30,
            "length_penalty": 1.02,
            "repetition_penalty": 2.0,
            "speed": 0.98,
        },
    },
    "lullaby": {  # 哄睡、催眠
        "va": {"valence": 0.4, "arousal": 0.12, "dominance": 0.35},
        "xtts_params": {
            "temperature": 0.50,
            "top_p": 0.55,
            "top_k": 20,
            "length_penalty": 0.95,
            "repetition_penalty": 2.0,
            "speed": 0.88,
        },
    },
    # —— 轻松、可爱、调侃类 ——
    "playful": {  # 俏皮
        "va": {"valence": 0.7, "arousal": 0.55, "dominance": 0.55},
        "xtts_params": {
            "temperature": 0.80,
            "top_p": 0.80,
            "top_k": 50,
            "length_penalty": 1.05,
            "repetition_penalty": 2.0,
            "speed": 1.04,
        },
    },
    "teasing": {  # 打趣、调侃
        "va": {"valence": 0.5, "arousal": 0.60, "dominance": 0.60},
        "xtts_params": {
            "temperature": 0.85,
            "top_p": 0.80,
            "top_k": 50,
            "length_penalty": 1.05,
            "repetition_penalty": 2.1,
            "speed": 1.03,
        },
    },
    "flirty": {  # 暧昧、撒娇
        "va": {"valence": 0.7, "arousal": 0.45, "dominance": 0.50},
        "xtts_params": {
            "temperature": 0.75,
            "top_p": 0.80,
            "top_k": 40,
            "length_penalty": 1.00,
            "repetition_penalty": 2.0,
            "speed": 0.98,
        },
    },
    "cute": {  # 可爱、卖萌
        "va": {"valence": 0.75, "arousal": 0.60, "dominance": 0.45},
        "xtts_params": {
            "temperature": 0.85,
            "top_p": 0.85,
            "top_k": 60,
            "length_penalty": 1.05,
            "repetition_penalty": 2.1,
            "speed": 1.06,
        },
    },
    # —— 鼓励、激励、教练类 ——
    "encouraging": {  # 鼓励
        "va": {"valence": 0.65, "arousal": 0.55, "dominance": 0.60},
        "xtts_params": {
            "temperature": 0.75,
            "top_p": 0.78,
            "top_k": 50,
            "length_penalty": 1.06,
            "repetition_penalty": 2.0,
            "speed": 1.03,
        },
    },
    "motivational": {  # 打鸡血
        "va": {"valence": 0.70, "arousal": 0.80, "dominance": 0.75},
        "xtts_params": {
            "temperature": 0.90,
            "top_p": 0.88,
            "top_k": 70,
            "length_penalty": 1.10,
            "repetition_penalty": 2.2,
            "speed": 1.08,
        },
    },
    # —— 严肃、指令、权威类 ——
    "serious": {  # 严肃但不凶
        "va": {"valence": 0.1, "arousal": 0.40, "dominance": 0.70},
        "xtts_params": {
            "temperature": 0.65,
            "top_p": 0.65,
            "top_k": 40,
            "length_penalty": 1.05,
            "repetition_penalty": 2.2,
            "speed": 0.99,
        },
    },
    "stern": {  # 严厉训诫
        "va": {"valence": -0.2, "arousal": 0.60, "dominance": 0.85},
        "xtts_params": {
            "temperature": 0.70,
            "top_p": 0.65,
            "top_k": 40,
            "length_penalty": 1.08,
            "repetition_penalty": 2.3,
            "speed": 1.00,
        },
    },
    "authoritative": {  # 权威、果断
        "va": {"valence": 0.0, "arousal": 0.50, "dominance": 0.90},
        "xtts_params": {
            "temperature": 0.68,
            "top_p": 0.68,
            "top_k": 50,
            "length_penalty": 1.08,
            "repetition_penalty": 2.3,
            "speed": 1.02,
        },
    },
    # —— 讲解、播报、叙述类 ——
    "teacher": {  # 老师式讲解
        "va": {"valence": 0.3, "arousal": 0.45, "dominance": 0.65},
        "xtts_params": {
            "temperature": 0.70,
            "top_p": 0.70,
            "top_k": 40,
            "length_penalty": 1.06,
            "repetition_penalty": 2.0,
            "speed": 1.00,
        },
    },
    "newsreader": {  # 新闻播报
        "va": {"valence": 0.0, "arousal": 0.35, "dominance": 0.70},
        "xtts_params": {
            "temperature": 0.60,
            "top_p": 0.60,
            "top_k": 30,
            "length_penalty": 1.10,
            "repetition_penalty": 2.2,
            "speed": 1.00,
        },
    },
    "narrator": {  # 旁白、纪录片
        "va": {"valence": 0.25, "arousal": 0.35, "dominance": 0.60},
        "xtts_params": {
            "temperature": 0.65,
            "top_p": 0.70,
            "top_k": 50,
            "length_penalty": 1.08,
            "repetition_penalty": 2.0,
            "speed": 0.98,
        },
    },
    "storyteller": {  # 讲故事、起伏一些
        "va": {"valence": 0.5, "arousal": 0.50, "dominance": 0.55},
        "xtts_params": {
            "temperature": 0.78,
            "top_p": 0.80,
            "top_k": 60,
            "length_penalty": 1.08,
            "repetition_penalty": 2.0,
            "speed": 1.00,
        },
    },
    # —— 情绪化、状态类 ——
    "anxious": {  # 焦虑
        "va": {"valence": -0.4, "arousal": 0.80, "dominance": 0.35},
        "xtts_params": {
            "temperature": 0.85,
            "top_p": 0.80,
            "top_k": 50,
            "length_penalty": 0.98,
            "repetition_penalty": 2.2,
            "speed": 1.05,
        },
    },
    "tired": {  # 疲惫
        "va": {"valence": -0.3, "arousal": 0.20, "dominance": 0.40},
        "xtts_params": {
            "temperature": 0.55,
            "top_p": 0.55,
            "top_k": 20,
            "length_penalty": 0.95,
            "repetition_penalty": 2.0,
            "speed": 0.92,
        },
    },
    "sleepy": {  # 困倦、含糊
        "va": {"valence": -0.1, "arousal": 0.15, "dominance": 0.35},
        "xtts_params": {
            "temperature": 0.52,
            "top_p": 0.55,
            "top_k": 20,
            "length_penalty": 0.95,
            "repetition_penalty": 2.0,
            "speed": 0.90,
        },
    },
    "bored": {  # 无聊、敷衍
        "va": {"valence": -0.2, "arousal": 0.25, "dominance": 0.45},
        "xtts_params": {
            "temperature": 0.60,
            "top_p": 0.60,
            "top_k": 30,
            "length_penalty": 0.98,
            "repetition_penalty": 2.0,
            "speed": 0.97,
        },
    },
    "surprised": {  # 惊讶（正向）
        "va": {"valence": 0.5, "arousal": 0.85, "dominance": 0.55},
        "xtts_params": {
            "temperature": 0.92,
            "top_p": 0.88,
            "top_k": 70,
            "length_penalty": 1.05,
            "repetition_penalty": 2.2,
            "speed": 1.08,
        },
    },
    "curious": {  # 好奇、探询
        "va": {"valence": 0.4, "arousal": 0.50, "dominance": 0.50},
        "xtts_params": {
            "temperature": 0.75,
            "top_p": 0.75,
            "top_k": 40,
            "length_penalty": 1.00,
            "repetition_penalty": 2.0,
            "speed": 1.01,
        },
    },
    # —— 专注、清晰度优先（ASR/字幕友好） ——
    "clear": {  # 清晰吐字，少随机性（配字幕/ASR友好）
        "va": {"valence": 0.2, "arousal": 0.35, "dominance": 0.55},
        "xtts_params": {
            "temperature": 0.50,
            "top_p": 0.55,
            "top_k": 30,
            "length_penalty": 1.10,
            "repetition_penalty": 2.4,
            "speed": 0.98,
        },
    },
    "focus": {  # 稳定、专注
        "va": {"valence": 0.25, "arousal": 0.40, "dominance": 0.60},
        "xtts_params": {
            "temperature": 0.60,
            "top_p": 0.60,
            "top_k": 40,
            "length_penalty": 1.08,
            "repetition_penalty": 2.2,
            "speed": 0.99,
        },
    },
}
