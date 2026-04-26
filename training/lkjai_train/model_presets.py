MODEL_PRESETS = {
    "quick": (512, 64, 2, 64, 4, 2, 128, 5, 20),
    "tiny-scratch": (512, 64, 2, 64, 4, 2, 128, 5, 20),
    "scratch-20m": (8192, 1024, 8, 448, 8, 2, 1152, 80000, 120000),
    "scratch-30m-debug": (8192, 1024, 8, 512, 8, 2, 1664, 20000, 120000),
    "scratch-40m": (8192, 1024, 10, 576, 8, 2, 1536, 400000, 120000),
    "scratch-60m": (8192, 1024, 12, 640, 8, 2, 1792, 120000, 120000),
    "scratch-93m-max": (8192, 1024, 15, 768, 12, 2, 1920, 120000, 120000),
}


def model_shape(model_preset: str) -> tuple[int, int, int, int, int, int, int, int, int]:
    if model_preset not in MODEL_PRESETS:
        raise ValueError(f"unknown TRAIN_MODEL_PRESET={model_preset}")
    return MODEL_PRESETS[model_preset]
