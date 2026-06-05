import os


def _resolve_model_dir():
    candidates = [
        "/root/catkin_ws/models/sherpa-onnx-zipformer-vi-2025-04-20",
        "/root/catkin_ws/src/mir_robot/tm/sherpa-onnx-zipformer-vi-2025-04-20",
        "/home/dung/mir_project/sherpa-onnx-zipformer-vi-2025-04-20",
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return candidates[0]


VOICE_MODEL = {
    "model_dir": _resolve_model_dir(),
    "tokens": "tokens.txt",
    "encoder": "encoder-epoch-12-avg-8.onnx",
    "decoder": "decoder-epoch-12-avg-8.onnx",
    "joiner": "joiner-epoch-12-avg-8.onnx",
    "num_threads": 2,
    "sample_rate": 16000,
    "feature_dim": 80,
}

ROBOT_CONFIG = {
    "ip": "192.168.0.177",
    "sample_rate": 16000,
    "voice_record_duration": 4,
}
