import sherpa_onnx, os, soundfile as sf
text = "Xin chào, tôi là rô bốt phục vụ 24 7. Rất vui được phục vụ hai bạn."
model_dir = "/home/tuanminh/mir_project/src/vits-piper-vi_VN-25hours_single-low"
tts_config = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
            model=os.path.join(model_dir, "vi_VN-25hours_single-low.onnx"),
            tokens=os.path.join(model_dir, "tokens.txt"),
            data_dir=os.path.join(model_dir, "espeak-ng-data")
        )
    )
)
tts = sherpa_onnx.OfflineTts(tts_config)
audio = tts.generate(text, speed=1.0)
sf.write("/home/tuanminh/mir_project/src/test_giong_25hours.wav", audio.samples, audio.sample_rate, subtype='PCM_16')
print("Đã tạo file test: /home/tuanminh/mir_project/src/test_giong_25hours.wav")
