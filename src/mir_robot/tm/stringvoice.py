import sherpa_onnx
import sounddevice as sd
import numpy as np
import queue
import sys

# Khởi tạo hàng đợi để chứa các mảnh âm thanh từ Microphone
audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    """Hàm này chạy ngầm, liên tục hút âm thanh từ mic ném vào queue."""
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(indata.copy())

def main():
    # 1. Nạp "bộ não" AI
    print("⏳ Đang nạp mô hình, vui lòng đợi...")
    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder="./sherpa-onnx-zipformer-vi-2025-04-20/encoder-epoch-12-avg-8.onnx",
        decoder="./sherpa-onnx-zipformer-vi-2025-04-20/decoder-epoch-12-avg-8.onnx",
        joiner="./sherpa-onnx-zipformer-vi-2025-04-20/joiner-epoch-12-avg-8.onnx",
        tokens="./sherpa-onnx-zipformer-vi-2025-04-20/tokens.txt",
        num_threads=2, # Máy bạn khỏe có thể tăng lên 4 cho lẹ
    )
    
    SAMPLE_RATE = 16000 # Tần số chuẩn của mô hình
    
    print("\n✅ Nạp thành công!")
    print("👉 Bấm phím [ENTER] để bắt đầu nói...")
    input() # Chờ bạn gõ Enter
    
    print("🎙️ Đang nghe... (Nói xong hãy bấm [ENTER] lần nữa)")
    
    # 2. Bật ghi âm
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', callback=audio_callback)
    with stream:
        input() # Luồng ghi âm chạy cho đến khi bạn bấm Enter lần 2
    
    print("⏳ Đang giải mã...")
    
    # 3. Gom toàn bộ âm thanh lại thành một cục
    audio_chunks = []
    while not audio_queue.empty():
        audio_chunks.append(audio_queue.get())
    
    if not audio_chunks:
        print("⚠️ Không thu được âm thanh nào.")
        return
        
    audio_data = np.concatenate(audio_chunks, axis=0).flatten()
    
    # 4. Cho AI dịch và LẤY STRING RA
    s = recognizer.create_stream()
    s.accept_waveform(SAMPLE_RATE, audio_data)
    recognizer.decode_stream(s)
    
    # Đây chính là cái biến String mà bạn đang cần!
    lenh_giong_noi = s.result.text 
    
    print("\n" + "="*50)
    print("🎯 CHUỖI BẠN VỪA NÓI LÀ:", lenh_giong_noi)
    print("="*50 + "\n")

if __name__ == "__main__":
    main()