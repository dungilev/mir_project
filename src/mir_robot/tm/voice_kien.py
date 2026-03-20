#!/usr/bin/env python3
"""
restaurant_robot/main.py
BỘ NÃO CHÍNH - Điều phối toàn bộ hệ thống robot phục vụ nhà hàng.
"""


import sys
import time
import os
import re
import math
import tempfile
import subprocess
import wave
import shutil
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

try:
    import sherpa_onnx
except Exception:
    sherpa_onnx = None

try:
    import sounddevice as sd
except Exception:
    sd = None

try:
    import numpy as np
except Exception:
    np = None


# Thư viện cho Text-to-Speech
try:
   from gtts import gTTS
except Exception:
   gTTS = None

try:
   import pygame
except Exception:
   pygame = None


try:
   from restaurant_robot.config import VOICE_MODEL, ROBOT_CONFIG
   from restaurant_robot.nlp_parser import parse, ParseResult
   from restaurant_robot.order_manager import OrderManager
   from restaurant_robot.state_machine import StateMachine, RobotState
   from restaurant_robot.robot_controller import RobotController
except Exception:
   VI_NUMBER_WORDS = {
       "mot": 1, "một": 1,
       "hai": 2,
       "ba": 3,
       "bon": 4, "bốn": 4, "tu": 4, "tư": 4,
       "nam": 5, "năm": 5, "lam": 5, "lăm": 5,
       "sau": 6, "sáu": 6,
       "bay": 7, "bảy": 7,
       "tam": 8, "tám": 8,
       "chin": 9, "chín": 9,
       "muoi": 10, "mười": 10,
   }

   VI_NUMBER_PHRASES = {
       "mười một": 11, "muoi mot": 11,
       "mười hai": 12, "muoi hai": 12,
       "mười ba": 13, "muoi ba": 13,
       "mười bốn": 14, "muoi bon": 14,
       "mười lăm": 15, "muoi lam": 15,
       "mười sáu": 16, "muoi sau": 16,
       "mười bảy": 17, "muoi bay": 17,
       "mười tám": 18, "muoi tam": 18,
       "mười chín": 19, "muoi chin": 19,
   }

   def _to_number(token: str) -> Optional[int]:
       if token.isdigit():
           return int(token)
       return VI_NUMBER_WORDS.get(token)

   def _extract_table_and_qty(cleaned: str):
       tokens = re.findall(r"\d+|[\wÀ-ỹ]+", cleaned.lower())

       table = None
       qty = 1

       for phrase, value in VI_NUMBER_PHRASES.items():
           if phrase in cleaned:
               if table is None and ("bàn" in cleaned or "ban" in cleaned):
                   table = value
               qty = value

       for idx, token in enumerate(tokens):
           value = _to_number(token)
           if value is not None:
               qty = value
               if idx > 0 and tokens[idx - 1] in ("bàn", "ban", "so", "số"):
                   table = value

       if table is None and ("bàn" in cleaned or "ban" in cleaned):
           for token in tokens:
               value = _to_number(token)
               if value is not None:
                   table = value
                   break

       return table, qty

   @dataclass
   class ParseResult:
       intent: str = "UNKNOWN"
       raw_text: str = ""
       table: Optional[int] = None
       qty: int = 1

   def parse(text: str) -> ParseResult:
       cleaned = (text or "").strip().lower()
       if not cleaned:
           return ParseResult(intent="EMPTY", raw_text=text)

       table, qty = _extract_table_and_qty(cleaned)

       if "về" in cleaned or "ve" in cleaned:
           return ParseResult(intent="GO_HOME", raw_text=text, table=table, qty=qty)
       if "hủy" in cleaned or "huy" in cleaned:
           return ParseResult(intent="CANCEL", raw_text=text, table=table, qty=qty)
       if "thanh toán" in cleaned or "thanh toan" in cleaned:
           return ParseResult(intent="PAY", raw_text=text, table=table, qty=qty)
       if "ok" in cleaned or "xác nhận" in cleaned or "xac nhan" in cleaned:
           return ParseResult(intent="CONFIRM", raw_text=text, table=table, qty=qty)
       if "bàn" in cleaned or "ban" in cleaned:
           return ParseResult(intent="GO_TABLE", raw_text=text, table=table, qty=qty)
       if "nước" in cleaned or "nuoc" in cleaned or "ly" in cleaned:
           return ParseResult(intent="ORDER", raw_text=text, table=table, qty=qty)

       return ParseResult(intent="UNKNOWN", raw_text=text, table=table, qty=qty)

   @dataclass
   class _Order:
       items: list = field(default_factory=list)

       def get_total(self):
           return sum(item.get("price", 0) for item in self.items)

   class OrderManager:
       def __init__(self):
           self.orders = {}

       def _get(self, table):
           if table not in self.orders:
               self.orders[table] = _Order()
           return self.orders[table]

       def add_items_to_table(self, table, items):
           order = self._get(table)
           order.items.extend(items)
           return True, [f"Đã thêm {len(items)} món cho bàn {table}."]

       def get_order_summary(self, table):
           order = self.orders.get(table)
           if not order or not order.items:
               return None, ""
           qty = len(order.items)
           return order, f"{qty} nước lọc"

       def cancel_items(self, table):
           if table in self.orders:
               self.orders[table].items = []

       def confirm_order(self, table):
           return True

       def mark_preparing(self, table):
           return True

       def mark_ready(self, table):
           return True

       def mark_delivering(self, table):
           return True

       def mark_delivered(self, table):
           return True

       def pay_and_close(self, table):
           order = self.orders.pop(table, None)
           total = order.get_total() if order else 0
           return True, total

       def cleanup_table(self, table):
           self.orders.pop(table, None)

   class RobotState(Enum):
       IDLE = auto()
       GOING_TO_TABLE = auto()
       TAKING_ORDER = auto()
       CONFIRMING_ORDER = auto()
       GOING_TO_KITCHEN = auto()
       WAITING_FOR_FOOD = auto()
       DELIVERING_FOOD = auto()
       SERVING_CUSTOMER = auto()
       RETURNING_HOME = auto()
       PROCESSING_PAYMENT = auto()

   class StateMachine:
       def __init__(self):
           self.state = RobotState.IDLE
           self.current_table = None

       def get_state(self):
           return self.state

       def transition_to(self, state):
           self.state = state

       def get_state_name(self):
           return self.state.name

       def get_state_description(self):
           return "Sẵn sàng nhận lệnh"

   class RobotController:
       def __init__(self, init_ros_node=True):
           self.init_ros_node = init_ros_node

       def navigate_to_table(self, table_number):
           return True, f"Đi tới bàn {table_number}"

       def navigate_to_kitchen(self):
           return True, "Đi tới bếp"

       def navigate_to_home(self):
           return True, "Về vị trí trực"

       def cancel_navigation(self):
           return True

       def is_navigation_done(self):
           return True

       def stop(self):
           return True


def _resolve_model_dir():
    candidates = [
         "/root/catkin_ws/models/sherpa-onnx-zipformer-vi-2025-04-20",
         "/root/catkin_ws/src/mir_robot/tm/sherpa-onnx-zipformer-vi-2025-04-20",
         "/home/tuanminh/mir_project/sherpa-onnx-zipformer-vi-2025-04-20",
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
ROBOT_CONFIG = {"ip": "192.168.0.177", "sample_rate": 16000, "voice_record_duration": 4}

class RestaurantRobot:
   def __init__(self, simulation=False):
        self.simulation = simulation
        self.text_mode = "--text" in sys.argv
        self.audio_enabled = True
        self.recognizer = None
        self.voice_ready = False
        self._record_sample_rate = ROBOT_CONFIG.get("sample_rate", 16000)
        self._last_record_error = None
        self._last_capture_backend_error = None
        self._use_arecord_fallback = False
        self._force_pulse_capture = os.environ.get("FORCE_PULSE_CAPTURE", "1") == "1"
        self._speech_buffer = ""
        self._empty_voice_frames = 0
        self._finalize_after_empty = 1
        self._ignore_until = 0.0

        print("=" * 60)
        print("🍜 ROBOT PHỤC VỤ NHÀ HÀNG - Khởi động")
        print("=" * 60)

        self.audio_enabled = self._init_audio_output()

        if not self.text_mode:
            self._init_voice_recognizer()

        self._configure_input_backend()

        self.voice_ready = (sd is not None and self.recognizer is not None)
        if not self.text_mode and not self.voice_ready:
            self.text_mode = True
            print("⚠️ Voice mode chưa sẵn sàng. Tự chuyển sang chế độ --text.")

        self.order_manager = OrderManager()
        self.state_machine = StateMachine()
        self.robot = RobotController(init_ros_node=not simulation)

        self.current_table = None
        self.waiting_confirmation = False

        print("\n✅ Hệ thống sẵn sàng!")
        self.speak("Hệ thống robot phục vụ xin chào. Tôi đã sẵn sàng nhận lệnh từ chủ nhân.")
        self._print_help()


   def _configure_input_backend(self):
       if self._force_pulse_capture and shutil.which("parec") is not None:
           self._use_arecord_fallback = True
           self._record_sample_rate = int(os.environ.get("PULSE_RECORD_RATE", "44100"))
           src = os.environ.get("PULSE_SOURCE", "@DEFAULT_SOURCE@")
           print(f"✅ Dùng một nguồn mic Pulse cố định: {src}")
           return

       if sd is None:
           self._use_arecord_fallback = True
           return
       try:
           devices = sd.query_devices()
           has_input = any((d.get("max_input_channels", 0) or 0) > 0 for d in devices)
           if not has_input:
               self._use_arecord_fallback = True
               print("⚠️ Sounddevice không thấy mic input, chuyển sang thu âm ALSA (arecord).")
       except Exception:
           self._use_arecord_fallback = True


   def _record_with_arecord(self, duration, sample_rate):
       if np is None:
           return None

       capture_device = os.environ.get("ARECORD_DEVICE", "pulse")
       channels = 2 if capture_device == "pulse" else 1
       seconds = max(1, int(math.ceil(duration)))
       temp_path = None
       try:
           with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
               temp_path = tmp.name

           cmd = [
               "arecord", "-q",
               "-D", capture_device,
               "-f", "S16_LE",
               "-r", str(int(sample_rate)),
               "-c", str(channels),
               "-d", str(seconds),
               temp_path,
           ]
           result = subprocess.run(
               cmd,
               check=False,
               stdout=subprocess.DEVNULL,
               stderr=subprocess.PIPE,
               text=True,
           )
           if result.returncode != 0:
               err = (result.stderr or "").strip() or f"arecord failed rc={result.returncode}"
               if err != self._last_capture_backend_error:
                   print(f"⚠️ arecord backend lỗi: {err}")
                   self._last_capture_backend_error = err
               return None

           with wave.open(temp_path, "rb") as wav_file:
               wav_channels = wav_file.getnchannels()
               frames = wav_file.readframes(wav_file.getnframes())
           samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
           if wav_channels > 1 and len(samples) >= wav_channels:
               samples = samples.reshape(-1, wav_channels).mean(axis=1)
           return samples
       except Exception:
           return None
       finally:
           if temp_path and os.path.exists(temp_path):
               try:
                   os.remove(temp_path)
               except Exception:
                   pass


   def _record_with_parec(self, duration, sample_rate):
       if np is None:
           return None
       if shutil.which("parec") is None:
           return None

       source = os.environ.get("PULSE_SOURCE", "").strip()
       seconds = max(1, int(math.ceil(duration)))
       cmd = [
           "parec",
           "--rate", str(int(sample_rate)),
           "--channels", "1",
           "--format", "s16le",
           "--raw",
       ]
       if source:
           cmd[1:1] = ["--device", source]
       try:
           result = subprocess.run(
               cmd,
               stdout=subprocess.PIPE,
               stderr=subprocess.DEVNULL,
               timeout=seconds + 1,
               check=False,
           )
           raw = result.stdout or b""
           if len(raw) < 2:
               return None
           samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
           return samples
       except Exception:
           return None


   def _init_audio_output(self):
       if pygame is None:
           print("⚠️ Không có pygame, tắt phát âm thanh.")
           return False

       candidates = []
       env_device = os.environ.get("AUDIODEV")
       if env_device:
           candidates.append(env_device)
       candidates.extend(["default", "plughw:1,0", "plughw:0,3", "hw:1,0", "hw:0,3"])

       tried = set()
       for device in candidates:
           if not device or device in tried:
               continue
           tried.add(device)
           try:
               os.environ["AUDIODEV"] = device
               pygame.mixer.init()
               print(f"✅ Audio output sẵn sàng ({device})")
               return True
           except Exception:
               continue

       try:
           if "AUDIODEV" in os.environ:
               del os.environ["AUDIODEV"]
           pygame.mixer.init()
           print("✅ Audio output sẵn sàng (mặc định)")
           return True
       except Exception as e:
           print(f"⚠️ Không khởi tạo được audio output: {e}")
           return False


   def speak(self, text):
       """Hàm phát âm thanh tiếng Việt từ Text"""
       print(f"\n🔊 Robot nói: {text}")
      
       # Nếu đang chạy chế độ text thì không cần phát tiếng
       if self.text_mode:
           return


       if not self.audio_enabled:
           return

       if gTTS is None or pygame is None:
           return


       try:
           # Tạo file âm thanh mp3 từ text
           tts = gTTS(text=text, lang='vi')
           filename = "robot_voice_temp.mp3"
           tts.save(filename)
          
           # Phát âm thanh
           pygame.mixer.music.load(filename)
           pygame.mixer.music.play()
          
           # Chờ cho đến khi robot nói xong mới chạy lệnh tiếp
           while pygame.mixer.music.get_busy():
               pygame.time.Clock().tick(10)
           self._ignore_until = time.time() + 0.8
              
       except Exception as e:
           print(f"❌ Lỗi phát âm thanh (Có thể do mất mạng): {e}")


   def _accumulate_voice_command(self, text):
       normalized = (text or "").strip()
       if normalized:
           self._empty_voice_frames = 0
           if not self._speech_buffer:
               self._speech_buffer = normalized
           elif normalized not in self._speech_buffer:
               self._speech_buffer = f"{self._speech_buffer} {normalized}".strip()

           try:
               quick_result = parse(self._speech_buffer)
               actionable = (
                   (quick_result.intent == "GO_TABLE" and quick_result.table is not None)
                   or quick_result.intent in ("GO_HOME", "CANCEL", "PAY", "CONFIRM")
                   or (quick_result.intent == "ORDER" and getattr(quick_result, "qty", 0) > 0)
               )
               if actionable:
                   final_text = self._speech_buffer.strip()
                   self._speech_buffer = ""
                   self._empty_voice_frames = 0
                   print(f"   🧾 Câu đầy đủ: \"{final_text}\"")
                   return final_text
           except Exception:
               pass
           return None

       if not self._speech_buffer:
           return None

       self._empty_voice_frames += 1
       if self._empty_voice_frames < self._finalize_after_empty:
           return None

       final_text = self._speech_buffer.strip()
       self._speech_buffer = ""
       self._empty_voice_frames = 0
       if final_text:
           print(f"   🧾 Câu đầy đủ: \"{final_text}\"")
       return final_text


   def _init_voice_recognizer(self):
       cfg = VOICE_MODEL
       model_dir = cfg["model_dir"]

       required_files = [
           os.path.join(model_dir, cfg["tokens"]),
           os.path.join(model_dir, cfg["encoder"]),
           os.path.join(model_dir, cfg["decoder"]),
           os.path.join(model_dir, cfg["joiner"]),
       ]
       missing_files = [path for path in required_files if not os.path.isfile(path)]
       if missing_files:
           print("❌ Thiếu file model giọng nói:")
           for path in missing_files:
               print(f"   - {path}")
           self.recognizer = None
           return

       if sherpa_onnx is None:
           print("⚠️ Không có sherpa_onnx, tắt nhận diện giọng nói.")
           self.recognizer = None
           return

       try:
           self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
               tokens=f"{model_dir}/{cfg['tokens']}",
               encoder=f"{model_dir}/{cfg['encoder']}",
               decoder=f"{model_dir}/{cfg['decoder']}",
               joiner=f"{model_dir}/{cfg['joiner']}",
               num_threads=cfg["num_threads"],
               sample_rate=cfg["sample_rate"],
               feature_dim=cfg["feature_dim"],
           )
           print("✅ Model nhận diện giọng nói đã sẵn sàng")
       except Exception as e:
           print(f"❌ Lỗi model giọng nói: {e}")
           self.recognizer = None


   def _print_help(self):
       print("\n" + "=" * 60)
       print("📋 HƯỚNG DẪN SỬ DỤNG")
       print("=" * 60)
       print("Nhấn Ctrl+C để thoát.\n")


   def listen(self):
       now = time.time()
       if now < self._ignore_until:
           time.sleep(max(0.0, self._ignore_until - now))
           return ""

       if self.recognizer is None:
           if not getattr(self, "_voice_unavailable_warned", False):
               print("⚠️ Voice mode chưa sẵn sàng. Hãy chạy với --text để test nhanh.")
               self._voice_unavailable_warned = True
           time.sleep(0.3)
           return ""

       sample_rate = ROBOT_CONFIG["sample_rate"]
       record_rate = int(self._record_sample_rate)
       duration = ROBOT_CONFIG["voice_record_duration"]


       state_desc = self.state_machine.get_state_description()
       print(f"\n🎤 [{self.state_machine.get_state_name()}] {state_desc}")
       print("   Đang nghe...", end=" ", flush=True)


       try:
           if self._use_arecord_fallback:
               samples = self._record_with_parec(duration, record_rate)
               if samples is None:
                   samples = self._record_with_arecord(duration, record_rate)
               if samples is None:
                   raise RuntimeError("Pulse/ALSA capture failed")
           else:
               recording = sd.rec(
                   int(duration * record_rate),
                   samplerate=record_rate,
                   channels=1,
                   dtype='float32'
               )
               sd.wait()
               samples = recording[:, 0]

           if record_rate != sample_rate and np is not None and len(samples) > 1:
               target_len = int(len(samples) * (sample_rate / float(record_rate)))
               if target_len > 1:
                   x_old = np.linspace(0.0, 1.0, num=len(samples), endpoint=True)
                   x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=True)
                   samples = np.interp(x_new, x_old, samples).astype('float32')

           stream = self.recognizer.create_stream()
           stream.accept_waveform(sample_rate, samples)
           self.recognizer.decode_stream(stream)


           text = stream.result.text.strip()
           if text:
               print(f"📝 \"{text}\"")
           else:
               print("(im lặng)")
           return text
       except Exception as e:
           message = str(e)
           if "Error querying device -1" in message:
               self._use_arecord_fallback = True
               print("⚠️ Không lấy được input device mặc định, chuyển sang ALSA arecord.")

           if "Invalid sample rate" in message and not self._use_arecord_fallback:
               try:
                   input_idx = sd.default.device[0]
                   device_info = sd.query_devices(input_idx, 'input')
                   default_sr = int(device_info.get('default_samplerate', sample_rate))
                   if default_sr > 0 and default_sr != self._record_sample_rate:
                       self._record_sample_rate = default_sr
                       print(f"⚠️ Mic không hỗ trợ {sample_rate}Hz, chuyển sang ghi âm {default_sr}Hz và tự resample.")
               except Exception:
                   pass

           if message != self._last_record_error:
               print(f"❌ Lỗi thu âm: {e}")
               self._last_record_error = message
           time.sleep(0.2)
           return ""


   def process_voice_input(self, text):
       if not text:
           return
       result = parse(text)
       print(f"   🧠 {result}")
       state = self.state_machine.get_state()
       handler = self._get_handler(state)
       handler(result)


   def _tick_state_machine(self):
       state = self.state_machine.get_state()
       handler = self._get_handler(state)
       handler(ParseResult(intent="EMPTY", raw_text=""))


   def _get_handler(self, state):
       handlers = {
           RobotState.IDLE: self._handle_idle,
           RobotState.GOING_TO_TABLE: self._handle_going_to_table,
           RobotState.TAKING_ORDER: self._handle_taking_order,
           RobotState.CONFIRMING_ORDER: self._handle_confirming_order,
           RobotState.GOING_TO_KITCHEN: self._handle_going_to_kitchen,
           RobotState.WAITING_FOR_FOOD: self._handle_waiting_for_food,
           RobotState.DELIVERING_FOOD: self._handle_delivering_food,
           RobotState.SERVING_CUSTOMER: self._handle_serving_customer,
           RobotState.RETURNING_HOME: self._handle_returning_home,
           RobotState.PROCESSING_PAYMENT: self._handle_payment,
       }
       return handlers.get(state, self._handle_unknown)


   # --- CÁC HANDLER GIAO TIẾP ---


   def _handle_idle(self, result: ParseResult):
       if result.intent == "CALL_ROBOT":
           table = result.table
           if table:
               self.speak(f"Vâng, robot đang di chuyển đến bàn {table}.")
               self._go_to_table(table)
           else:
               self.speak("Dạ quý khách ở bàn số mấy ạ?")


       elif result.intent == "GO_TABLE":
           table = result.table
           if table:
               self.speak(f"Đang đi đến bàn {table}.")
               self._go_to_table(table)


       elif result.intent == "GO_HOME":
           self.speak("Tôi đang ở vị trí trực rồi ạ.")


       elif result.intent != "UNKNOWN" and result.intent != "EMPTY":
           print(f"   ℹ️ Robot đang ở vị trí trực.")


   def _handle_going_to_table(self, result: ParseResult):
       if self.robot.is_navigation_done():
           self.state_machine.transition_to(RobotState.TAKING_ORDER)
           self.speak(f"Xin chào chủ nhân tại bàn {self.current_table}. Hiện tại quán chỉ phục vụ nước lọc. Chủ nhân muốn dùng mấy ly ạ?")
       else:
           if result.intent == "CANCEL" or result.intent == "GO_HOME":
               self.robot.cancel_navigation()
               self.state_machine.transition_to(RobotState.IDLE)
               self.speak("Đã hủy lệnh đến bàn, tôi đang quay về vị trí trực.")
               self.robot.navigate_to_home()


   def _handle_taking_order(self, result: ParseResult):
       if result.intent == "ORDER":
           qty = getattr(result, 'qty', 1)
          
           # BÍ QUYẾT Ở ĐÂY: Nhân bản món ăn theo biến qty
           # Nếu qty = 2, items_to_add sẽ là [{"name": "nước lọc"}, {"name": "nước lọc"}]
           items_to_add = [{"name": "nước lọc", "price": 5000}] * qty
          
           self._add_items_to_order(items_to_add)
           self._confirm_order()


       elif result.intent == "CANCEL":
           self.order_manager.cancel_items(self.current_table)
           self.speak("Đã hủy đơn hàng.")


       elif result.intent == "PAY":
           self._process_payment(self.current_table)


       elif result.intent == "GO_HOME":
           self.state_machine.transition_to(RobotState.IDLE)
           self.speak("Tôi xin phép quay về vị trí trực.")
           self.robot.navigate_to_home()


       elif result.intent == "UNKNOWN" and result.raw_text:
           qty = getattr(result, 'qty', 1)
           items_to_add = [{"name": "nước lọc", "price": 5000}] * qty
           self._add_items_to_order(items_to_add)
           self._confirm_order()


   def _handle_confirming_order(self, result: ParseResult):
       if result.intent == "CONFIRM":
           self.order_manager.confirm_order(self.current_table)
           self.speak("Đơn hàng đã được xác nhận. Quý khách vui lòng đợi một chút, tôi sẽ đi lấy nước lọc ngay.")
           self._go_to_kitchen_for_order()


       elif result.intent == "DENY" or result.intent == "CANCEL":
           self.order_manager.cancel_items(self.current_table)
           self.state_machine.transition_to(RobotState.TAKING_ORDER)
           self.speak("Dạ vâng, tôi đã hủy đơn. Quý khách muốn gọi lại không ạ?")


   def _handle_going_to_kitchen(self, result: ParseResult):
       if self.robot.is_navigation_done():
           self.state_machine.transition_to(RobotState.WAITING_FOR_FOOD)
           self.order_manager.mark_preparing(self.current_table)
           self.speak(f"Đã đến bếp. Bếp vui lòng chuẩn bị nước lọc cho bàn {self.current_table}.")


   def _handle_waiting_for_food(self, result: ParseResult):
       if result.intent == "CONFIRM":
           self.order_manager.mark_ready(self.current_table)
           self.speak(f"Đã nhận đủ nước lọc. Đang mang ra bàn {self.current_table}.")
           self.state_machine.transition_to(RobotState.DELIVERING_FOOD)
           self.robot.navigate_to_table(self.current_table)


   def _handle_delivering_food(self, result: ParseResult):
       if self.robot.is_navigation_done():
           self.state_machine.transition_to(RobotState.SERVING_CUSTOMER)
           self.order_manager.mark_delivering(self.current_table)
           self.speak(f"Nước lọc của bàn {self.current_table} đã tới. Mời quý khách nhận đồ và nói OK khi đã lấy xong ạ.")


   def _handle_serving_customer(self, result: ParseResult):
       if result.intent == "CONFIRM":
           self.order_manager.mark_delivered(self.current_table)
           self.speak("Chúc quý khách dùng ngon miệng. Tôi xin phép trở về vị trí trực.")
           self._return_home()


       elif result.intent == "PAY":
           self._process_payment(self.current_table)


   def _handle_returning_home(self, result: ParseResult):
       if self.robot.is_navigation_done():
           self.state_machine.transition_to(RobotState.IDLE)
           if self.current_table:
               self.order_manager.cleanup_table(self.current_table)
               self.current_table = None
           print("   🏠 Đã về vị trí trực!")


   def _handle_payment(self, result: ParseResult):
       if result.intent == "CONFIRM":
           # Hỗ trợ lấy trả về nếu hàm pay_and_close trả về tuple
           res = self.order_manager.pay_and_close(self.current_table)
           self.speak(f"Cảm ơn quý khách đã thanh toán. Hẹn gặp lại quý khách.")
           self._return_home()
       elif result.intent == "DENY":
           self.speak("Dạ vâng, đã hủy thanh toán.")
           self.state_machine.transition_to(RobotState.SERVING_CUSTOMER)


   def _handle_unknown(self, result: ParseResult):
       pass


   # --- CÁC HÀNH ĐỘNG PHỤC VỤ CƠ BẢN ---


   def _go_to_table(self, table_number):
       self.current_table = table_number
       self.state_machine.current_table = table_number
       self.state_machine.transition_to(RobotState.GOING_TO_TABLE)
       success, msg = self.robot.navigate_to_table(table_number)
       if self.simulation or not success:
           self.state_machine.transition_to(RobotState.TAKING_ORDER)
           self.speak(f"Xin chào quý khách tại bàn {self.current_table}. Hiện tại quán chỉ phục vụ nước lọc. Quý khách muốn dùng mấy ly ạ?")


   def _add_items_to_order(self, items):
       if not self.current_table:
           return
      
       try:
           # Bắt kết quả trả về từ OrderManager để xem có thành công không
           res = self.order_manager.add_items_to_table(self.current_table, items)
           if isinstance(res, tuple) and len(res) == 2:
               success, messages = res
               for msg in messages:
                   print(f"   📝 {msg}")
           else:
               print(f"   📝 Đã xử lý thêm món: {items}")
       except Exception as e:
           print(f"   ⚠️ Lỗi khi thêm món (OrderManager): {e}")


   def _confirm_order(self):
       order, summary = self.order_manager.get_order_summary(self.current_table)
      
       # Nếu giỏ hàng có đồ thì mới chuyển trạng thái và đọc xác nhận
       if order and order.items:
           self.state_machine.transition_to(RobotState.CONFIRMING_ORDER)
          
           # BÍ QUYẾT: Dùng luôn chuỗi 'summary' của OrderManager để phát âm.
           # Robot sẽ tự động đọc "2 nước lọc", "3 phở bò" rất mượt mà.
           self.speak(f"Quý khách đã gọi: {summary}. Quý khách xác nhận đúng không ạ?")
       else:
           self.speak("Xin lỗi, tôi không thể thêm món này vào đơn. Hệ thống đang báo lỗi thực đơn.")


   def _read_order(self):
       pass


   def _go_to_kitchen_for_order(self):
       self.state_machine.transition_to(RobotState.GOING_TO_KITCHEN)
       success, msg = self.robot.navigate_to_kitchen()
       if self.simulation:
           self.state_machine.transition_to(RobotState.WAITING_FOR_FOOD)
           self.order_manager.mark_preparing(self.current_table)


   def _process_payment(self, table_number):
       self.current_table = table_number
       self.state_machine.current_table = table_number
       order, summary = self.order_manager.get_order_summary(table_number)
      
       if order and order.items:
           self.state_machine.transition_to(RobotState.PROCESSING_PAYMENT)
           total = order.get_total()
           self.speak(f"Tổng hóa đơn của bàn {table_number} là {total} đồng. Quý khách có đồng ý thanh toán không ạ?")
       else:
           self.speak(f"Bàn {table_number} hiện tại chưa gọi món nào ạ.")


   def _return_home(self):
       self.state_machine.transition_to(RobotState.RETURNING_HOME)
       self.robot.navigate_to_home()
       if self.simulation:
           self.state_machine.transition_to(RobotState.IDLE)
           if self.current_table:
               self.order_manager.cleanup_table(self.current_table)
               self.current_table = None


   def run(self):
       try:
           while True:
               text = self.listen()
               final_text = self._accumulate_voice_command(text)
               if final_text:
                   self.process_voice_input(final_text)
              
               if self.state_machine.get_state() in (
                   RobotState.GOING_TO_TABLE,
                   RobotState.GOING_TO_KITCHEN,
                   RobotState.DELIVERING_FOOD,
                   RobotState.RETURNING_HOME,
               ):
                   if self.robot.is_navigation_done():
                       self._tick_state_machine()
       except KeyboardInterrupt:
           self.speak("Hệ thống đang tắt. Tạm biệt.")
           self.robot.stop()


   def run_text_mode(self):
       print("\n💻 CHẾ ĐỘ TEXT (gõ lệnh thay vì nói)")
       try:
           while True:
               state_name = self.state_machine.get_state_name()
               text = input(f"\n[{state_name}] 🎤 Nhập lệnh: ").strip()
               if text.lower() in ('quit', 'exit', 'thoát'):
                   break
               self.process_voice_input(text)
       except (KeyboardInterrupt, EOFError):
           print("\n🛑 Tắt hệ thống.")




def main():
   sim_mode = "--sim" in sys.argv or "--text" in sys.argv
   text_mode = "--text" in sys.argv


   robot = RestaurantRobot(simulation=sim_mode)


   if text_mode or robot.text_mode:
       robot.run_text_mode()
   else:
       robot.run()


if __name__ == "__main__":
   main()