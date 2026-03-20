# Copyright (c) 2018-2022, Martin Günther (DFKI GmbH) and contributors
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of the copyright holder nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# ---------------------------------------------------------------------------
# File: rosbridge.py
# Mô tả: Module kết nối với ROS thông qua giao thức WebSocket (rosbridge).
#        Cho phép publish/subscribe topic và gọi service của ROS từ Python
#        mà không cần cài đặt ROS trực tiếp trên máy client.
# ---------------------------------------------------------------------------

import websocket   # Thư viện WebSocket để kết nối tới rosbridge server
import threading  # Thư viện đa luồng để chạy WebSocket ở luồng riêng

import json        # Dùng để mã hóa/giải mã dữ liệu JSON (định dạng giao tiếp của rosbridge)
import traceback  # Dùng để in chi tiết lỗi khi xảy ra exception
import time       # Dùng để tạo độ trễ (dùng trong callService đồng bộ)

import string     # Dùng để tạo chuỗi ký tự ngẫu nhiên (cho ID duy nhất)
import random     # Dùng để sinh số/ký tự ngẫu nhiên


# ===========================================================================
# Lớp RosbridgeSetup
# ---------------------------------------------------------------------------
# Cung cấp giao diện cấp cao để tương tác với ROS qua rosbridge WebSocket.
# Người dùng chỉ cần làm việc với lớp này để publish/subscribe/callService.
# ===========================================================================
class RosbridgeSetup:
    def __init__(self, host, port):
        """
        Khởi tạo kết nối tới rosbridge server.
        - host: địa chỉ IP của robot/máy chạy rosbridge (ví dụ: '192.168.12.20')
        - port: cổng WebSocket của rosbridge (thường là 9090)
        """
        self.callbacks = {}           # Từ điển lưu danh sách callback theo tên topic
        self.service_callbacks = {}   # Từ điển lưu callback theo ID của service call
        self.resp = None              # Biến lưu kết quả trả về khi gọi service đồng bộ
        self.connection = RosbridgeWSConnection(host, port)  # Tạo kết nối WebSocket thực tế
        self.connection.registerCallback(self.onMessageReceived)  # Đăng ký hàm xử lý tin nhắn nhận được

    def publish(self, topic, obj):
        """
        Gửi (publish) một message tới một topic của ROS.
        - topic: tên topic ROS (ví dụ: '/cmd_vel')
        - obj: nội dung message dưới dạng dict Python (ví dụ: {'linear': {'x': 0.5}, 'angular': {'z': 0.0}})
        """
        # Tạo gói lệnh theo định dạng rosbridge protocol
        pub = {"op": "publish", "topic": topic, "msg": obj}
        self.send(pub)  # Gửi gói lệnh qua WebSocket

    def subscribe(self, topic, callback, throttle_rate=-1):
        """
        Đăng ký lắng nghe (subscribe) một topic của ROS.
        - topic: tên topic cần lắng nghe
        - callback: hàm sẽ được gọi mỗi khi có tin nhắn mới từ topic đó
        - throttle_rate: giới hạn tần suất nhận tin (ms), -1 = không giới hạn
        """
        # Chỉ gửi lệnh subscribe nếu đây là lần đầu đăng ký callback cho topic này
        if self.addCallback(topic, callback):
            sub = {"op": "subscribe", "topic": topic}  # Tạo gói lệnh subscribe
            if throttle_rate > 0:
                # Thêm tham số giới hạn tần suất nếu được chỉ định
                sub['throttle_rate'] = throttle_rate

            self.send(sub)  # Gửi lệnh subscribe tới rosbridge

    def unhook(self, callback):
        """
        Gỡ bỏ một callback đã đăng ký trước đó.
        Nếu một topic không còn callback nào, sẽ tự động unsubscribe khỏi topic đó.
        - callback: hàm callback cần gỡ bỏ
        """
        keys_for_deletion = []  # Danh sách các topic cần unsubscribe

        # Duyệt qua tất cả topic và danh sách callback của chúng
        for key, values in self.callbacks.items():
            for value in values:
                if callback == value:  # Tìm thấy callback cần xóa
                    print("Found!")
                    values.remove(value)  # Xóa callback khỏi danh sách
                    if len(values) == 0:
                        # Nếu không còn callback nào cho topic này, đánh dấu để unsubscribe
                        keys_for_deletion.append(key)

        # Unsubscribe và xóa các topic không còn callback
        for key in keys_for_deletion:
            self.unsubscribe(key)       # Gửi lệnh unsubscribe tới rosbridge
            self.callbacks.pop(key)    # Xóa topic khỏi từ điển callbacks

    def unsubscribe(self, topic):
        """
        Hủy đăng ký lắng nghe một topic.
        - topic: tên topic cần hủy đăng ký
        """
        unsub = {"op": "unsubscribe", "topic": topic}  # Tạo gói lệnh unsubscribe
        self.send(unsub)  # Gửi lệnh tới rosbridge

    def callService(self, serviceName, callback=None, msg=None, timeout=30.0):
        """
        Gọi một ROS service.
        - serviceName: tên service cần gọi (ví dụ: '/move_base/clear_costmaps')
        - callback: hàm xử lý kết quả trả về (nếu None => gọi đồng bộ, chờ kết quả)
        - msg: tham số truyền vào service (dict), None nếu không có tham số
        - timeout: thời gian chờ tối đa (giây), mặc định 30s. 0 = không timeout.
        """
        id = self.generate_id()  # Tạo ID duy nhất để theo dõi response của service call này
        # Tạo gói lệnh gọi service theo định dạng rosbridge
        call = {"op": "call_service", "id": id, "service": serviceName}
        if msg is not None:
            call['args'] = msg  # Thêm tham số nếu có

        if callback is None:
            # --- Chế độ đồng bộ (blocking): chờ đến khi có kết quả trả về ---
            self.resp = None

            def internalCB(msg):
                # Callback nội bộ: lưu kết quả vào self.resp khi nhận được response
                self.resp = msg
                return None

            self.addServiceCallback(id, internalCB)  # Đăng ký callback nội bộ
            self.send(call)                          # Gửi lệnh gọi service

            # Vòng lặp chờ cho đến khi nhận được response (kiểm tra mỗi 10ms)
            start_time = time.time()
            while self.resp is None:
                if timeout > 0 and (time.time() - start_time) > timeout:
                    raise TimeoutError(
                        "Service call to '%s' timed out after %.1f seconds" % (serviceName, timeout)
                    )
                time.sleep(0.01)

            return self.resp  # Trả về kết quả sau khi nhận được

        # --- Chế độ bất đồng bộ (non-blocking): dùng callback được cung cấp ---
        self.addServiceCallback(id, callback)  # Đăng ký callback do người dùng cung cấp
        self.send(call)                        # Gửi lệnh gọi service
        return None  # Trả về ngay lập tức, kết quả sẽ được xử lý trong callback

    def send(self, obj):
        """
        Chuyển đổi dict Python thành chuỗi JSON và gửi qua WebSocket.
        - obj: dict Python chứa gói lệnh rosbridge (publish/subscribe/call_service...)
        """
        try:
            # json.dumps() chuyển dict thành chuỗi JSON để gửi qua WebSocket
            self.connection.sendString(json.dumps(obj))
        except Exception:
            traceback.print_exc()  # In chi tiết lỗi
            raise                  # Ném lại exception để caller biết có lỗi

    def generate_id(self, chars=16):
        """
        Tạo một chuỗi ID ngẫu nhiên gồm chữ cái và chữ số.
        - chars: độ dài chuỗi ID (mặc định 16 ký tự)
        - Dùng SystemRandom để đảm bảo tính ngẫu nhiên an toàn hơn
        """
        return ''.join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(chars))

    def addServiceCallback(self, id, callback):
        """
        Lưu callback cho một service call theo ID.
        - id: ID duy nhất của service call
        - callback: hàm sẽ được gọi khi nhận được response từ service
        """
        self.service_callbacks[id] = callback  # Lưu vào từ điển theo ID

    def addCallback(self, topic, callback):
        """
        Đăng ký callback cho một topic.
        - topic: tên topic
        - callback: hàm xử lý tin nhắn khi nhận được từ topic
        - Trả về True nếu đây là callback đầu tiên cho topic (cần gửi lệnh subscribe)
        - Trả về False nếu topic đã có callback rồi (không cần gửi subscribe lại)
        """
        if topic in self.callbacks:
            # Topic đã được subscribe, chỉ thêm callback mới vào danh sách
            self.callbacks[topic].append(callback)
            return False  # Không cần gửi lệnh subscribe lại

        # Đây là callback đầu tiên cho topic này, tạo danh sách mới
        self.callbacks[topic] = [callback]
        return True  # Cần gửi lệnh subscribe tới rosbridge

    def is_connected(self):
        """Kiểm tra xem WebSocket đã kết nối thành công chưa. Trả về True/False."""
        return self.connection.connected

    def is_errored(self):
        """Kiểm tra xem kết nối có đang gặp lỗi không. Trả về True/False."""
        return self.connection.errored

    def onMessageReceived(self, message):
        """
        Hàm xử lý tất cả tin nhắn nhận được từ rosbridge qua WebSocket.
        Phân loại tin nhắn theo trường 'op' và chuyển tới callback phù hợp.
        - message: chuỗi JSON nhận được từ WebSocket
        """
        try:
            # Giải mã chuỗi JSON thành dict Python
            obj = json.loads(message)

            if 'op' in obj:  # Kiểm tra xem tin nhắn có chứa trường 'op' không
                option = obj['op']  # Lấy loại thao tác (publish, service_response, ...)

                if option == "publish":  # Tin nhắn từ một topic mà ta đã subscribe
                    topic = obj["topic"]   # Lấy tên topic
                    msg = obj["msg"]       # Lấy nội dung tin nhắn
                    if topic in self.callbacks:
                        # Gọi tất cả callback đã đăng ký cho topic này
                        for callback in self.callbacks[topic]:
                            try:
                                callback(msg)  # Gọi callback với nội dung tin nhắn
                            except Exception:
                                print("exception on callback", callback, "from", topic)
                                traceback.print_exc()
                                raise

                elif option == "service_response":  # Kết quả trả về từ một service call
                    if "id" in obj:  # Kiểm tra xem response có kèm ID không
                        id = obj["id"]          # Lấy ID để xác định service call nào
                        values = obj["values"]  # Lấy giá trị kết quả trả về
                        if id in self.service_callbacks:
                            try:
                                # Gọi callback tương ứng với ID của service call
                                self.service_callbacks[id](values)
                            except Exception:
                                print("exception on callback ID:", id)
                                traceback.print_exc()
                                raise
                    else:
                        print("Missing ID!")  # Cảnh báo nếu response không có ID
                else:
                    print("Recieved unknown option - it was: ", option)  # Loại op không xác định
            else:
                print("No OP key!")  # Cảnh báo nếu tin nhắn không có trường 'op'
        except Exception:
            print("exception in onMessageReceived")
            print("message", message)  # In lại tin nhắn gây lỗi
            traceback.print_exc()
            raise


# ===========================================================================
# Lớp RosbridgeWSConnection
# ---------------------------------------------------------------------------
# Quản lý kết nối WebSocket thực tế tới rosbridge server.
# Lớp này ở tầng thấp hơn (low-level), thường không dùng trực tiếp bên ngoài.
# ===========================================================================
class RosbridgeWSConnection:
    def __init__(self, host, port):
        """
        Khởi tạo kết nối WebSocket tới rosbridge server.
        - host: địa chỉ IP của rosbridge server
        - port: cổng WebSocket (thường là 9090)
        """
        # Tạo đối tượng WebSocketApp với URL và các hàm xử lý sự kiện
        self.ws = websocket.WebSocketApp(
            ("ws://%s:%d/" % (host, port)),  # URL kết nối WebSocket
            on_message=self.on_message,       # Hàm gọi khi nhận được tin nhắn
            on_error=self.on_error,           # Hàm gọi khi có lỗi
            on_close=self.on_close            # Hàm gọi khi kết nối đóng
        )
        self.ws.on_open = self.on_open  # Hàm gọi khi kết nối được mở thành công

        # Chạy WebSocket trong một thread riêng để không block chương trình chính
        self.run_thread = threading.Thread(target=self.run)
        self.run_thread.start()  # Bắt đầu thread kết nối

        self.connected = False  # Trạng thái kết nối ban đầu là chưa kết nối
        self.errored = False    # Trạng thái lỗi ban đầu là không có lỗi
        self.callbacks = []     # Danh sách hàm xử lý tin nhắn nhận được

    def on_open(self):
        """Được gọi tự động khi kết nối WebSocket mở thành công."""
        print("### ROS bridge connected ###")  # Thông báo kết nối thành công
        self.connected = True  # Đánh dấu trạng thái đã kết nối

    def sendString(self, message):
        """
        Gửi một chuỗi (JSON) qua WebSocket.
        - message: chuỗi JSON cần gửi
        """
        if not self.connected:
            # Chưa kết nối, không thể gửi tin nhắn
            print("Error: not connected, could not send message")
            # TODO: ném exception để caller biết gửi thất bại
        else:
            self.ws.send(message)  # Gửi tin nhắn qua WebSocket

    def on_error(self, error):
        """Được gọi tự động khi WebSocket gặp lỗi."""
        self.errored = True             # Đánh dấu trạng thái lỗi
        print("Error: %s" % error)      # In thông báo lỗi ra màn hình

    def on_close(self):
        """Được gọi tự động khi kết nối WebSocket bị đóng."""
        self.connected = False              # Đánh dấu trạng thái đã ngắt kết nối
        print("### ROS bridge closed ###")  # Thông báo kết nối đã đóng

    def run(self, *args):
        """
        Hàm chạy trong thread riêng, duy trì vòng lặp WebSocket.
        run_forever() sẽ giữ kết nối WebSocket hoạt động liên tục cho đến khi bị đóng.
        """
        self.ws.run_forever()  # Chạy vòng lặp sự kiện WebSocket mãi mãi

    def on_message(self, message):
        """
        Được gọi tự động khi nhận được tin nhắn từ WebSocket.
        Chuyển tin nhắn tới tất cả callback đã đăng ký.
        - message: chuỗi JSON nhận được từ rosbridge
        """
        # Gọi lần lượt tất cả hàm xử lý đã đăng ký (thường chỉ có onMessageReceived của RosbridgeSetup)
        for callback in self.callbacks:
            callback(message)

    def registerCallback(self, callback):
        """
        Đăng ký một hàm xử lý tin nhắn nhận được từ WebSocket.
        - callback: hàm nhận một tham số là chuỗi JSON tin nhắn
        """
        self.callbacks.append(callback)  # Thêm callback vào danh sách
