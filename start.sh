#!/bin/bash
# =============================================================
#  MiR Robot Project - Khởi động 1 lệnh duy nhất
#  Cách dùng: ./start.sh [tuỳ chọn]
#    ./start.sh                    -> Khởi động tất cả (ROS1 + Bridge)
#    ./start.sh build              -> Build lại image rồi khởi động
#    ./start.sh stop               -> Dừng tất cả container
#    ./start.sh shell              -> Mở terminal vào container ROS1
#    ./start.sh gazebo             -> Chạy Gazebo simulation
#    ./start.sh joystick [IP]      -> Điều khiển MiR thật bằng tay cầm
#    ./start.sh roslaunch pkg file -> Chạy bất kỳ roslaunch nào
#    ./start.sh rviz [config]       -> Mở RViz (config: navigation|description|test)
#    ./start.sh run <file.py> [args] -> Chạy file Python trong thư mục tm
#    ./start.sh run list             -> Liệt kê các file Python trong thư mục tm
#    ./start.sh run-host <file.py> [args] -> Chạy file Python trên host (không qua Docker)
#    ./start.sh run-host list             -> Liệt kê file Python tm (host)
# =============================================================

set -e
cd "$(dirname "$0")"

# Cho phép Docker hiển thị GUI
xhost +local:docker 2>/dev/null || true

case "${1:-start}" in
    build)
        echo "🔨 Đang build lại Docker images..."
        docker compose build
        echo "🚀 Đang khởi động containers..."
        docker compose up -d
        echo ""
        echo "✅ Đã khởi động xong!"
        echo "   - mir_noetic_env: ROS1 Noetic (MiR Robot)"
        echo "   - mir_ros1_bridge: ROS1 <-> ROS2 Bridge"
        echo ""
        echo "📌 Để vào container ROS1:  ./start.sh shell"
        echo "📌 Để chạy Gazebo:         ./start.sh gazebo"
        ;;

    start)
        echo "🚀 Đang khởi động containers..."
        docker compose up -d
        echo ""
        echo "✅ Đã khởi động xong!"
        echo "   - mir_noetic_env: ROS1 Noetic (MiR Robot)"
        echo "   - mir_ros1_bridge: ROS1 <-> ROS2 Bridge"
        echo ""
        echo "📌 Để vào container ROS1:  ./start.sh shell"
        echo "📌 Để chạy Gazebo:         ./start.sh gazebo"
        echo ""
        echo "📡 Từ host ROS2 Humble, bạn có thể:"
        echo "   ros2 topic list          -> Xem topic từ ROS1"
        echo "   ros2 topic echo /scan    -> Đọc dữ liệu laser"
        ;;

    stop)
        echo "🛑 Đang dừng tất cả containers..."
        docker compose down
        echo "✅ Đã dừng."
        ;;

    shell)
        echo "🔧 Đang mở terminal vào container ROS1..."
        docker exec -it mir_noetic_env bash
        ;;

    gazebo)
        echo "🌍 Đang khởi động Gazebo MiR Robot..."
        docker exec -it mir_noetic_env bash -c \
            "source /opt/ros/noetic/setup.bash && \
             source /root/catkin_ws/devel/setup.bash && \
             roslaunch mir_gazebo mir_maze_world.launch"
        ;;

    roslaunch)
        # Chạy bất kỳ roslaunch command nào
        # Ví dụ: ./start.sh roslaunch mir_navigation amcl.launch
        shift
        echo "🚀 Đang chạy: roslaunch $*"
        docker exec -it mir_noetic_env bash -c \
            "source /opt/ros/noetic/setup.bash && \
             source /root/catkin_ws/devel/setup.bash && \
             roslaunch $*"
        ;;

    joystick)
        # Điều khiển MiR thật bằng tay cầm (KẾT NỐI TRỰC TIẾP - không cần mir_bridge)
        # Ví dụ: ./start.sh joystick 192.168.0.177
        MIR_IP="${2:-192.168.12.20}"
        echo "🎮 Đang kết nối Joystick TRỰC TIẾP đến MiR tại $MIR_IP..."
        echo ""
        echo "📌 Hướng dẫn:"
        echo "   - GIỮ NÚT R1/RB trước khi đẩy cần gạt"
        echo "   - Cần gạt trái: Điều khiển tiến/lùi/xoay"
        echo "   - Nút Y: Tăng tốc   |   Nút A: Giảm tốc"
        echo "   - Nhả R1 = Robot dừng ngay"
        echo ""
        docker exec -it mir_noetic_env bash -c \
            "source /opt/ros/noetic/setup.bash && \
             source /root/catkin_ws/devel/setup.bash && \
             roslaunch mir_driver mir_joystick_direct.launch mir_hostname:=$MIR_IP"
        ;;

    rviz)
        # Mở RViz - tuỳ chọn config: navigation | description | test
        # Ví dụ: ./start.sh rviz
        #         ./start.sh rviz navigation
        #         ./start.sh rviz description
        RVIZ_CONFIG=""
        case "${2:-}" in
            navigation)
                RVIZ_CONFIG="-d /root/catkin_ws/src/mir_robot/mir_navigation/rviz/navigation.rviz"
                ;;
            description)
                RVIZ_CONFIG="-d /root/catkin_ws/src/mir_robot/mir_description/rviz/mir_description.rviz"
                ;;
            test)
                RVIZ_CONFIG="-d /root/catkin_ws/src/mir_robot/tm/rviz/testrviz.rviz"
                ;;
        esac
        echo "🖥️  Đang mở RViz${2:+ (config: $2)}..."
        docker exec -it mir_noetic_env bash -c \
            "source /opt/ros/noetic/setup.bash && \
             source /root/catkin_ws/devel/setup.bash && \
             rosrun rviz rviz $RVIZ_CONFIG"
        ;;

    run)
        # Chạy file Python trong thư mục tm
        # Ví dụ: ./start.sh run navigationcacdiem.py bep
        #         ./start.sh run navigationcacdiem.py "ban 1"
        #         ./start.sh run list
        TM_DIR="/root/catkin_ws/src/mir_robot/tm"
        if [[ -z "${2:-}" || "${2}" == "list" ]]; then
            echo "📂 Các file Python trong thư mục tm:"
            docker exec mir_noetic_env bash -c \
                "ls ${TM_DIR}/*.py 2>/dev/null | xargs -I{} basename {}"
        else
            SCRIPT="${2}"
            shift 2
            ARGS="$*"
            echo "🐍 Đang chạy: python3 ${SCRIPT} ${ARGS}"
            docker exec -it mir_noetic_env bash -c \
                "source /opt/ros/noetic/setup.bash && \
                 source /root/catkin_ws/devel/setup.bash && \
                 pkill -f '^python3[[:space:]]+${SCRIPT}([[:space:]]|$)' 2>/dev/null || true && \
                 sleep 0.3 && \
                 export FORCE_PULSE_CAPTURE=1 && \
                 cd ${TM_DIR} && python3 ${SCRIPT} ${ARGS}"
        fi
        ;;

    run-host)
        # Chạy file Python trên host để dùng trực tiếp camera/GPU của máy
        TM_DIR_HOST="$(pwd)/src/mir_robot/tm"
        if [[ -z "${2:-}" || "${2}" == "list" ]]; then
            echo "📂 Các file Python trong thư mục tm (host):"
            ls "${TM_DIR_HOST}"/*.py 2>/dev/null | xargs -I{} basename {}
        else
            SCRIPT="${2}"
            shift 2
            ARGS="$*"
            if [[ ! -f "${TM_DIR_HOST}/${SCRIPT}" ]]; then
                echo "❌ Không tìm thấy file: ${TM_DIR_HOST}/${SCRIPT}"
                exit 1
            fi

            PYTHON_BIN="python3"
            if [[ -x "$(pwd)/.venv-gpu/bin/python" ]]; then
                PYTHON_BIN="$(pwd)/.venv-gpu/bin/python"
            elif [[ -x "$(pwd)/.venv/bin/python" ]]; then
                PYTHON_BIN="$(pwd)/.venv/bin/python"
            fi

            echo "🐍 Đang chạy host: ${PYTHON_BIN} ${SCRIPT} ${ARGS}"
            echo "📌 Gợi ý GPU: KHOANGCACH_DEVICE=gpu ./start.sh run-host ${SCRIPT}"
            export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
            export QT_QPA_FONTDIR="${QT_QPA_FONTDIR:-/usr/share/fonts/truetype/dejavu}"
            export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION:-none}"
            export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
            export OPENCV_FFMPEG_CAPTURE_OPTIONS="${OPENCV_FFMPEG_CAPTURE_OPTIONS:-video_codec;rawvideo}"
            export PYTHONUNBUFFERED=1
            cd "${TM_DIR_HOST}" && exec "${PYTHON_BIN}" "${SCRIPT}" ${ARGS}
        fi
        ;;

    joystick-bridge)
        # Điều khiển qua mir_bridge (cách cũ - cần mir_bridge)
        MIR_IP="${2:-192.168.12.20}"
        echo "🎮 Đang kết nối Joystick qua mir_bridge đến MiR tại $MIR_IP..."
        docker exec -it mir_noetic_env bash -c \
            "source /opt/ros/noetic/setup.bash && \
             source /root/catkin_ws/devel/setup.bash && \
             roslaunch mir_driver mir_joystick_teleop.launch mir_hostname:=$MIR_IP"
        ;;

    *)
        echo "Cách dùng: ./start.sh [start|build|stop|shell|gazebo|rviz|joystick|roslaunch|run|run-host ...]"
        echo ""
        echo "Ví dụ chạy Python:"
        echo "   ./start.sh run list"
        echo "   ./start.sh run navigationcacdiem.py bep"
        echo "   ./start.sh run navigationcacdiem.py ban1"
        exit 1
        ;;
esac
