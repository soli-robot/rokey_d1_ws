import time
import os
import sys
import rclpy
import threading
from queue import Queue
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
from pathlib import Path
import cv2
from std_msgs.msg import Bool

# 추가: 병렬 처리를 위한 콜백 그룹
from rclpy.callback_groups import ReentrantCallbackGroup
# ---------------------------------------------------------
# [체크 5] 라이브러리 의존성 (TurtleBot4 Specifics)
# ---------------------------------------------------------
# 'turtlebot4_navigation' 패키지가 설치되지 않은 일반 PC에서는 
# 이 코드를 import 하는 순간 에러가 발생합니다. 
# 로봇 제어 라이브러리가 설치되어 있는지 꼭 확인하세요.
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Navigator

class YOLOImageSubscriber(Node):
    def __init__(self, model):
        super().__init__('yolo_image_subscriber')
        
        # [수정] 병렬 처리가 가능하도록 그룹 설정
        self.callback_group = ReentrantCallbackGroup()
        
        self.model = model
        self.bridge = CvBridge()
        self.image_queue = Queue(maxsize=1)
        self.should_shutdown = False
        self.classNames = model.names if hasattr(model, 'names') else ['Object']
        self.display_image = None

        # 터틀봇4 네비게이터 (내부에서 자체 노드를 생성함)
        # ---------------------------------------------------------
        # [체크 2] 네임스페이스 (Namespace)
        # ---------------------------------------------------------
        # '/robot4'는 특정 로봇의 ID입니다. 다른 로봇 사용 시 이 부분을 일일이 수정해야 합니다.
        # self.declare_parameter('namespace', '/robot4') 처럼 파라미터화하는 것이 좋습니다.
        self.navigator = TurtleBot4Navigator(namespace='/robot4')
        self.is_undocked = False
        self.detection_pub = self.create_publisher(Bool, '/robot4/web_detection', 10)

        # [수정] 구독 설정 시 콜백 그룹 지정
        # ---------------------------------------------------------
        # [체크 3] 카메라 토픽 이름 (Topic Name)
        # ---------------------------------------------------------
        # 웹캠 드라이버나 카메라 종류가 바뀌면 토픽 이름이 달라집니다.
        # 특히 OAK-D 내장 카메라를 쓸지, 외부 USB 웹캠을 쓸지에 따라 경로를 확인해야 합니다.
        self.subscription = self.create_subscription(
            Image,
            '/robot4/webcam/image_raw',
            self.listener_callback,
            10,
            callback_group=self.callback_group)

        self.thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.thread.start()
        self.get_logger().info("YOLOv8 Detection 및 Navigator 노드가 시작되었습니다.")


    def listener_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if not self.image_queue.full():
                self.image_queue.put(img)
        except Exception as e:
            self.get_logger().error(f"이미지 변환 실패: {e}")

    def detection_loop(self):
        while rclpy.ok() and not self.should_shutdown:
            try:
                img = self.image_queue.get(timeout=0.5)
            except:
                continue

            # ---------------------------------------------------------
            # [체크 4] YOLO 추론 장치 (Device: CPU vs GPU)
            # ---------------------------------------------------------
            # 현재는 기본값입니다. NVIDIA GPU가 있는 노트북이면 'cuda'를, 
            # 라즈베리 파이 같은 장치라면 'cpu'나 'openvino' 최적화가 필요할 수 있습니다.
            # 예: self.model.to('cuda')
            results = self.model.predict(img, stream=True, conf=0.6, verbose=False)

            for r in results:
                if not hasattr(r, 'boxes') or r.boxes is None:
                    continue
                
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label_name = self.classNames[cls]

                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(img, f"{label_name} {conf:.2f}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # --- [수정] 터틀봇4 제어 로직: 별도 쓰레드로 분리 ---
                    if label_name == 'car' and not self.is_undocked:
                        self.get_logger().info(f"★★★ {label_name} 발견! 별도 쓰레드에서 언도킹 시작 ★★★")
                        # 중복 실행 방지를 위해 즉시 플래그 설정
                        self.is_undocked = True
                        # 핵심: Navigator 전용 쓰레드 실행 (충돌 방지)
                        threading.Thread(target=self.safe_undock_task, daemon=True).start()

            self.display_image = img

    def safe_undock_task(self):
        """Navigator 전용 태스크: 언도킹 완료 후 팀원에게 신호 전송"""
        try:
            if self.navigator.getDockedStatus():
                self.get_logger().info("로봇 도킹 확인. 언도킹 명령 전송...")
                self.navigator.undock()
                
                while not self.navigator.isTaskComplete():
                    time.sleep(0.1)
                
                self.get_logger().info("언도킹 작업 성공 완료!")

                # [추가] 언도킹이 완료된 시점에 True 신호 전송
                detection_msg = Bool()
                detection_msg.data = True
                self.detection_pub.publish(detection_msg)
                self.get_logger().info("Undoking Complete!!")
            else:
                self.get_logger().warn("이미 언도킹 상태입니다. 신호만 전송합니다.")
                # 이미 나와있는 상태여도 인식이 되었다면 신호를 보내 다음 단계 진행 유도
                detection_msg = Bool()
                detection_msg.data = True
                self.detection_pub.publish(detection_msg)

        except Exception as e:
            self.get_logger().error(f"안전 언도킹 작업 중 오류: {e}")
            self.is_undocked = False

    def update_window(self):
        if self.display_image is not None:
            cv2.imshow("YOLOv8 Detection", self.display_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.should_shutdown = True

def main():
    # ---------------------------------------------------------
    # [체크 1] 모델 파일 경로 (가장 빈번한 오류 발생 지점)
    # ---------------------------------------------------------
    # 다른 PC로 옮기면 사용자 폴더명(csg)이 달라지므로 실행이 안 됩니다.
    # 팁: 패키지 내 상대 경로를 사용하거나 실행 시 인자로 받으세요.
    model_path = '/home/csg/webcam_detection_ws/src/webcam_obj_det/models/best.pt'
    if not os.path.exists(model_path):
        sys.exit(1)
    model = YOLO(model_path)

    rclpy.init()
    node = YOLOImageSubscriber(model)

    try:
        while rclpy.ok() and node.context.ok() and not node.should_shutdown:
            try:
                rclpy.spin_once(node, timeout_sec=0.05)
                node.update_window()
            except IndexError:
                continue 
            except Exception as e:
                break
    except KeyboardInterrupt:
        pass
    finally:
        node.should_shutdown = True
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()