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

# 터틀봇4 네비게이션 라이브러리 추가
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Navigator

class YOLOImageSubscriber(Node):
    def __init__(self, model):
        super().__init__('yolo_image_subscriber')
        self.model = model
        self.bridge = CvBridge()
        self.image_queue = Queue(maxsize=1)
        self.should_shutdown = False
        self.classNames = model.names if hasattr(model, 'names') else ['Object']

        # 터틀봇4 네비게이터 및 상태 변수 초기화
        self.navigator = TurtleBot4Navigator()
        self.is_undocked = False  # 중복 실행 방지 플래그

        # 이미지 구독 (터틀봇4 OAK-D 카메라 토픽)
        self.subscription = self.create_subscription(
            Image,
            '/robot4/oakd/rgb/preview/image_raw',
            self.listener_callback,
            10)

        # 추론 쓰레드 시작
        self.thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.thread.start()
        self.get_logger().info("YOLOv8 Detection 및 Navigator 노드가 시작되었습니다.")

    def listener_callback(self, msg):
        # 이 줄을 추가해서 터미널에 메시지가 뜨는지 확인합니다.
        self.get_logger().info("이미지 한 장 도착!")
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if not self.image_queue.full():
                self.image_queue.put(img)
        except Exception as e:
            self.get_logger().error(f"이미지 변환 실패: {e}")

    def detection_loop(self):
        while not self.should_shutdown:
            try:
                img = self.image_queue.get(timeout=0.5)
            except:
                continue

            # 추론 수행 (stream=True로 메모리 효율 최적화)
            results = self.model.predict(img, stream=True, conf=0.6)

            for r in results:
                if not hasattr(r, 'boxes') or r.boxes is None:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0]) if box.cls is not None else 0
                    conf = float(box.conf[0]) if box.conf is not None else 0.0
                    label_name = self.classNames[cls]

                    # --- 터틀봇4 액션 제어 로직 ---
                    # 인식된 객체가 'RC_car'이고 아직 언도킹 전이라면 실행
                    if label_name == 'RC_car' and not self.is_undocked:
                        self.get_logger().info("★★★ RC카 발견! 언도킹 액션을 시작합니다. ★★★")
                        
                        # 도킹 상태 확인 후 언도킹 실행
                        if self.navigator.getDockedStatus():
                            self.navigator.undock()
                            self.is_undocked = True # 실행 완료 후 플래그 고정
                            self.get_logger().info("언도킹 완료.")
                        else:
                            self.get_logger().warn("이미 언도킹 상태이거나 도크를 찾을 수 없습니다.")

                    # 시각화
                    display_label = f"{label_name} {conf:.2f}"
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(img, display_label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("YOLOv8 Detection", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.get_logger().info("사용자 요청(q)으로 종료합니다.")
                self.should_shutdown = True
                break

def main():
    model_path = '/home/csg/webcam_detection_ws/src/webcam_obj_det/models/best.pt'

    if not os.path.exists(model_path):
        print(f"가중치 파일을 찾을 수 없습니다: {model_path}")
        sys.exit(1)

    # 모델 로드
    suffix = Path(model_path).suffix.lower()
    if suffix == '.pt':
        model = YOLO(model_path)
    elif suffix in ['.onnx', '.engine']:
        model = YOLO(model_path, task='detect')
    else:
        print(f"지원하지 않는 모델 형식입니다: {suffix}")
        sys.exit(1)

    rclpy.init()
    node = YOLOImageSubscriber(model)

    try:
        while rclpy.ok() and not node.should_shutdown:
            rclpy.spin_once(node, timeout_sec=0.01)
    except KeyboardInterrupt:
        pass
    finally:
        node.should_shutdown = True
        node.thread.join(timeout=1.0)
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()
        print("시스템이 정상 종료되었습니다.")

if __name__ == '__main__':
    main()