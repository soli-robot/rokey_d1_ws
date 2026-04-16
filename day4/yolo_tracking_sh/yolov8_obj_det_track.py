import os
import sys
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage # 압축 이미지 메시지 타입
from std_msgs.msg import Bool # True/False 전송용
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np

class YOLOTracker(Node):
    def __init__(self, model):
        super().__init__('yolo_tracker')
        self.model = model
        self.bridge = CvBridge()

        # 1. QoS 설정 (영상 끊김 방지)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # 2. 압축 이미지 구독 (CompressedImage)
        self.subscription = self.create_subscription(
            CompressedImage,
            '/robot4/oakd/rgb/image_raw/compressed', # 토픽명 확인 필요
            self.listener_callback,
            qos_profile)

        # 3. 'car' 감지 결과 발행용 퍼블리셔
        self.car_detection_pub = self.create_publisher(Bool, '/car_detected', 10)

        self.classNames = model.names if hasattr(model, 'names') else ['Object']
        self.should_shutdown = False

    def listener_callback(self, msg):
        # 압축 이미지 디코딩
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return

        # 4. 이미지 리사이징 (704x704)
        img_resized = cv2.resize(img, (704, 704))

        # YOLO 트래킹 실행
        results = self.model.track(source=img_resized, stream=True, persist=True)
        
        car_found = False # 이번 프레임에서 car 감지 여부

        for r in results:
            if not hasattr(r, 'boxes') or r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                label_name = self.classNames[cls_id]
                conf = float(box.conf[0])
                track_id = int(box.id[0]) if box.id is not None else -1

                # 'car' 감지 체크 (모델의 클래스 이름이 'car'인지 확인)
                if label_name == 'car':
                    car_found = True

                # 시각화
                label = f"{label_name} ID:{track_id} {conf:.2f}"
                cv2.rectangle(img_resized, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img_resized, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 5. car 감지 여부 토픽 발행
        detection_msg = Bool()
        detection_msg.data = car_found
        self.car_detection_pub.publish(detection_msg)

        # 결과 화면 출력
        cv2.imshow("YOLOv8 704x704 Tracking", img_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.should_shutdown = True

def main():
    # 학습시킨 가중치 파일 경로
    model_path = "turtle_weight(train9).pt"

    if not os.path.exists(model_path):
        print(f"File not found: {model_path}")
        sys.exit(1)

    model = YOLO(model_path)

    rclpy.init()
    node = YOLOTracker(model)
    print("--- YOLO Tracking (Compressed & Resize 704) 시작 ---")

    try:
        while rclpy.ok() and not node.should_shutdown:
            rclpy.spin_once(node, timeout_sec=0.01)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
