import os
import sys
import rclpy
import time
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np

#다른 노드(파일)에서 ros2 topic pub /robot4/web_detection std_msgs/msg/Bool "{data: true}" 명령을 보내거나
#코드상에서 해당 토픽을 발행하면 is_active가 True가 되어 감지를 시작합니다.

class YOLOTracker(Node):
    def __init__(self, model):
        super().__init__('yolo_tracker')
        self.model = model
        self.bridge = CvBridge()
        
        # 트래킹 활성화 상태 변수 (False로 시작)
        self.is_active = False
        self.last_msg_time = time.time()

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        # 1. 실행 명령을 받을 구독자 (다른 파일에서 True를 보내면 동작 시작)
        self.trigger_sub = self.create_subscription(
            Bool,
            '/robot4/web_detection',  #웹캠에서 받는 토픽 이름
            self.trigger_callback,
            10)

        # 2. 이미지 구독
        self.subscription = self.create_subscription(
            CompressedImage,
            '/robot4/oakd/rgb/image_raw/compressed',
            self.listener_callback,
            qos_profile)

        # 3. 결과 발행 퍼블리셔
        self.car_detection_pub = self.create_publisher(Bool, '/car_detected', 50)

        # 모델 클래스 이름 로드 확인
        self.classNames = model.names if hasattr(model, 'names') else {0: 'person', 2: 'car'} 
        self.should_shutdown = False

    def trigger_callback(self, msg):
        """명령 토픽을 받으면 상태를 업데이트합니다."""
        if msg.data is True:
            self.get_logger().info("Execution signal received. Starting YOLO tracking...")
            self.is_active = True
        else:
            self.get_logger().info("Stop signal received.")
            self.is_active = False

    def listener_callback(self, msg):
        # 1. 활성화 상태 체크
        if not self.is_active:
            return

        # 2. 이미지 디코딩
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return

        img_resized = cv2.resize(img, (704, 704))

        # 3. YOLO 트래킹 실행 (stream=True)
        results = self.model.track(source=img_resized, persist=True, stream=True, verbose=False)
        
        car_found = False

        # 4. 결과 순회 및 시각화
        for r in results:
            if r.boxes is None:
                continue
            
            for box in r.boxes:
                # --- [중요: 여기서 x1, y1, x2, y2를 정의합니다] ---
                # box.xyxy[0]은 [xmin, ymin, xmax, ymax] 형태의 텐서입니다.
                coords = box.xyxy[0].tolist() 
                x1, y1, x2, y2 = map(int, coords) # 정수형으로 변환하여 변수 할당
                
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label_name = self.classNames.get(cls_id, f"Id-{cls_id}")
                
                # 색상 결정 (car는 빨강, 나머지는 파랑)
                if label_name == 'car':
                    car_found = True
                    color = (0, 0, 255) # BGR: Red
                else:
                    color = (255, 0, 0) # BGR: Blue

                # 5. 화면에 그리기 (x1, y1, x2, y2가 위에서 정의되었으므로 에러가 나지 않습니다)
                cv2.rectangle(img_resized, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img_resized, f"{label_name} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 6. 결과 발행 로직 (1초 간격)
        current_time = time.time()
        if car_found and (current_time - self.last_msg_time >= 1.0):
            detection_msg = Bool()
            detection_msg.data = True
            self.car_detection_pub.publish(detection_msg)
            self.last_msg_time = current_time
            self.get_logger().info("Car detected! Result sent to /car_detected")

        # 7. 영상 표시
        cv2.imshow("YOLOv8 Tracking", img_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.should_shutdown = True

def main():
    model_path = "tt_y8_b8_p25_weight.pt"
    if not os.path.exists(model_path):
        print(f"File not found: {model_path}")
        sys.exit(1)

    model = YOLO(model_path)
    rclpy.init()
    node = YOLOTracker(model)
    
    print("--- YOLO Wait Mode (Waiting for /robot4/web_detection) ---")

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