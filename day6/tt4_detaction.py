#!/usr/bin/env python3

import time
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.duration import Duration

from sensor_msgs.msg import Image as ROSImage, CameraInfo, CompressedImage
from geometry_msgs.msg import PointStamped
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point

from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

import numpy as np
import cv2
import threading
from std_msgs.msg import Bool
from ultralytics import YOLO
from rclpy.time import Time
from message_filters import Subscriber, ApproximateTimeSynchronizer
from rclpy.callback_groups import ReentrantCallbackGroup


class DepthToMap(Node):
    def __init__(self):
        super().__init__('depth_to_map_node')

        self.get_logger().info("노드 초기화 시작...")

        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.model = YOLO('yolov8n.pt')
        self.K = None

        ns = '/robot4'
        self.depth_topic = f'{ns}/oakd/stereo/image_raw'
        self.rgb_topic = f'{ns}/oakd/rgb/image_raw/compressed'
        self.info_topic = f'{ns}/oakd/rgb/camera_info'
        self.trigger_topic = f'{ns}/detect_trigger'

        self.rgb_image = None
        self.depth_image = None
        self.camera_frame = None
        self.camera_stamp = None
        self.clicked_point = None

        self.waiting_for_frame = False
        self.trigger_time = None   # b. 추가: 트리거 받은 시각 저장

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.main_group = ReentrantCallbackGroup()

        self.video_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            durability=DurabilityPolicy.VOLATILE
        )

        self.trigger_sub = self.create_subscription(
            Bool,
            self.trigger_topic,
            self.trigger_callback,
            10
        )

        self.rgb_pub = self.create_publisher(ROSImage, f'{ns}/rgb_processed', 1)

        self.info_sub = self.create_subscription(
            CameraInfo,
            self.info_topic,
            self.camera_info_callback,
            1
        )

        # 트리거가 오기 전까지는 None
        self.rgb_sub = None
        self.depth_sub = None
        self.ts = None

        # b. 추가: 타임아웃 감시 타이머
        self.timeout_timer = self.create_timer(0.5, self.check_wait_timeout)

        self.get_logger().info("노드 초기화 완료. 트리거 대기 중...")

        self.goal_pub = self.create_publisher(
            PointStamped, 
            f'{ns}/detected_car_pose', 
            10
        )

    def camera_info_callback(self, msg):
        with self.lock:
            if self.K is None:
                self.K = np.array(msg.k).reshape(3, 3)
                self.get_logger().info(
                    f"Camera intrinsics loaded: "
                    f"fx={self.K[0,0]:.2f}, fy={self.K[1,1]:.2f}, "
                    f"cx={self.K[0,2]:.2f}, cy={self.K[1,2]:.2f}"
                )

        if self.info_sub is not None:
            self.destroy_subscription(self.info_sub)
            self.info_sub = None

    def trigger_callback(self, msg):
        self.get_logger().warn("트리거 콜백 실행")

        if not msg.data:
            return

        with self.lock:
            if self.waiting_for_frame:
                self.get_logger().warn("이미 프레임 수신 대기 중입니다. 이번 트리거는 무시합니다.")
                return

            self.waiting_for_frame = True
            self.trigger_time = time.time()   # b. 추가

        self.get_logger().info("🔥 트리거 수신: RGB/Depth 구독 시작")
        self.start_sync_subscribers()

    def check_wait_timeout(self):
        """ b. 추가: 프레임 대기 상태가 너무 오래 지속되면 자동 해제 """
        with self.lock:
            if self.waiting_for_frame and self.trigger_time is not None:
                if time.time() - self.trigger_time > 3.0:
                    self.get_logger().warn("프레임 수신 타임아웃. 대기 상태를 해제합니다.")
                    self.waiting_for_frame = False
                    self.trigger_time = None
                    self.stop_sync_subscribers()

    def start_sync_subscribers(self):
        if self.rgb_sub is not None or self.depth_sub is not None:
            self.get_logger().warn("이미 subscriber가 존재합니다.")
            return

        self.rgb_sub = Subscriber(
            self,
            CompressedImage,
            self.rgb_topic,
            qos_profile=self.video_qos,
            callback_group=self.main_group
        )

        self.depth_sub = Subscriber(
            self,
            ROSImage,
            self.depth_topic,
            qos_profile=self.video_qos,
            callback_group=self.main_group
        )

        self.ts = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=0.5
        )
        self.ts.registerCallback(self.synced_callback)

    def stop_sync_subscribers(self):
        try:
            if self.rgb_sub is not None and hasattr(self.rgb_sub, 'sub'):
                self.destroy_subscription(self.rgb_sub.sub)
        except Exception as e:
            self.get_logger().warn(f"RGB subscriber 해제 중 예외: {e}")

        try:
            if self.depth_sub is not None and hasattr(self.depth_sub, 'sub'):
                self.destroy_subscription(self.depth_sub.sub)
        except Exception as e:
            self.get_logger().warn(f"Depth subscriber 해제 중 예외: {e}")

        self.rgb_sub = None
        self.depth_sub = None
        self.ts = None

    def synced_callback(self, rgb_msg, depth_msg):
        self.get_logger().info("✅ 동기화된 RGB/Depth 프레임 수신")
        self.get_logger().info(f"실제 프레임 ID 확인: {depth_msg.header.frame_id}")

        try:
            np_arr = np.frombuffer(rgb_msg.data, np.uint8)
            rgb_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            depth_image = self.bridge.imgmsg_to_cv2(
                depth_msg,
                desired_encoding='passthrough'
            )

            if rgb_image is None:
                raise ValueError("RGB 이미지 디코딩 실패")

        except Exception as e:
            self.get_logger().error(f"이미지 변환 실패: {e}")
            with self.lock:
                self.waiting_for_frame = False
                self.trigger_time = None
            self.stop_sync_subscribers()
            return

        with self.lock:
            self.rgb_image = rgb_image
            self.depth_image = depth_image
            self.camera_frame = depth_msg.header.frame_id
            self.camera_stamp = depth_msg.header.stamp

        # 한 쌍 받았으면 더 이상 안 받음
        self.stop_sync_subscribers()

        with self.lock:
            self.waiting_for_frame = False
            self.trigger_time = None

        self.yolo_callback()

    def yolo_callback(self):
        self.get_logger().info("YOLO 추론 시작")

        with self.lock:
            if self.rgb_image is None or self.depth_image is None:
                self.get_logger().warn("이미지가 준비되지 않았습니다.")
                return

            input_frame = self.rgb_image.copy()
            depth_frame = self.depth_image.copy()
            frame_id = self.camera_frame
            stamp = self.camera_stamp

        try:
            results = self.model.predict(input_frame, conf=0.5, verbose=False)

            best_car = None
            best_conf = -1.0

            for r in results:
                for box in r.boxes:
                    class_id = int(box.cls[0])
                    label = self.model.names[class_id]
                    conf = float(box.conf[0])

                    if label == 'car' and conf > best_conf:
                        best_conf = conf
                        best_car = box

            if best_car is None:
                self.get_logger().warn("이 프레임에서 car를 찾지 못했습니다.")
                return

            c_x, c_y, w, h = best_car.xywh[0].tolist()
            self.clicked_point = (int(c_x), int(c_y))
            self.get_logger().info(
                f"🚗 car 감지: center=({int(c_x)}, {int(c_y)}), conf={best_conf:.2f}"
            )

            x1, y1, x2, y2 = map(int, best_car.xyxy[0].tolist())
            vis = input_frame.copy()
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(vis, self.clicked_point, 5, (0, 0, 255), -1)

            self.process_and_publish(vis, depth_frame, frame_id, stamp)

        except Exception as e:
            self.get_logger().error(f"YOLO 추론 에러: {e}")

    def process_and_publish(self, rgb_vis, depth, frame_id, stamp):
        self.get_logger().info(f"📍 좌표 변환 시작 (출발 프레임: {frame_id})")
        
        if self.K is None:
            self.get_logger().warn("CameraInfo가 없어 3D 변환 불가")
            return

        try:
            x, y = self.clicked_point
            h, w = depth.shape[:2]

            # 1. Depth 추출 (5x5 영역의 중앙값 사용)
            patch = depth[max(0, y-2):min(h, y+3), max(0, x-2):min(w, x+3)]
            valid = patch[(patch > 200) & (patch < 5000)]

            if valid.size == 0:
                self.get_logger().warn("유효한 depth 값이 없습니다.")
                return

            z = float(np.median(valid)) / 1000.0
            if not (0.2 < z < 5.0):
                self.get_logger().warn(f"목표물 범위 초과: z={z:.2f}m")
                return 

            # 2. 카메라 좌표계(XYZ) 계산
            fx, fy = self.K[0, 0], self.K[1, 1]
            cx, cy = self.K[0, 2], self.K[1, 2]
            X = (x - cx) * z / fx
            Y = (y - cy) * z / fy
            Z = z

            # 3. PointStamped 메시지 생성
            pt_camera = PointStamped()
            pt_camera.header.stamp = Time().to_msg()
            pt_camera.header.frame_id = frame_id # "oakd_rgb_camera_optical_frame"가 자동으로 들어감
            pt_camera.point.x, pt_camera.point.y, pt_camera.point.z = X, Y, Z

            # 4. 대상 프레임 설정 (확인된 odom 사용)
            target_frame = 'map' 

            # 5. 좌표 변환 수행
            pt_target = self.tf_buffer.transform(
                pt_camera,
                target_frame,
                timeout=Duration(seconds=1.0)
            )

            self.goal_pub.publish(pt_target)

            self.get_logger().info(
                f"✨ 변환 성공! [{target_frame} 기준]\n"
                f"   결과 좌표: X={pt_target.point.x:.2f}, Y={pt_target.point.y:.2f}, Z={pt_target.point.z:.2f}"
            )

            # 결과 이미지 발행 (시각화 확인용)
            rgb_msg = self.bridge.cv2_to_imgmsg(rgb_vis, encoding="bgr8")
            self.rgb_pub.publish(rgb_msg)

        except Exception as e:
            self.get_logger().error(f"❌ TF 변환 실패: {e}")

        self.get_logger().info("✅ 1회 사이클 완료\n")


def main():
    rclpy.init()
    node = DepthToMap()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()