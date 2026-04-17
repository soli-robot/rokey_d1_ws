## 🚀 터틀봇 4 비전 인식 시스템 실행 매뉴얼

### **Step 1: 웹캠 노드 실행 (이미지 송출)**
먼저 '눈'을 떠야 합니다. 웹캠 데이터를 우리가 약속한 토픽 이름으로 쏴주는 단계입니다.

* **터미널 1**
    ```bash
    # 1. 도메인 설정
    export ROS_DOMAIN_ID=4
    # 2. 웹캠 실행 (토픽 리매핑 필수!)
    ros2 run v4l2_camera v4l2_camera_node --ros-args \
    -p video_device:="/dev/video2" \
    -r /image_raw:=/robot4/webcam/image_raw
    ```
    > **체크**: 영상이 잘 나오는지 확인하려면 새 터미널에서 `ros2 run rqt_image_view rqt_image_view`를 켜서 `/robot4/webcam/image_raw`를 선택해 보세요.

---

### **Step 2: 워크스페이스 빌드 및 환경 적용**
우리가 작성한 코드가 최신 상태인지 확인하고 환경을 불러옵니다.

* **터미널 2**
    ```bash
    # 1. 워크스페이스 이동
    cd ~/webcam_detection_ws
    # 2. 빌드 (수정사항이 있다면 필수)
    colcon build --symlink-install --packages-select webcam_obj_det
    # 3. 환경 설정 적용
    source install/setup.bash
    ```

---

### **Step 3: YOLO 인식 및 언도킹 노드 실행**
이제 주인공인 '인식 노드'를 실행합니다.

* **터미널 2 (계속)**
    ```bash
    # 1. 도메인 설정 확인
    export ROS_DOMAIN_ID=4
    # 2. 노드 실행
    ros2 run webcam_obj_det detect_node
    ```
    > **작동 확인**: 
    > 1. 화면에 웹캠 영상 창이 뜨는지 확인.
    > 2. `car` (RC카)를 카메라 앞에 가져다 대기.
    > 3. 로그에 "★★★ car 발견!"이 뜨고 로봇이 실제로 언도킹을 시작하는지 확인.

---

### **Step 4: (선택) 통신 상태 모니터링**
만약 동작이 안 된다면, 신호가 가고 있는지 확인해야 합니다.

* **터미널 3**
    ```bash
    # 언도킹 완료 신호가 나가는지 모니터링
    export ROS_DOMAIN_ID=4
    ros2 topic echo /robot4/web_detection
    ```

### **1. 로봇의 도킹 상태 확인**
로봇이 충전 단자에 **물리적으로 딱 붙어 있어야** `undock()` 명령이 먹힙니다. 단자에서 살짝 떨어져 있으면 로봇은 "난 이미 나와 있어"라고 판단하고 아무것도 안 할 수 있으니, 꼭 손으로 밀어서 밀착시켜 주세요.

### **2. `/dev/video2` 확인**
PC에 웹캠이 여러 개 꽂혀 있거나 내장 카메라가 있다면 숫자가 `0`, `1`, `2` 중 무엇인지 확인해야 합니다. 
* `ls /dev/video*` 명령어로 장치 번호를 먼저 확인하는 습관을 들이면 좋습니다.

### **3. ROS_DOMAIN_ID 일치**
교실이나 실습실처럼 여러 대의 로봇이 있는 곳에서는 **로봇 본체와 내 PC의 도메인 ID가 반드시 같아야** 합니다.

카메라 데이터를 전송할 때, 데이터의 '무게'를 줄일 것인지 아니면 '순수함'을 유지할 것인지를 결정하는 물리적 선택입니다.
주요 파라미터 (Parameters)

    format: CompressedImage에서 주로 사용하며, jpeg나 png 형식을 지정합니다. (JPEG는 손실 압축으로 용량이 매우 작고, PNG는 무손실 압축으로 용량이 조금 더 큽니다.)

    data: Image는 픽셀 배열 그 자체를 보내고, CompressedImage는 압축된 바이트 스트림을 보냅니다.

    Reliability: 고용량인 Image를 보낼 때는 네트워크 부하를 줄이기 위해 Best Effort 설정을 고민하기도 합니다.

Image_Raw 를 쓰는 이유는 PC와 로봇이 동일한 고속 WiFi 망에 있고 영상 끊김이 없을 때.
YOLOv8 모델이 작은 물체나 미세한 특징을 잡아내야 할 때 (압축 노이즈 방지).
PC의 CPU 사양이 넉넉하여 압축 해제 루틴을 추가하고 싶지 않을 때.
