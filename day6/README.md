1. 패키지 폴더를 이용해서 실행 시키기

 - xml 코드 추가할것. 
 
<buildtool_depend>ament_python</buildtool_depend>

<!-- ROS 2 핵심 라이브러리 -->
<depend>rclpy</depend>
<depend>std_msgs</depend>
<depend>geometry_msgs</depend>
<depend>sensor_msgs</depend>

<!-- 이미지 처리 및 변환 관련 -->
<depend>cv_bridge</depend>
<depend>message_filters</depend>

<!-- 좌표 변환(TF2) 관련 (매우 중요) -->
<depend>tf2_ros</depend>
<depend>tf2_geometry_msgs</depend>

<!-- 실행 시 필요한 파이썬 패키지 (선택 사항이나 권장) -->
<exec_depend>python3-numpy</exec_depend>
<exec_depend>python3-opencv</exec_depend>

 - setup 추가할것
 
  'position = .tt4_detection_edit:main' - 패키지 명 추가할것
  
 - 패키지 맨 위 폴더에 안에 같이 있는 가중치 복사 하기 
  ex) test_ws/src/package 중 test_ws밑에 src 옆에 가중피 .pt 파일 복사하기. 
  
 

2. 필요한것. 
localization launch 파일 실행
rviz initialization 하고 해당 코드를 실행해야 한다.
 
 명령어 
 

 명령어 실행할 때 ros2 run (패키지 명) position --ros-args -r /tf:=/robot4/tf -r /tf_static:=/robot4/tf_static 으로 할것.
 

3. 실행 결과

욜로를 실행해서 car가 위치되어 있는 map 상 좌표를 토픽으로 발행한다. 

토픽 발행
데이터 타입 : PointStamped
토픽 이름 : robot4/detected_car_pose
토픽 코드 줄 : 95
 
 
 
