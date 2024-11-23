from picamera2 import Picamera2
import cv2
import sys

try:
    # 初始化相机
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
    picam2.start()

    # 检查相机是否正常启动
    try:
        # 尝试捕获一帧，判断是否能正常工作
        test_frame = picam2.capture_array()
        if test_frame is None:
            print("相机启动失败，请检查相机连接或配置。")
            sys.exit(1)  # 退出程序
        print("相机已成功启动")
    except Exception as e:
        print(f"无法捕获图像，可能是相机未正确启动: {e}")
        sys.exit(1)  # 退出程序

    frame_count = 0
    frame_interval = 10  # 每隔10帧抽取一次

    try:
        while True:
            # 捕获当前帧
            frame = picam2.capture_array()

            if frame is None:
                print("无法捕获到画面，请检查相机是否正常运行。")
                break

            # 显示当前视频流
            cv2.imshow("Video Stream", frame)

            # 抽取帧并保存
            if frame_count % frame_interval == 0:
                cv2.imwrite(f"frame_{frame_count}.jpg", frame)
                print(f"Saved frame {frame_count}")

            frame_count += 1

            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        picam2.close()
        cv2.destroyAllWindows()

except Exception as e:
    print(f"发生错误：{e}")
    sys.exit(1)  # 退出程序
