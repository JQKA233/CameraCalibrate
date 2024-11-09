import cv2
import numpy as np

# 棋盘格的尺寸
chessboard_size = (5, 6)  # (内角点数-1) x (内角点数-1)
# 创建棋盘格的世界坐标系点，棋盘格的每个格点
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# 标准差用于亚像素角点检测
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 初始化摄像头
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

while True:
    # 读取摄像头帧
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头帧")
        break

    # 转换到灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 寻找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # 如果找到了角点，优化角点位置
    if ret:
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # 绘制角点
        cv2.drawChessboardCorners(frame, chessboard_size, corners, ret)

    # 显示结果
    cv2.imshow('Chessboard Corner Detection', frame)

    # 按 'q' 退出循环
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# 释放摄像头资源
cap.release()
# 关闭所有 OpenCV 窗口
cv2.destroyAllWindows()