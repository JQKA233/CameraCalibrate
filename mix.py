import cv2
from tkinter import Tk, Label, Button
from PIL import Image, ImageTk
import numpy as np
import glob

# 初始化全局变量
photo_image = None
file_index = 1  # 初始化文件索引

# 创建窗口
root = Tk()
root.title("相机实时图像")

# 创建一个 Label 用于显示图像
label = Label(root)
label.pack()

# 棋盘格的尺寸
chessboard_size = (6, 6)  # (内角点数-1) x (内角点数-1)
# 创建棋盘格的世界坐标系点，棋盘格的每个格点
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# 标准差用于亚像素角点检测
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# 拍照函数
def take_photo():
    global file_index
    ret, frame = cap.read()
    if ret:
        # 将图像转换为 tkinter 可以使用的格式
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=im)
        global photo_image
        photo_image = imgtk
        label.config(image=photo_image)
        label.image = photo_image  # 保持对图像的引用
        # 保存图像，文件名递增
        filename = f'image/{file_index}.jpg'
        cv2.imwrite(filename, frame)
        print(f"照片已保存为{filename}")
        file_index += 1  # 增加文件索引


def calibration_cam():
    # 存储所有图片的角点位置
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    # 准备读取图片
    images = glob.glob('image/*.jpg')  # 读取棋盘格图片

    # 检查图片路径
    print("图片路径：", images)

    # 遍历棋盘格图片
    for fname in images:
        img = cv2.imread(fname)
        cv2.imshow("frame", img)
        print(fname)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 寻找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        # 如果找到了角点，添加到列表中
        if ret == True:
            objpoints.append(objp)

            # 优化角点位置
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                             (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(corners)

            # 在图像上绘制角点
            cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)
        else:
            print(f"在图片 {fname} 中未检测到棋盘格角点")

    cv2.destroyAllWindows()

    # 检查有效图像数量
    if len(objpoints) == 0 or len(imgpoints) == 0:
        print("没有检测到足够的棋盘格角点，无法进行相机标定。")
    else:
        # 相机标定
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        # 打印结果
        print("相机矩阵：\n", mtx)
        print("畸变系数：\n", dist)
        print("旋转向量：\n", rvecs)
        print("平移向量：\n", tvecs)

        # 反投影误差
        total_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        print("总的反投影误差: {:.3f}".format(total_error / len(objpoints)))


# 创建拍照按钮
take_photo_button = Button(root, text="拍照", command=take_photo)
take_photo_button.pack()
calibration_button = Button(root, text="标定相机", command=calibration_cam)
calibration_button.pack()

# 开启摄像头
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    root.destroy()
    exit()


# 显示实时图像
def show_frame():
    global cap
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 转换到灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 寻找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        # 如果找到了角点，优化角点位置
        if ret:
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            frame_clone = frame
            # 绘制角点
            cv2.drawChessboardCorners(frame, chessboard_size, corners, ret)

        # 显示结果
        # cv2.imshow('Chessboard Corner Detection', frame_clone)

        im = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=im)
        global photo_image
        photo_image = imgtk
        label.config(image=photo_image)
        label.image = photo_image  # 保持对图像的引用
    root.after(10, show_frame)


# 启动实时图像显示
show_frame()

# 运行 Tkinter 事件循环
root.mainloop()

# 释放摄像头资源
cap.release()
