import cv2
from tkinter import Tk, Label, Button, Text, Scrollbar, messagebox, END
from PIL import Image, ImageTk
import numpy as np
import glob
import os
from tkinter import Frame

# 初始化全局变量
photo_image = None
file_index = 1  # 初始化文件索引

# 创建窗口
root = Tk()
root.title("相机实时图像")

# 设置窗口的最小大小
root.geometry('800x600')  # 可以根据需要调整这个大小

# 创建一个 Frame 用于放置按钮
button_frame = Frame(root)
button_frame.pack(side='bottom', fill='x')

# 创建拍照按钮
take_photo_button = Button(button_frame, text="拍照", command=lambda: take_photo())
take_photo_button.pack(side='left', padx=10, pady=10)
calibration_button = Button(button_frame, text="标定相机", command=lambda: calibration_cam())
calibration_button.pack(side='left', padx=10, pady=10)
clear_button = Button(button_frame, text="清空图像", command=lambda: clear_directory())
clear_button.pack(side='right', padx=10, pady=10)

# 创建一个 Label 用于显示图像
label = Label(root)
label.pack(expand=True, fill='both')

# 创建一个 Text 用于显示标定参数
text = Text(root, wrap='word', font=('Helvetica', 12))
text.pack(expand=True, fill='both', padx=10, pady=10)

# 创建滚动条
scrollbar = Scrollbar(root, command=text.yview)
scrollbar.pack(side='right', fill='y', padx=10)

# 配置 Text 控件使用滚动条
text.config(yscrollcommand=scrollbar.set)

# 棋盘格的尺寸
chessboard_size = (6, 6)  # (内角点数-1) x (内角点数-1)
# 创建棋盘格的世界坐标系点，棋盘格的每个格点
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# 标准差用于亚像素角点检测
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def clear_directory():
    directory_path = 'image'
    # 列出目录中的所有文件和目录
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            # 如果是文件，则删除
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
                print(f"已删除文件：{file_path}")
            # 如果是目录，则递归删除
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
                print(f"已删除目录：{file_path}")
        except Exception as e:
            print(f'无法删除 {file_path}。原因: {e}')


# 拍照函数
def take_photo():
    global file_index
    ret, frame = cap.read()
    if ret:
        im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
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


# 相机标定函数
def calibration_cam():
    # 存储所有图片的角点位置
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    # 准备读取图片
    images = glob.glob('image/*.jpg')  # 读取棋盘格图片

    # 遍历棋盘格图片
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 寻找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        # 如果找到了角点，添加到列表中
        if ret:
            objpoints.append(objp)

            # 优化角点位置
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)
        else:
            print(f"在图片 {fname} 中未检测到棋盘格角点")

    # 检查有效图像数量
    if len(objpoints) == 0 or len(imgpoints) == 0:
        messagebox.showerror("错误", "没有检测到足够的棋盘格角点，无法进行相机标定。")
    else:
        # 相机标定
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        # 打印结果到 Text 控件
        result_text = f"相机矩阵：\n{mtx}\n畸变系数：\n{dist}\n旋转向量：\n{rvecs}\n平移向量：\n{tvecs}"
        text.delete('1.0', END)  # 清空 Text 控件
        text.insert('1.0', result_text)  # 插入标定结果


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
            # 绘制角点
            cv2.drawChessboardCorners(frame, chessboard_size, corners, ret)

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
