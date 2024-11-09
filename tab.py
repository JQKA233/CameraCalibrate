import cv2
from tkinter import Tk, Label, Button
from PIL import Image, ImageTk

# 初始化全局变量
photo_image = None
file_index = 1  # 初始化文件索引

# 创建窗口
root = Tk()
root.title("相机实时图像")

# 创建一个 Label 用于显示图像
label = Label(root)
label.pack()


# 拍照函数
def take_photo():
    global file_index
    ret, frame = cap.read()
    if ret:
        # 将图像转换为 tkinter 可以使用的格式
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (640, 480))
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


# 创建拍照按钮
button = Button(root, text="拍照", command=take_photo)
button.pack()

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
        frame = cv2.resize(frame, (640, 480))
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
