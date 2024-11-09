import cv2
import numpy as np
import glob

# 棋盘格的尺寸
chessboard_size = (6, 6)  # (内角点数-1) x (内角点数-1)
# 创建棋盘格的世界坐标系点，棋盘格的每个格点
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# 存储所有图片的角点位置
object_points = []  # 3d point in real world space
image_points = []  # 2d points in image plane.

# 准备读取图片
images = glob.glob('image/*.jpg')  # 读取棋盘格图片

# 检查图片路径
print("图片路径：", images)

# 遍历棋盘格图片
for file_name in images:
    img = cv2.imread(file_name)
    cv2.imshow("frame", img)
    cv2.waitKey(0)
    print(file_name)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 寻找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # 如果找到了角点，添加到列表中
    if ret:
        object_points.append(objp)

        # 优化角点位置
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        image_points.append(corners)

        # 在图像上绘制角点
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)
    else:
        print(f"在图片 {file_name} 中未检测到棋盘格角点")

cv2.destroyAllWindows()

# 检查有效图像数量
if len(object_points) == 0 or len(image_points) == 0:
    print("没有检测到足够的棋盘格角点，无法进行相机标定。")
else:
    # 相机标定
    ret, mtx, dist, rotate_vecs, translate_vecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)

    # 打印结果
    print("相机矩阵：\n", mtx)
    print("畸变系数：\n", dist)
    print("旋转向量：\n", rotate_vecs)
    print("平移向量：\n", translate_vecs)

    # 反投影误差
    total_error = 0
    for i in range(len(object_points)):
        imgpoints2, _ = cv2.projectPoints(object_points[i], rotate_vecs[i], translate_vecs[i], mtx, dist)
        error = cv2.norm(image_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    print("总的反投影误差: {:.3f}".format(total_error / len(object_points)))
