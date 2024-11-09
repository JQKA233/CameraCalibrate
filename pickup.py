import cv2
import numpy as np
import glob

# 拍照并保存
def take_photo(save_path):
    cap = cv2.VideoCapture(0)  # 参数0通常表示系统的默认摄像头
    if not cap.isOpened():
        print("无法打开摄像头")
        return False

    ret, frame = cap.read()
    if ret:
        cv2.imwrite(save_path, frame)
        print(f"照片已保存为{save_path}")
        return True
    else:
        print("拍照失败")
        return False

    cap.release()

# 相机标定
def calibrate_camera(objpoints, imgpoints, images):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    w, h = 9, 6
    objp = np.zeros((w * h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("ret:", ret)
    print("mtx:\n", mtx)
    print("dist:\n", dist)
    print("rvecs:\n", rvecs)
    print("tvecs:\n", tvecs)

# 主程序
def main():
    # 拍照并保存
    photo_path = 'calibration_image.jpg'
    if take_photo(photo_path):
        images = glob.glob('calibration_image.jpg')  # 只使用刚拍摄的照片进行标定
        objpoints = []  # 在世界坐标系中的三维点
        imgpoints = []  # 在图像平面的二维点

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
            if ret:
                cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                objpoints.append(objp)
                imgpoints.append(corners)

        # 至少需要一对角点进行标定
        if len(objpoints) > 0 and len(imgpoints) > 0:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            calibrate_camera(objpoints, imgpoints, images)

if __name__ == "__main__":
    take_photo('image/1.jpg')