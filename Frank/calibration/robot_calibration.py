import cv2
import numpy as np


# 假设我们有一个函数 get_robot_pose() 返回当前机械臂的位姿
# 返回的位姿是一个 4x4 的变换矩阵
def get_robot_pose():
    # 此处应为获取机械臂当前位姿的实际代码
    # 这是一个示例变换矩阵
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]], dtype=float)


# 假设我们有一个函数 capture_image() 捕捉当前相机的图像
def capture_image():
    # 此处应为捕捉相机图像的实际代码
    # 这是一个示例图像
    return np.ones((480, 640, 3), dtype=np.uint8) * 255


# 定义标定板规格
pattern_size = (9, 6)  # 棋盘格内角点的数量
square_size = 0.025  # 单个方格的大小，单位为米

# 准备数据
robot_poses = []
image_points = []
object_points = []

# 准备标定板的3D点
objp = np.zeros((np.prod(pattern_size), 3), dtype=np.float32)
objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
objp *= square_size

# 采集16组数据
for i in range(16):
    # 获取机械臂位姿
    robot_pose = get_robot_pose()
    robot_poses.append(robot_pose[:3, :])

    # 捕捉图像
    image = capture_image()

    # 转为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 查找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret:
        image_points.append(corners)
        object_points.append(objp)

        # 在图像上绘制角点
        cv2.drawChessboardCorners(image, pattern_size, corners, ret)
        cv2.imshow('Image', image)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# 标定相机
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None,
                                                                    None)

# 准备手眼标定的数据
# 机械臂运动的旋转和平移（旋转向量和平移向量）
R_gripper2base = [pose[:3, :3] for pose in robot_poses]
t_gripper2base = [pose[:3, 3] for pose in robot_poses]

# 相机观察到标定板的旋转和平移（旋转向量和平移向量）
R_target2cam = [cv2.Rodrigues(rvec)[0] for rvec in rvecs]
t_target2cam = [tvec for tvec in tvecs]

# 使用OpenCV的calibrateHandEye进行手眼标定
R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam)

# 打印结果
print("手眼标定结果：")
print("旋转矩阵 (R_cam2gripper): \n", R_cam2gripper)
print("平移向量 (t_cam2gripper): \n", t_cam2gripper)
