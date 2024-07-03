import numpy as np  # 导入NumPy库，用于高效的多维数组操作
import open3d as o3d  # 导入Open3D库，用于三维数据处理和可视化
import pyrealsense2 as rs  # 导入RealSense SDK的Python接口，用于操作RealSense相机
import torch  # 导入PyTorch，一个开源的机器学习库
from torchvision.transforms import transforms  # 导入torchvision库中的transforms模块，用于图像预处理
import cv2  # 导入OpenCV库，用于图像处理


# 定义初始化相机的函数
def initialize_camera():
    pipeline = rs.pipeline()  # 创建一个pipeline，用于管理RealSense数据流的配置和流动
    config = rs.config()  # 创建一个配置对象，用来配置pipeline
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # 启用深度流，分辨率640x480，16位，30帧/秒
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 启用彩色流，同样的分辨率和帧率
    profile = pipeline.start(config)  # 启动pipeline流，根据配置
    return pipeline, profile  # 返回pipeline对象和设备的配置信息


# 定义从RealSense相机获取深度图和彩色图的函数
def get_frames(pipeline):
    frames = pipeline.wait_for_frames()  # 从pipeline等待获取一帧数据
    depth_frame = frames.get_depth_frame()  # 获取深度帧
    color_frame = frames.get_color_frame()  # 获取彩色帧
    if not depth_frame or not color_frame:  # 如果没有获取到帧，则返回None
        return None, None
    depth_image = np.asanyarray(depth_frame.get_data())  # 将深度帧数据转换为NumPy数组
    color_image = np.asanyarray(color_frame.get_data())  # 将彩色帧数据转换为NumPy数组
    return depth_image, color_image  # 返回深度图和彩色图的数组


def visualize_tsdf_volume(tsdf_volume):
    # 从 TSDF 体积中提取三维网格
    mesh = tsdf_volume.extract_triangle_mesh()

    # 计算网格的顶点法线，以便在可视化时产生更好的光照效果
    mesh.compute_vertex_normals()

    # 可视化三维网格
    o3d.visualization.draw_geometries([mesh], window_name="3D Mesh Visualization", width=800, height=600)


# 定义创建TSDF体积的函数
def create_tsdf_volume(depth_image, color_image, rs_intrinsics):
    # 从pyrealsense2内参对象中提取参数
    width = rs_intrinsics.width
    height = rs_intrinsics.height
    fx = rs_intrinsics.fx
    fy = rs_intrinsics.fy
    cx = rs_intrinsics.ppx
    cy = rs_intrinsics.ppy

    # 创建Open3D的PinholeCameraIntrinsic对象
    o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    # 初始化TSDF体积，设置体素大小和截断距离
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.004,  # 体素长度
        sdf_trunc=0.02,  # 截断距离
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)  # 颜色类型
    # 创建RGBD图像
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color_image),
        o3d.geometry.Image(depth_image),
        depth_scale=1000.0,  # 深度比例
        depth_trunc=3.0,  # 深度截断值
        convert_rgb_to_intensity=False)  # 是否转换RGB到强度
    # 将RGBD图像数据融合到TSDF体积
    volume.integrate(
        rgbd_image,  # rgbd_image: 这是一个包含彩色和深度信息的RGBD图像对象，已从相应的深度和彩色图像创建
        o3d_intrinsics,  # 使用Open3D的PinholeCameraIntrinsic类将相机内参转换为Open3D需要的格式
        np.eye(4))  # np.eye(4): 创建一个4x4的单位矩阵，用作相机位姿。这里假设相机固定于原点，朝向Z轴正方向，没有旋转或平移。
    # visualize_tsdf_volume(volume)
    return volume  # 返回TSDF体积对象


# 定义加载训练好的抓取质量评估网络的函数
def load_grasp_model(model_path):
    model = torch.load(model_path)  # 加载模型
    model.eval()  # 设置为评估模式
    return model  # 返回模型


def extract_point_cloud(mesh):
    # 从网格中提取点云和法线
    pcd = mesh.sample_points_poisson_disk(number_of_points=500)  # 使用泊松磁盘采样法获取稠密点云
    pcd.estimate_normals()  # 计算法线
    return pcd


def evaluate_grasp_quality(model, point_cloud):
    # 将点云转换为模型需要的格式
    points = np.asarray(point_cloud.points)
    points = torch.tensor(points).float().unsqueeze(0)  # 假设模型需要batch dimension和浮点数

    # 运行模型进行预测
    model.eval()
    with torch.no_grad():
        output = model(points)
        grasp_quality = output.squeeze().item()  # 获取抓取质量评分
    return grasp_quality


def visualize_point_cloud(point_cloud):
    # 设置点云的颜色，可选步骤
    point_cloud.paint_uniform_color([1, 0.706, 0])  # 设置点云颜色为金黄色

    # 创建一个可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Point Cloud Visualization', width=800, height=600)

    # 将点云添加到可视化窗口
    vis.add_geometry(point_cloud)

    # 运行可视化窗口，直到用户关闭它
    vis.run()
    vis.destroy_window()
# 定义主函数
def main():
    pipeline, profile = initialize_camera()  # 初始化相机并获取配置
    # model = load_grasp_model('grasp_quality_model.pth')  # 加载模型

    try:
        while True:  # 创建一个循环，持续从相机获取数据并处理
            depth_image, color_image = get_frames(pipeline)  # 获取深度和彩色图像
            if depth_image is not None:  # 如果获取到图像
                tsdf_volume = create_tsdf_volume(depth_image, color_image, profile.get_stream(
                    rs.stream.depth).as_video_stream_profile().get_intrinsics())  # 创建TSDF体积
                mesh = tsdf_volume.extract_triangle_mesh()
                mesh.compute_vertex_normals()

                point_cloud = extract_point_cloud(mesh)
                grasp_quality = evaluate_grasp_quality(model, point_cloud)
                print(f"Grasp Quality: {grasp_quality}")
    finally:
        pipeline.stop()  # 停止pipeline


if __name__ == '__main__':
    main()  # 运行主函数
