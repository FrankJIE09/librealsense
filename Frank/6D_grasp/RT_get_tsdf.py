import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import cv2
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
def initialize_camera():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    return pipeline, profile

def get_frames(pipeline):
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        return None, None
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    return depth_image, color_image

def create_tsdf_volume(depth_image, color_image, camera_intrinsics):
    width, height, fx, fy, cx, cy = camera_intrinsics.width, camera_intrinsics.height, camera_intrinsics.fx, camera_intrinsics.fy, camera_intrinsics.ppx, camera_intrinsics.ppy
    o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.004,
        sdf_trunc=0.2,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color_image),
        o3d.geometry.Image(depth_image),
        depth_scale=1000.0,
        depth_trunc=1.0,
        convert_rgb_to_intensity=False)
    volume.integrate(rgbd_image, o3d_intrinsics, np.eye(4))
    return volume

def main():
    pipeline, profile = initialize_camera()
    vis = o3d.visualization.Visualizer()
    vis.create_window("3D Mesh Visualization", width=800, height=600)
    mesh = None
    pcd = None  # Initialize variable for point cloud

    try:
        while True:
            depth_image, color_image = get_frames(pipeline)
            if depth_image is not None and color_image is not None:
                intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
                tsdf_volume = create_tsdf_volume(depth_image, color_image, intrinsics)

                if mesh is not None:
                    vis.remove_geometry(mesh)
                if pcd is not None:
                    vis.remove_geometry(pcd)

                mesh = tsdf_volume.extract_triangle_mesh()
                mesh.compute_vertex_normals()
                pcd = mesh.sample_points_poisson_disk(number_of_points=1000)  # Sample point cloud from mesh
                pcd.estimate_normals()
                # visualize_point_cloud(pcd)

                vis.add_geometry(mesh)
                vis.add_geometry(pcd)  # Add point cloud to visualization
                vis.update_geometry(mesh)
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()

                cv2.imshow("Depth Image", cv2.convertScaleAbs(depth_image, alpha=0.03))
                cv2.imshow("Color Image", color_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        vis.destroy_window()

if __name__ == '__main__':
    main()
