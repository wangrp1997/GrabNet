# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#

import sys
import os
sys.path.append('.')
sys.path.append('..')

import torch
import numpy as np
from psbody.mesh import Mesh, MeshViewers
from psbody.mesh.sphere import Sphere
from psbody.mesh.colors import name_to_rgb
from grabnet.tools.train_tools import point2point_signed
from grabnet.tools.utils import aa2rotmat
from grabnet.tools.utils import makepath
from grabnet.tools.utils import to_cpu


def vis_results(dorig, coarse_net, refine_net, rh_model, show_gen=True, show_rec=True, save=False, save_dir = None):

    with torch.no_grad():
        imw, imh = 400, 1000
        cols = len(dorig['bps_object'])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if show_rec:
            mvs = MeshViewers(window_width=imw * cols, window_height=imh, shape=[3, cols], keepalive=True)
            drec_cnet = coarse_net(**dorig)
            verts_rh_rec_cnet = rh_model(**drec_cnet).vertices

            _, h2o, _ = point2point_signed(verts_rh_rec_cnet, dorig['verts_object'])

            drec_cnet['trans_rhand_f'] = drec_cnet['transl']
            drec_cnet['global_orient_rhand_rotmat_f'] = aa2rotmat(drec_cnet['global_orient']).view(-1, 3, 3)
            drec_cnet['fpose_rhand_rotmat_f'] = aa2rotmat(drec_cnet['hand_pose']).view(-1, 15, 3, 3)
            drec_cnet['verts_object'] = dorig['verts_object']
            drec_cnet['h2o_dist']= h2o.abs()

            drec_rnet = refine_net(**drec_cnet)
            verts_rh_rec_rnet = rh_model(**drec_rnet).vertices

            for cId in range(0, len(dorig['bps_object'])):
                try:
                    from copy import deepcopy
                    meshes = deepcopy(dorig['mesh_object'])
                    obj_mesh = meshes[cId]
                except:
                    obj_mesh = points_to_spheres(points=to_cpu(dorig['verts_object'][cId]), radius=0.002, vc=name_to_rgb['green'])


                hand_mesh_orig = Mesh(v=to_cpu(dorig['verts_rhand'][cId]), f=rh_model.faces, vc=name_to_rgb['blue'])
                hand_mesh_rec_cnet= Mesh(v=to_cpu(verts_rh_rec_cnet[cId]), f=rh_model.faces, vc=name_to_rgb['green'])
                hand_mesh_rec_rnet = Mesh(v=to_cpu(verts_rh_rec_rnet[cId]), f=rh_model.faces, vc=name_to_rgb['red'])

                if 'rotmat' in dorig:
                    rotmat = dorig['rotmat'][cId].T
                    obj_mesh = obj_mesh.rotate_vertices(rotmat)
                    hand_mesh_orig.rotate_vertices(rotmat)
                    hand_mesh_rec_cnet.rotate_vertices(rotmat)
                    hand_mesh_rec_rnet.rotate_vertices(rotmat)

                hand_mesh_rec_cnet.reset_face_normals()
                hand_mesh_rec_rnet.reset_face_normals()
                hand_mesh_orig.reset_face_normals()

                mvs[0][cId].set_static_meshes([hand_mesh_orig, obj_mesh], blocking=True)
                mvs[1][cId].set_static_meshes([hand_mesh_rec_cnet, obj_mesh], blocking=True)
                mvs[2][cId].set_static_meshes([hand_mesh_rec_rnet, obj_mesh], blocking=True)

                if save:
                    save_path = os.path.join(save_dir, str(cId))
                    makepath(save_path)
                    hand_mesh_rec_rnet.write_ply(filename=save_path + '/rh_mesh_gen_%d.ply' % cId)
                    obj_mesh[0].write_ply(filename=save_path + '/obj_mesh_%d.ply' % cId)

        if show_gen:
            mvs = MeshViewers(window_width=imw * cols, window_height=imh, shape=[2, cols], keepalive=True)

            drec_cnet = coarse_net.sample_poses(dorig['bps_object'])
            verts_rh_gen_cnet = rh_model(**drec_cnet).vertices

            _, h2o, _ = point2point_signed(verts_rh_gen_cnet, dorig['verts_object'].to(device))

            drec_cnet['trans_rhand_f'] = drec_cnet['transl']
            drec_cnet['global_orient_rhand_rotmat_f'] = aa2rotmat(drec_cnet['global_orient']).view(-1, 3, 3)
            drec_cnet['fpose_rhand_rotmat_f'] = aa2rotmat(drec_cnet['hand_pose']).view(-1, 15, 3, 3)
            drec_cnet['verts_object'] = dorig['verts_object'].to(device)
            drec_cnet['h2o_dist']= h2o.abs()

            drec_rnet = refine_net(**drec_cnet)
            verts_rh_gen_rnet = rh_model(**drec_rnet).vertices


            for cId in range(0, len(dorig['bps_object'])):
                try:
                    from copy import deepcopy
                    meshes = deepcopy(dorig['mesh_object'])
                    obj_mesh = meshes[cId]
                except:
                    obj_mesh = points_to_spheres(to_cpu(dorig['verts_object'][cId]), radius=0.002, vc=name_to_rgb['green'])

                hand_mesh_gen_cnet = Mesh(v=to_cpu(verts_rh_gen_cnet[cId]), f=rh_model.faces, vc=name_to_rgb['pink'])
                hand_mesh_gen_rnet = Mesh(v=to_cpu(verts_rh_gen_rnet[cId]), f=rh_model.faces, vc=name_to_rgb['gray'])

                if 'rotmat' in dorig:
                    rotmat = dorig['rotmat'][cId].T
                    obj_mesh = obj_mesh.rotate_vertices(rotmat)
                    hand_mesh_gen_cnet.rotate_vertices(rotmat)
                    hand_mesh_gen_rnet.rotate_vertices(rotmat)

                hand_mesh_gen_cnet.reset_face_normals()
                hand_mesh_gen_rnet.reset_face_normals()

                mvs[0][cId].set_static_meshes([hand_mesh_gen_cnet, obj_mesh], blocking=True)
                mvs[1][cId].set_static_meshes([hand_mesh_gen_rnet, obj_mesh], blocking=True)

                if save:
                    save_path = os.path.join(save_dir, str(cId))
                    makepath(save_path)
                    hand_mesh_gen_rnet.write_ply(filename=save_path + '/rh_mesh_gen_%d.ply' % cId)
                    obj_mesh[0].write_ply(filename=save_path + '/obj_mesh_%d.ply' % cId)



def points_to_spheres(points, radius=0.1, vc=name_to_rgb['blue']):

    spheres = Mesh(v=[], f=[])
    for pidx, center in enumerate(points):
        clr = vc[pidx] if len(vc) > 3 else vc
        spheres.concatenate_mesh(Sphere(center, radius).to_mesh(color=clr))
    return spheres

def cage(length=1,vc=name_to_rgb['black']):

    cage_points = np.array([[-1., -1., -1.],
                            [1., 1., 1.],
                            [1., -1., 1.],
                            [-1., 1., -1.]])
    c = Mesh(v=length * cage_points, f=[], vc=vc)
    return c



def create_video(path, fps=30,name='movie'):
    import os
    import subprocess

    src = os.path.join(path,'%*.png')
    movie_path = os.path.join(path,'%s.mp4'%name)
    i = 0
    while os.path.isfile(movie_path):
        movie_path = os.path.join(path,'%s_%02d.mp4'%(name,i))
        i +=1


    cmd = 'ffmpeg -f image2 -r %d -i %s -b:v 6400k -pix_fmt yuv420p %s' % (fps, src, movie_path)
    subprocess.call(cmd.split(' '))
    while not os.path.exists(movie_path):
        continue

def visualize_two_meshes(mesh1_path, mesh2_path, window_width=1920, window_height=780):
    """
    同时可视化两个点云文件
    Args:
        mesh1_path: 第一个点云文件路径（手部网格）
        mesh2_path: 第二个点云文件路径（物体网格）
        window_width: 窗口宽度
        window_height: 窗口高度
    """
    print(f"正在加载文件: {mesh1_path} 和 {mesh2_path}")
    
    # 创建网格查看器
    print("创建可视化窗口...")
    mvs = MeshViewers(window_width=window_width, window_height=window_height, shape=[1, 1], keepalive=True)
    
    # 加载两个点云
    print("加载网格文件...")
    try:
        mesh1 = Mesh(filename=mesh1_path)
        print(f"成功加载第一个网格，顶点数: {len(mesh1.v)}")
        mesh2 = Mesh(filename=mesh2_path)
        print(f"成功加载第二个网格，顶点数: {len(mesh2.v)}")
    except Exception as e:
        print(f"加载网格文件时出错: {str(e)}")
        return
    
    # 设置颜色
    print("设置网格颜色...")
    mesh1.vc = name_to_rgb['pink']  # 手部网格设为粉色
    mesh2.vc = name_to_rgb['green']  # 物体网格设为绿色
    
    # 重置法线
    print("重置法线...")
    mesh1.reset_face_normals()
    mesh2.reset_face_normals()
    
    # 显示网格
    print("显示网格...")
    try:
        mvs[0][0].set_static_meshes([mesh1, mesh2], blocking=False)
        print("网格已显示，按 'q' 键退出")
        input("按回车键继续...")  # 等待用户输入
    except Exception as e:
        print(f"显示网格时出错: {str(e)}")

def visualize_two_meshes_open3d(mesh1_path, mesh2_path):
    """
    使用Open3D库实时可视化两个点云文件
    Args:
        mesh1_path: 第一个点云文件路径（手部网格）
        mesh2_path: 第二个点云文件路径（物体网格）
    """
    try:
        import open3d as o3d
    except ImportError:
        print("请先安装open3d: pip install open3d")
        return

    print(f"正在加载文件: {mesh1_path} 和 {mesh2_path}")
    
    # 加载网格
    print("加载网格文件...")
    try:
        mesh1 = o3d.io.read_triangle_mesh(mesh1_path)
        print(f"成功加载第一个网格，顶点数: {len(mesh1.vertices)}")
        mesh2 = o3d.io.read_triangle_mesh(mesh2_path)
        print(f"成功加载第二个网格，顶点数: {len(mesh2.vertices)}")
    except Exception as e:
        print(f"加载网格文件时出错: {str(e)}")
        return

    # 设置颜色
    print("设置网格颜色...")
    mesh1.paint_uniform_color([1, 0.41, 0.7])  # 粉色
    mesh2.paint_uniform_color([0, 0.8, 0])  # 绿色

    # 计算法线
    mesh1.compute_vertex_normals()
    mesh2.compute_vertex_normals()

    # 创建可视化窗口
    print("创建可视化窗口...")
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="网格可视化", width=1920, height=1080)

    # 添加网格到可视化器
    vis.add_geometry(mesh1)
    vis.add_geometry(mesh2)

    # 设置视角
    view_control = vis.get_view_control()
    view_control.set_zoom(0.8)
    view_control.set_front([0, 0, -1])
    view_control.set_lookat([0, 0, 0])
    view_control.set_up([0, -1, 0])

    # 设置渲染选项
    opt = vis.get_render_option()
    opt.mesh_show_back_face = True
    opt.background_color = np.array([0.1, 0.1, 0.1])
    opt.point_size = 1.0
    opt.light_on = True

    print("显示网格...")
    print("操作说明：")
    print("- 左键拖动：旋转视角")
    print("- 右键拖动：平移视角")
    print("- 滚轮：缩放")
    print("- Shift + 左键拖动：旋转模型")
    print("- Ctrl + 左键拖动：平移模型")
    print("- 按 'Q' 或 'ESC' 键退出")

    # 运行可视化
    vis.run()
    vis.destroy_window()
    print("可视化完成")

def visualize_two_meshes_trimesh(mesh1_path, mesh2_path):
    """
    使用trimesh库可视化两个点云文件
    Args:
        mesh1_path: 第一个点云文件路径（手部网格）
        mesh2_path: 第二个点云文件路径（物体网格）
    """
    try:
        import trimesh
        import trimesh.transformations as tf
    except ImportError:
        print("请先安装trimesh: pip install trimesh")
        return

    print(f"正在加载文件: {mesh1_path} 和 {mesh2_path}")
    
    # 加载网格
    print("加载网格文件...")
    try:
        mesh1 = trimesh.load(mesh1_path)
        print(f"成功加载第一个网格，顶点数: {len(mesh1.vertices)}")
        mesh2 = trimesh.load(mesh2_path)
        print(f"成功加载第二个网格，顶点数: {len(mesh2.vertices)}")
    except Exception as e:
        print(f"加载网格文件时出错: {str(e)}")
        return

    # 设置颜色
    print("设置网格颜色...")
    # 创建粉色和绿色的颜色数组
    pink = np.array([255, 105, 180, 255], dtype=np.uint8)
    green = np.array([0, 204, 0, 255], dtype=np.uint8)
    
    # 为每个面设置颜色
    mesh1.visual.face_colors = np.tile(pink, (len(mesh1.faces), 1))
    mesh2.visual.face_colors = np.tile(green, (len(mesh2.faces), 1))

    # 创建场景
    print("创建场景...")
    scene = trimesh.Scene([mesh1, mesh2])
    
    # 设置相机
    print("设置相机视角...")
    # 计算场景的边界框
    bounds = scene.bounds
    # 设置相机距离为边界框对角线长度的2倍
    distance = np.linalg.norm(bounds[1] - bounds[0]) * 2
    
    # 设置多个视角
    angles = [
        [0, 0],      # 正面
        [90, 0],     # 上方
        [0, 90],     # 侧面
        [45, 45]     # 斜45度
    ]
    
    print("渲染多个视角...")
    for i, (elev, azim) in enumerate(angles):
        # 设置相机位置
        scene.set_camera(elev=elev, azim=azim, distance=distance)
        
        # 渲染图片
        try:
            # 渲染图片
            png_path = f"mesh_view_{i}.png"
            # 渲染场景
            png = scene.save_image(resolution=(1920, 1080))
            # 保存图片
            with open(png_path, 'wb') as f:
                f.write(png)
            print(f"已保存视角 {i+1} 的图片: {png_path}")
        except Exception as e:
            print(f"保存视角 {i+1} 的图片失败: {str(e)}")
    
    print("可视化完成")
    print("请查看生成的PNG图片文件")

def visualize_two_meshes_matplotlib(mesh1_path, mesh2_path):
    """
    使用matplotlib库实时可视化两个点云文件
    Args:
        mesh1_path: 第一个点云文件路径（手部网格）
        mesh2_path: 第二个点云文件路径（物体网格）
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import trimesh
    except ImportError:
        print("请先安装必要的库:")
        print("pip install matplotlib trimesh")
        return

    print(f"正在加载文件: {mesh1_path} 和 {mesh2_path}")
    
    # 加载网格
    print("加载网格文件...")
    try:
        mesh1 = trimesh.load(mesh1_path)
        print(f"成功加载第一个网格，顶点数: {len(mesh1.vertices)}")
        mesh2 = trimesh.load(mesh2_path)
        print(f"成功加载第二个网格，顶点数: {len(mesh2.vertices)}")
    except Exception as e:
        print(f"加载网格文件时出错: {str(e)}")
        return

    # 创建图形
    print("创建图形...")
    plt.ion()  # 打开交互模式
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 渲染手部网格
    ax.plot_trisurf(
        mesh1.vertices[:, 0],
        mesh1.vertices[:, 1],
        mesh1.vertices[:, 2],
        triangles=mesh1.faces,
        color='pink',
        alpha=0.8
    )
    
    # 渲染物体网格
    ax.plot_trisurf(
        mesh2.vertices[:, 0],
        mesh2.vertices[:, 1],
        mesh2.vertices[:, 2],
        triangles=mesh2.faces,
        color='green',
        alpha=0.8
    )
    
    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # 设置标题
    ax.set_title('实时交互式查看 - 使用鼠标旋转视角')
    
    print("显示网格...")
    print("使用鼠标可以旋转、缩放视角")
    print("按 'q' 键退出")
    
    # 显示图形并保持窗口打开
    plt.show(block=True)
    
    print("可视化完成")

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print("使用方法: python vis_tools.py <mesh1_path> <mesh2_path>")
        sys.exit(1)
        
    mesh1_path = sys.argv[1]
    mesh2_path = sys.argv[2]
    
    if not os.path.exists(mesh1_path):
        print(f"错误：文件不存在 - {mesh1_path}")
        sys.exit(1)
    if not os.path.exists(mesh2_path):
        print(f"错误：文件不存在 - {mesh2_path}")
        sys.exit(1)
    
    print("开始可视化...")
    # 使用Open3D可视化
    visualize_two_meshes_open3d(mesh1_path, mesh2_path)
    print("可视化完成")
