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

import numpy as np
import torch
import os
import argparse
import sys
import trimesh
sys.path.append('.')
sys.path.append('..')

import mano
# from grabnet.tools.vis_tools import vis_results
from grabnet.data.dataloader import LoadData
from grabnet.tools.cfg_parser import Config
from grabnet.train.trainer import Trainer
from lib.viztools.viz_o3d_utils import VizContext
from grabnet.tools.utils import to_cpu, aa2rotmat
from grabnet.tools.train_tools import point2point_signed

# 定义颜色映射
name_to_rgb = {
    'yellow': [1.0, 1.0, 0.0],
    'red': [1.0, 0.0, 0.0],
    'green': [0.0, 1.0, 0.0],
    'blue': [0.0, 0.0, 1.0],
    'pink': [1.0, 0.4, 0.7],
    'gray': [0.5, 0.5, 0.5]
}

def visualize_grasp(frame_data, coarse_net, refine_net, rh_model):
    """可视化抓取结果"""
    with torch.no_grad():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        viz_ctx = VizContext(non_block=True)
        viz_ctx.init()

        # 获取 coarse net 结果
        drec_cnet = coarse_net.sample_poses(frame_data['bps_object'])
        verts_rh_gen_cnet = rh_model(**drec_cnet).vertices
        
        # 计算手部和物体之间的距离
        _, h2o, _ = point2point_signed(verts_rh_gen_cnet, frame_data['verts_object'].to(device))
        drec_cnet['h2o_dist'] = h2o.abs()

        # 获取 refine net 结果
        drec_cnet['trans_rhand_f'] = drec_cnet['transl']
        drec_cnet['global_orient_rhand_rotmat_f'] = aa2rotmat(drec_cnet['global_orient']).view(-1, 3, 3)
        drec_cnet['fpose_rhand_rotmat_f'] = aa2rotmat(drec_cnet['hand_pose']).view(-1, 15, 3, 3)
        drec_cnet['verts_object'] = frame_data['verts_object'].to(device)
        drec_rnet = refine_net(**drec_cnet)
        verts_rh_gen_rnet = rh_model(**drec_rnet).vertices

        show_next = False
        def next_sample(_):
            nonlocal show_next
            show_next = True

        viz_ctx.register_key_callback('D', next_sample)
        print('按D键显示下一个样本')

        for cId in range(len(frame_data['bps_object'])):
            show_next = False
            obj_mesh = frame_data['mesh_object'][cId]
            obj_verts = obj_mesh.vertices
            obj_faces = obj_mesh.faces

            hand_mesh_gen_rnet = to_cpu(verts_rh_gen_rnet[cId])
            hand_mesh_gen_cnet = to_cpu(verts_rh_gen_cnet[cId])

            # 更新可视化
            viz_ctx.update_by_mesh('hand_rnet', hand_mesh_gen_rnet, rh_model.faces, vcolors=[0.7, 0.7, 0.7])
            viz_ctx.update_by_mesh('hand_cnet', hand_mesh_gen_cnet, rh_model.faces, vcolors=[1.0, 0.4, 0.7])
            viz_ctx.update_by_mesh('obj', obj_verts, obj_faces, vcolors=[0.2, 0.8, 0.2])

            while not show_next:
                viz_ctx.step()

        viz_ctx.deinit()

def inference(grabnet):
    grabnet.coarse_net.eval()
    grabnet.refine_net.eval()

    ds_name = 'test'
    mesh_base = os.path.expanduser('~/Projects/GrabNet/grabnet/data/grabnet_dataset/tools/object_meshes/contact_meshes')
    ds_test = LoadData(dataset_dir=grabnet.cfg.dataset_dir, ds_name=ds_name)
    n_samples = 5

    rh_model = mano.load(model_path=grabnet.cfg.rhm_path,
                         model_type='mano',
                         num_pca_comps=45,
                         batch_size=n_samples,
                         flat_hand_mean=True).to(grabnet.device)

    grabnet.refine_net.rhm_train = rh_model
    test_obj_names = np.unique(ds_test.frame_objs)

    grabnet.logger(f'################# \n'
                          f'Colors Guide:'
                          f'                   \n'
                          f'Red   --->  Reconstructed grasp - CoarseNet\n'
                          f'Green --->  Reconstructed grasp - Refinent\n'
                          f'Blue  --->  Ground Truth Grasp\n'
                          f'Pink  --->  Generated grasp - CoarseNet\n'
                          f'Gray  --->  Generated grasp - RefineNet\n')

    for obj in test_obj_names:
        obj_frames = np.where(ds_test.frame_objs == obj)[0]
        rnd_frames = np.random.choice(obj_frames.shape[0], n_samples)
        obj_data = ds_test[obj_frames[rnd_frames]]
        frame_data = {k: obj_data[k].to(grabnet.device) for k in obj_data.keys()}
        obj_meshes = []
        rotmats = []
        for frame in range(n_samples):
            rot_mat = frame_data['root_orient_obj_rotmat'][frame].cpu().numpy().reshape(3, 3).T
            transl = frame_data['trans_obj'][frame].cpu().numpy()

            # 使用 trimesh 替代 psbody.mesh
            obj_mesh = trimesh.load(os.path.join(mesh_base, obj + '.ply'))
            # 应用旋转
            obj_mesh.vertices = np.dot(obj_mesh.vertices, rot_mat)
            # 应用平移
            obj_mesh.vertices += transl
            # 设置颜色
            obj_mesh.visual.vertex_colors = np.tile(np.array(name_to_rgb['yellow']), (len(obj_mesh.vertices), 1))

            obj_meshes.append(obj_mesh)
            rotmats.append(rot_mat)

        frame_data['mesh_object'] = obj_meshes
        frame_data['rotmat'] = rotmats
        save_dir = os.path.join(grabnet.cfg.work_dir, 'test_grasp_results')
        grabnet.logger(f'#################\n'
                              f'                   \n'
                              f'Showing results for the {obj.upper()}'
                              f'                      \n')
        visualize_grasp(frame_data, grabnet.coarse_net, grabnet.refine_net, rh_model)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GrabNet-Testing')

    parser.add_argument('--data-path', default = None, type=str,
                        help='The path to the folder that contains GrabNet data')

    parser.add_argument('--rhm-path', default = None, type=str,
                        help='The path to the folder containing MANO_RIHGT model')

    parser.add_argument('--config-path', default = None, type=str,
                        help='The path to the confguration of the trained GrabNet model')

    args = parser.parse_args()

    cfg_path = args.config_path
    data_path = args.data_path
    rhm_path = args.rhm_path


    cwd = os.getcwd()

    best_cnet = 'grabnet/models/coarsenet.pt'
    best_rnet = 'grabnet/models/refinenet.pt'
    vpe_path  = 'grabnet/configs/verts_per_edge.npy'
    c_weights_path = 'grabnet/configs/rhand_weight.npy'
    work_dir = cwd + '/tests'

    if cfg_path is None:
        cfg_path = 'grabnet/configs/grabnet_cfg.yaml'

    config = {
        'work_dir':work_dir,
        'vpe_path': vpe_path,
        'c_weights_path': c_weights_path,

    }

    cfg = Config(default_cfg_path=cfg_path, **config)

    if data_path is not None:
        cfg['dataset_dir'] = data_path
    if rhm_path is not None:
        cfg['rhm_path'] = rhm_path
    if cfg.best_cnet is  None:
        cfg['best_cnet'] = best_cnet
    if cfg.best_rnet is None:
        cfg['best_rnet'] = best_rnet

    grabnet = Trainer(cfg=cfg, inference=True)
    inference(grabnet)

