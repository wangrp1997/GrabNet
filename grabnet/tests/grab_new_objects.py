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
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
import os
import argparse
import trimesh

import mano
# from psbody.mesh import MeshViewers, Mesh
from grabnet.tools.meshviewer import Mesh
# from grabnet.tools.vis_tools import points_to_spheres
from grabnet.tools.utils import euler
from grabnet.tools.cfg_parser import Config
from grabnet.tests.tester import Tester

from bps_torch.bps import bps_torch

# from psbody.mesh.colors import name_to_rgb
from grabnet.tools.train_tools import point2point_signed
from grabnet.tools.utils import aa2rotmat
from grabnet.tools.utils import makepath
from grabnet.tools.utils import to_cpu
from lib.viztools.viz_o3d_utils import VizContext


def vis_results(dorig, coarse_net, refine_net, rh_model , save=False, save_dir = None):
    import copy
    with torch.no_grad():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        viz_ctx = VizContext(non_block=True)
        viz_ctx.init()

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

        show_next = False
        def next_sample(_):
            nonlocal show_next
            show_next = True

        viz_ctx.register_key_callback('D', next_sample)
        print('按D键显示下一个样本')

        for cId in range(0, len(dorig['bps_object'])):
            show_next = False
            try:
                meshes = copy.deepcopy(dorig['mesh_object'])
                obj_mesh = meshes[cId]
            except Exception:
                obj_verts = to_cpu(dorig['verts_object'][cId])
                obj_faces = np.zeros((0, 3), dtype=np.int32)
                obj_mesh = None
            else:
                obj_verts = obj_mesh.vertices
                obj_faces = obj_mesh.faces

            hand_mesh_gen_rnet = to_cpu(verts_rh_gen_rnet[cId])
            hand_mesh_gen_cnet = to_cpu(verts_rh_gen_cnet[cId])
            if 'rotmat' in dorig:
                rotmat = dorig['rotmat'][cId].T
                obj_verts = obj_verts @ rotmat
                hand_mesh_gen_rnet = hand_mesh_gen_rnet @ rotmat
                hand_mesh_gen_cnet = hand_mesh_gen_cnet @ rotmat

            viz_ctx.update_by_mesh('hand_rnet', hand_mesh_gen_rnet, rh_model.faces, vcolors=[0.7, 0.7, 0.7])
            viz_ctx.update_by_mesh('hand_cnet', hand_mesh_gen_cnet, rh_model.faces, vcolors=[1.0, 0.4, 0.7])
            if obj_faces.shape[0] > 0:
                viz_ctx.update_by_mesh('obj', obj_verts, obj_faces, vcolors=[0.2, 0.8, 0.2])
            else:
                viz_ctx.update_by_pc('obj_pc', obj_verts, pcolors=[0.2, 0.8, 0.2])

            while not show_next:
                viz_ctx.step()

            if save:
                save_path = os.path.join(save_dir, str(cId))
                makepath(save_path)
                # hand mesh
                hv = hand_mesh_gen_rnet.cpu().numpy() if isinstance(hand_mesh_gen_rnet, torch.Tensor) else hand_mesh_gen_rnet
                faces = rh_model.faces.cpu().numpy() if isinstance(rh_model.faces, torch.Tensor) else rh_model.faces
                try:
                    if hv is None or faces is None:
                        raise ValueError("顶点或面数据为None")
                    if not isinstance(hv, np.ndarray) or not isinstance(faces, np.ndarray):
                        raise ValueError(f"数据类型错误: hv类型={type(hv)}, faces类型={type(faces)}")
                    if hv.shape[1] != 3 or faces.shape[1] != 3:
                        raise ValueError(f"数据维度错误: hv.shape={hv.shape}, faces.shape={faces.shape}")
                    
                    hv = hv.astype(np.float64)
                    faces = faces.astype(np.int32)
                    hand_mesh = trimesh.Trimesh(vertices=hv, faces=faces, process=False)
                    hand_mesh.export(save_path + '/rh_mesh_gen_%d.ply' % cId)
                except Exception as e:
                    print(f'[保存失败] hand_mesh 第{cId}个样本，错误：{e}')
                    print(f'数据信息: hv.shape={hv.shape if hv is not None else None}, faces.shape={faces.shape if faces is not None else None}')
                
                # obj mesh
                if obj_faces.shape[0] > 0:
                    ov = obj_verts.cpu().numpy() if isinstance(obj_verts, torch.Tensor) else obj_verts
                    of = obj_faces.cpu().numpy() if isinstance(obj_faces, torch.Tensor) else obj_faces
                    try:
                        if ov is None or of is None:
                            raise ValueError("顶点或面数据为None")
                        if not isinstance(ov, np.ndarray) or not isinstance(of, np.ndarray):
                            raise ValueError(f"数据类型错误: ov类型={type(ov)}, of类型={type(of)}")
                        if ov.shape[1] != 3 or of.shape[1] != 3:
                            raise ValueError(f"数据维度错误: ov.shape={ov.shape}, of.shape={of.shape}")
                        
                        ov = ov.astype(np.float64)
                        of = of.astype(np.int32)
                        obj_mesh = trimesh.Trimesh(vertices=ov, faces=of, process=False)
                        obj_mesh.export(save_path + '/obj_mesh_%d.ply' % cId)
                    except Exception as e:
                        print(f'[保存失败] obj_mesh 第{cId}个样本，错误：{e}')
                        print(f'数据信息: ov.shape={ov.shape if ov is not None else None}, of.shape={of.shape if of is not None else None}')
                else:
                    try:
                        if obj_verts is None:
                            raise ValueError("点云数据为None")
                        if not isinstance(obj_verts, np.ndarray):
                            obj_verts = obj_verts.cpu().numpy() if isinstance(obj_verts, torch.Tensor) else obj_verts
                        np.save(os.path.join(save_path, f'obj_pc_{cId}.npy'), obj_verts)
                    except Exception as e:
                        print(f'[保存失败] obj_pc 第{cId}个样本，错误：{e}')
                        print(f'数据信息: obj_verts.shape={obj_verts.shape if obj_verts is not None else None}')
        viz_ctx.deinit()


def grab_new_objs(grabnet, objs_path, rot=True, n_samples=10, scale=1.):
    
    grabnet.coarse_net.eval()
    grabnet.refine_net.eval()

    rh_model = mano.load(model_path=grabnet.cfg.rhm_path,
                         model_type='mano',
                         num_pca_comps=45,
                         batch_size=n_samples,
                         flat_hand_mean=True).to(grabnet.device)

    grabnet.refine_net.rhm_train = rh_model

    grabnet.logger(f'################# \n'
                   f'Colors Guide:'
                   f'                   \n'
                   f'Gray  --->  GrabNet generated grasp\n')

    bps = bps_torch(custom_basis = grabnet.bps)

    if not isinstance(objs_path, list):
        objs_path = [objs_path]
        
    for new_obj in objs_path:

        rand_rotdeg = np.random.random([n_samples, 3]) * np.array([360, 360, 360])

        rand_rotmat = euler(rand_rotdeg)
        dorig = {'bps_object': [],
                 'verts_object': [],
                 'mesh_object': [],
                 'rotmat':[]}

        for samples in range(n_samples):

            verts_obj, mesh_obj, rotmat = load_obj_verts(new_obj, rand_rotmat[samples], rndrotate=rot, scale=scale)
            
            bps_object = bps.encode(torch.from_numpy(verts_obj), feature_type='dists')['dists']

            dorig['bps_object'].append(bps_object.to(grabnet.device))
            dorig['verts_object'].append(torch.from_numpy(verts_obj.astype(np.float32)).unsqueeze(0))
            dorig['mesh_object'].append(mesh_obj)
            dorig['rotmat'].append(rotmat)
            obj_name = os.path.basename(new_obj)

        dorig['bps_object'] = torch.cat(dorig['bps_object'])
        dorig['verts_object'] = torch.cat(dorig['verts_object'])

        save_dir = os.path.join(grabnet.cfg.work_dir, 'grab_new_objects')
        print(save_dir)
        grabnet.logger(f'#################\n'
                              f'                   \n'
                              f'Showing results for the {obj_name.upper()}'
                              f'                      \n')

        vis_results(dorig=dorig,
                    coarse_net=grabnet.coarse_net,
                    refine_net=grabnet.refine_net,
                    rh_model=rh_model,
                    save=True,
                    save_dir=save_dir
                    )

def load_obj_verts(mesh_path, rand_rotmat, rndrotate=True, scale=1., n_sample_verts=10000):
    np.random.seed(100)
    obj_mesh = Mesh(filename=mesh_path, vscale=scale)
    max_length = np.linalg.norm(obj_mesh.vertices, axis=1).max()
    if  max_length > .3:
        re_scale = max_length/.08
        print(f'The object is very large, down-scaling by {re_scale} factor')
        obj_mesh.vertices = obj_mesh.vertices/re_scale
    object_fullpts = obj_mesh.vertices
    maximum = object_fullpts.max(0, keepdims=True)
    minimum = object_fullpts.min(0, keepdims=True)
    offset = ( maximum + minimum) / 2
    verts_obj = object_fullpts - offset
    obj_mesh.vertices = verts_obj
    if rndrotate:
        obj_mesh.rotate_vertices(rand_rotmat)
    else:
        rand_rotmat = np.eye(3)
    while (obj_mesh.vertices.shape[0] < n_sample_verts):
        mesh = Mesh(vertices=obj_mesh.vertices, faces=obj_mesh.faces)
        mesh = mesh.subdivide()
        obj_mesh = Mesh(v=mesh.vertices, f=mesh.faces)
    verts_obj = obj_mesh.vertices
    verts_sample_id = np.random.choice(verts_obj.shape[0], n_sample_verts, replace=False)
    verts_sampled = verts_obj[verts_sample_id]
    return verts_sampled, obj_mesh, rand_rotmat

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GrabNet-Testing')

    parser.add_argument('--obj-path', required = True, type=str,
                        help='The path to the 3D object Mesh or Pointcloud')

    parser.add_argument('--rhm-path', required = True, type=str,
                        help='The path to the folder containing MANO_RIHGT model')

    parser.add_argument('--config-path', default= None, type=str,
                        help='The path to the confguration of the trained GrabNet model')

    args = parser.parse_args()

    cfg_path = args.config_path
    obj_path = args.obj_path
    rhm_path = args.rhm_path

    cwd = os.getcwd()
    work_dir = cwd + '/logs'

    best_cnet = 'grabnet/models/coarsenet.pt'
    best_rnet = 'grabnet/models/refinenet.pt'
    bps_dir   = 'grabnet/configs/bps.npz'


    if cfg_path is None:
        cfg_path = 'grabnet/configs/grabnet_cfg.yaml'


    config = {
        'work_dir': work_dir,
        'best_cnet': best_cnet,
        'best_rnet': best_rnet,
        'bps_dir': bps_dir,
        'rhm_path': rhm_path
    }

    cfg = Config(default_cfg_path=cfg_path, **config)

    grabnet = Tester(cfg=cfg)
    grab_new_objs(grabnet,obj_path, rot=True, n_samples=10)