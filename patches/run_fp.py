# # Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
# #
# # NVIDIA CORPORATION and its licensors retain all intellectual property
# # and proprietary rights in and to this software, related documentation
# # and any modifications thereto.  Any use, reproduction, disclosure or
# # distribution of this software and related documentation without an express
# # license agreement from NVIDIA CORPORATION is strictly prohibited.


# from estimater import *
# from datareader import *
# import argparse


# if __name__=='__main__':
#   parser = argparse.ArgumentParser()
#   # =========================
#   # 必须由外部传入的路径
#   # =========================
#   parser.add_argument(
#       '--test_scene_dir',
#       type=str,
#       required=True,
#       help='Scene directory, e.g. data_online/4'
#   )

#   parser.add_argument(
#       '--mesh_file',
#       type=str,
#       required=True,
#       help='Path to mesh file, e.g. data_online/4/mesh/model.obj'
#   )

#   # =========================
#   # 可选参数
#   # =========================
#   parser.add_argument('--est_refine_iter', type=int, default=5)
#   parser.add_argument('--track_refine_iter', type=int, default=2)
#   parser.add_argument('--debug', type=int, default=2)

#   parser.add_argument(
#       '--debug_dir',
#       type=str,
#       default=None,
#       help='Debug output directory (default: <scene_dir>/debug)'
#   )

#   args = parser.parse_args()

#   # =========================
#   # 路径统一收敛
#   # =========================
#   scene_dir = os.path.abspath(args.test_scene_dir)
#   mesh_file = os.path.abspath(args.mesh_file)

#   if args.debug_dir is None:
#       debug_dir = os.path.join(scene_dir, "debug")
#   else:
#       debug_dir = os.path.abspath(args.debug_dir)

#   os.makedirs(debug_dir, exist_ok=True)

#   print("===== FoundationPose Input =====")
#   print(f"Scene dir : {scene_dir}")
#   print(f"Mesh file : {mesh_file}")
#   print(f"Debug dir : {debug_dir}")
#   print("================================")
  
#   args = parser.parse_args()

#   set_logging_format()
#   set_seed(0)

#   mesh = trimesh.load(args.mesh_file)

#   debug = args.debug
#   debug_dir = args.debug_dir
#   os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

#   to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
#   bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

#   scorer = ScorePredictor()
#   refiner = PoseRefinePredictor()
#   glctx = dr.RasterizeCudaContext()
#   est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
#   logging.info("estimator initialization done")

#   reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)

#   for i in range(len(reader.color_files)):
#     logging.info(f'i:{i}')
#     color = reader.get_color(i)
#     depth = reader.get_depth(i)
#     if i==0:
#       mask = reader.get_mask(0).astype(bool)
#       pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)

#       if debug>=3:
#         m = mesh.copy()
#         m.apply_transform(pose)
#         m.export(f'{debug_dir}/model_tf.obj')
#         xyz_map = depth2xyzmap(depth, reader.K)
#         valid = depth>=0.001
#         pcd = toOpen3dCloud(xyz_map[valid], color[valid])
#         o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
#     else:
#       pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)

#     os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
#     np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4,4))

#     if debug>=1:
#       center_pose = pose@np.linalg.inv(to_origin)
#       vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
#       vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
#       cv2.imshow('1', vis[...,::-1])
#       cv2.waitKey(1)


#     if debug>=2:
#       os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
#       imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)
# Copyright (c) 2023, NVIDIA CORPORATION.
# All rights reserved.

from estimater import *
from datareader import *

import argparse
import os
import logging
import numpy as np
import trimesh
import imageio
import cv2
import nvdiffrast.torch as dr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # =========================
    # 必须由外部传入的路径
    # =========================
    parser.add_argument(
        '--test_scene_dir',
        type=str,
        required=True,
        help='Scene directory, e.g. data_online/4'
    )

    parser.add_argument(
        '--mesh_file',
        type=str,
        required=True,
        help='Path to mesh file, e.g. data_online/4/mesh/model.obj'
    )

    # =========================
    # 可选参数
    # =========================
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--debug', type=int, default=2)

    parser.add_argument(
        '--debug_dir',
        type=str,
        default=None,
        help='Debug output directory (default: <scene_dir>/debug)'
    )

    # =========================
    # 解析参数（只做一次）
    # =========================
    args = parser.parse_args()

    # =========================
    # 路径统一收敛（全局唯一来源）
    # =========================
    scene_dir = os.path.abspath(args.test_scene_dir)
    mesh_file = os.path.abspath(args.mesh_file)

    if args.debug_dir is None:
        debug_dir = os.path.join(scene_dir, "debug")
    else:
        debug_dir = os.path.abspath(args.debug_dir)

    os.makedirs(debug_dir, exist_ok=True)

    print("===== FoundationPose Input =====")
    print(f"Scene dir : {scene_dir}")
    print(f"Mesh file : {mesh_file}")
    print(f"Debug dir : {debug_dir}")
    print("================================")

    # =========================
    # 初始化
    # =========================
    set_logging_format()
    set_seed(0)

    mesh = trimesh.load(mesh_file)

    debug = args.debug

    # 清空并创建 debug 目录
    os.system(
        f'rm -rf {debug_dir}/* && '
        f'mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam'
    )

    # =========================
    # 预计算 bbox
    # =========================
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    # =========================
    # 初始化 FoundationPose
    # =========================
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()

    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=debug_dir,
        debug=debug,
        glctx=glctx
    )

    logging.info("estimator initialization done")

    # =========================
    # 读取数据
    # =========================
    reader = YcbineoatReader(
        video_dir=scene_dir,
        shorter_side=None,
        zfar=np.inf
    )

    # =========================
    # 主循环
    # =========================
    for i in range(len(reader.color_files)):
        logging.info(f'i: {i}')

        color = reader.get_color(i)
        depth = reader.get_depth(i)

        if i == 0:
            mask = reader.get_mask(0).astype(bool)
            pose = est.register(
                K=reader.K,
                rgb=color,
                depth=depth,
                ob_mask=mask,
                iteration=args.est_refine_iter
            )

            if debug >= 3:
                m = mesh.copy()
                m.apply_transform(pose)
                m.export(f'{debug_dir}/model_tf.obj')

                xyz_map = depth2xyzmap(depth, reader.K)
                valid = depth >= 0.001
                pcd = toOpen3dCloud(xyz_map[valid], color[valid])
                o3d.io.write_point_cloud(
                    f'{debug_dir}/scene_complete.ply', pcd
                )
        else:
            pose = est.track_one(
                rgb=color,
                depth=depth,
                K=reader.K,
                iteration=args.track_refine_iter
            )

        # =========================
        # 保存位姿
        # =========================
        os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
        np.savetxt(
            f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt',
            pose.reshape(4, 4)
        )

        # =========================
        # 可视化
        # =========================
        if debug >= 1:
            center_pose = pose @ np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(
                reader.K,
                img=color,
                ob_in_cam=center_pose,
                bbox=bbox
            )
            vis = draw_xyz_axis(
                color,
                ob_in_cam=center_pose,
                scale=0.1,
                K=reader.K,
                thickness=3,
                transparency=0,
                is_input_rgb=True
            )
            cv2.imshow('FoundationPose', vis[..., ::-1])
            cv2.waitKey(1)

        if debug >= 2:
            os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
            imageio.imwrite(
                f'{debug_dir}/track_vis/{reader.id_strs[i]}.png',
                vis
            )
