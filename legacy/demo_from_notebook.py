#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Click-to-Model end-to-end pipeline.

Flow:
  1) 分配新数据目录 <data_root>/<max+1>/(rgb depth masks mesh debug)
  2) 预加载 SAM 权重到 GPU         [TIMED: weight_load]
  3) RealSense 采集 rgb/depth/相机内参（不计时）
  4) 用户在第一帧上点击标注（不计时）
  5) SAM 推理生成 best_mask         [TIMED: sam_mask]
  6) SAM3D 子进程重建 mesh          [参考时长]
  7) FoundationPose 子进程位姿+可视化 [TIMED: foundationpose]
  8) 打印计时汇总
  9) --save_data=0 时删除本次新建的整个数据目录
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

# =========================================================
# 路径与默认配置
# =========================================================
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAM_REPO_DIR = os.path.join(SCRIPT_DIR, "segment-anything")
SAM3D_DIR = os.path.join(SCRIPT_DIR, "sam-3d-objects")
FP_DIR = os.path.join(SCRIPT_DIR, "FoundationPose")

SAM3D_PY = os.environ.get("SAM_PY") or sys.executable
FP_PY = os.environ.get("FP_PY") or sys.executable
SAM_CKPT = os.environ.get("SAM_CHECKPOINT") or os.path.join(
    SAM_REPO_DIR, "checkpoints", "sam_vit_h_4b8939.pth")


# =========================================================
# 1) 数据目录分配
# =========================================================
def allocate_new_data_dir(data_root):
    """选 max+1（空则 1），创建 rgb/depth/masks/mesh/debug 骨架。"""
    os.makedirs(data_root, exist_ok=True)
    nums = [int(d) for d in os.listdir(data_root)
            if d.isdigit() and os.path.isdir(os.path.join(data_root, d))]
    new_id = (max(nums) + 1) if nums else 1
    new_dir = os.path.join(data_root, str(new_id))
    os.makedirs(new_dir, exist_ok=False)
    for sub in ("rgb", "depth", "masks", "mesh", "debug"):
        os.makedirs(os.path.join(new_dir, sub), exist_ok=True)
    return new_id, new_dir


# =========================================================
# 3) RealSense 采集
# =========================================================
def capture_with_realsense(save_dir, n_frames, preview=True):
    """写 rgb/depth/cam_K.txt/camera_params.json。预览 + 按 s 开始录 N 帧后自停。"""
    try:
        import pyrealsense2 as rs
    except ImportError as e:
        raise RuntimeError(
            "未找到 pyrealsense2。请在运行 demo 的 Python 环境中安装：\n"
            "  pip install pyrealsense2\n"
            "或在已装 RealSense SDK 的环境中运行本脚本。"
        ) from e

    rgb_dir = os.path.join(save_dir, "rgb")
    depth_dir = os.path.join(save_dir, "depth")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    intr = (profile.get_stream(rs.stream.color)
                   .as_video_stream_profile()
                   .get_intrinsics())
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    K = np.array([[intr.fx, 0, intr.ppx],
                  [0, intr.fy, intr.ppy],
                  [0, 0, 1]], dtype=np.float64)
    np.savetxt(os.path.join(save_dir, "cam_K.txt"), K)

    cam_params = {
        "width": intr.width, "height": intr.height,
        "fx": intr.fx, "fy": intr.fy,
        "ppx": intr.ppx, "ppy": intr.ppy,
        "depth_scale": depth_scale,
    }
    with open(os.path.join(save_dir, "camera_params.json"), "w") as f:
        json.dump(cam_params, f, indent=4)

    print(f"[CAM] 已写入 cam_K.txt / camera_params.json")
    print(f"[CAM] 对准目标后按 's' 开始录制 {n_frames} 帧；'q' 取消")

    win = "RealSense Preview"
    if preview:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    recording = False
    idx = 0
    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            cf = aligned.get_color_frame()
            df = aligned.get_depth_frame()
            if not cf or not df:
                continue
            color = np.asanyarray(cf.get_data())
            depth = np.asanyarray(df.get_data())

            if recording:
                cv2.imwrite(os.path.join(rgb_dir, f"{idx:06d}.png"), color)
                cv2.imwrite(os.path.join(depth_dir, f"{idx:06d}.png"), depth)
                idx += 1
                if idx >= n_frames:
                    print(f"[CAM] 已录制 {idx} 帧")
                    break

            if preview:
                vis = color.copy()
                if recording:
                    cv2.putText(vis, f"REC {idx}/{n_frames}",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, (0, 0, 255), 2)
                else:
                    cv2.putText(vis, "Press 's' to record, 'q' to abort",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 255, 0), 2)
                cv2.imshow(win, vis)
                k = cv2.waitKey(1) & 0xFF
                if k == ord('s') and not recording:
                    recording = True
                    print(f"[CAM] 开始录制 {n_frames} 帧")
                elif k == ord('q'):
                    raise RuntimeError("用户取消采集")
    finally:
        pipeline.stop()
        if preview:
            cv2.destroyWindow(win)


# =========================================================
# 4) 点击标注
# =========================================================
def click_annotate(image_bgr):
    clicked_points, point_labels = [], []
    canvas = [image_bgr.copy()]
    win_name = "SAM Point Annotation"

    def redraw():
        canvas[0] = image_bgr.copy()
        for (x, y), lbl in zip(clicked_points, point_labels):
            color = (0, 255, 0) if lbl == 1 else (0, 0, 255)
            cv2.drawMarker(canvas[0], (x, y), color,
                           markerType=cv2.MARKER_STAR,
                           markerSize=18, thickness=2)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_points.append([x, y]); point_labels.append(1)
            print(f"[CLICK] 正点: ({x}, {y})"); redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            clicked_points.append([x, y]); point_labels.append(0)
            print(f"[CLICK] 负点: ({x}, {y})"); redraw()

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, on_mouse)
    redraw()

    print("[INFO] 左键=正点(绿), 右键=负点(红), Enter/Space=完成, c=清空, q=退出")
    while True:
        cv2.imshow(win_name, canvas[0])
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 32):
            break
        elif k == ord('c'):
            clicked_points.clear(); point_labels.clear(); redraw()
            print("[INFO] 已清空标注点")
        elif k == ord('q'):
            cv2.destroyAllWindows()
            raise RuntimeError("用户取消标注")
    cv2.destroyAllWindows()
    if not clicked_points:
        raise ValueError("未点击任何点")
    return clicked_points, point_labels


# =========================================================
# main
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Click-to-Model 端到端：相机采集 -> SAM 分割 -> SAM3D 重建 -> FoundationPose 位姿")
    parser.add_argument("--save_data", type=int, default=1, choices=[0, 1],
                        help="1=保留本次生成的数据目录；0=运行结束(或中断)后删除整个新目录")
    parser.add_argument("--n_frames", type=int, default=300,
                        help="RealSense 录制帧数（用于 FoundationPose 跟踪），默认 300")
    parser.add_argument("--no_capture", action="store_true",
                        help="跳过相机采集（调试用，需自己保证 rgb/depth/cam_K.txt 已就位）")
    parser.add_argument("--data_root", default=os.environ.get("DATA_ROOT"),
                        help="数据根目录，默认 <SCRIPT_DIR>/data_online")
    parser.add_argument("--cuda_device", type=int, default=0,
                        help="SAM 推理用的 GPU id（默认 0）")
    parser.add_argument("--no_gui", action="store_true",
                        help="禁用 OpenCV 预览窗口（无头环境）")
    args = parser.parse_args()

    data_root = args.data_root or os.path.join(SCRIPT_DIR, "data_online")
    os.makedirs(data_root, exist_ok=True)

    # ---------- 1) 分配新数据目录 ----------
    new_id, data_dir = allocate_new_data_dir(data_root)
    print(f"[INFO] DATA_ROOT  : {data_root}")
    print(f"[INFO] NEW_ID     : {new_id}")
    print(f"[INFO] DATA_DIR   : {data_dir}")

    # 任意一步失败都按 save_data=0 清理
    def cleanup_on_failure(msg):
        print(f"[ERR] {msg}")
        if not args.save_data:
            print(f"[INFO] save_data=0, 删除 {data_dir}")
            shutil.rmtree(data_dir, ignore_errors=True)

    try:
        # ---------- 2) 预加载 SAM 权重（计时 T0） ----------
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用")
        device = f"cuda:{args.cuda_device}"
        # CUDA warmup：避免首次 to(device) 含 context 创建时间
        _ = torch.zeros(1, device=device)
        torch.cuda.synchronize(device)

        if not os.path.exists(SAM_CKPT):
            raise FileNotFoundError(f"SAM 权重未找到: {SAM_CKPT}")
        if SAM_REPO_DIR not in sys.path:
            sys.path.insert(0, SAM_REPO_DIR)
        from segment_anything import sam_model_registry, SamPredictor

        print(f"[INFO] SAM 权重加载到 {device} ...")
        t0 = time.perf_counter()
        sam = sam_model_registry["vit_h"](checkpoint=SAM_CKPT)
        sam.to(device=device)
        predictor = SamPredictor(sam)
        torch.cuda.synchronize(device)
        weight_load_time = time.perf_counter() - t0
        print(f"[TIME] 模型权重加载: {weight_load_time:.3f}s")

        # ---------- 3) 相机采集（不计时） ----------
        if args.no_capture:
            print("[INFO] --no_capture 跳过相机采集")
        else:
            print("[INFO] 启动 RealSense 采集")
            capture_with_realsense(data_dir, args.n_frames,
                                   preview=not args.no_gui)

        # ---------- 4) 取第一帧 + 用户点击（不计时） ----------
        rgb_files = sorted(glob.glob(os.path.join(data_dir, "rgb", "*.png")))
        if not rgb_files:
            raise FileNotFoundError(f"{data_dir}/rgb 下没有 PNG")
        image_path = rgb_files[0]
        image_stem = os.path.splitext(os.path.basename(image_path))[0]
        image_bgr = cv2.imread(image_path)
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        if args.no_gui:
            raise RuntimeError("--no_gui 模式不支持点击标注")
        clicked_points, point_labels = click_annotate(image_bgr)

        # ---------- 5) SAM 推理生成 best_mask（计时 T1） ----------
        print("[INFO] 开始 SAM 推理 ...")
        t1 = time.perf_counter()
        predictor.set_image(image)
        masks, scores, _ = predictor.predict(
            point_coords=np.array(clicked_points),
            point_labels=np.array(point_labels),
            multimask_output=True,
        )
        best_idx = int(np.argmax(scores))
        best_mask = masks[best_idx]
        mask_path = os.path.join(data_dir, "masks", f"{image_stem}.png")
        cv2.imwrite(mask_path, (best_mask.astype(np.uint8) * 255))
        torch.cuda.synchronize(device)
        sam_mask_time = time.perf_counter() - t1
        print(f"[TIME] SAM best_mask: {sam_mask_time:.3f}s "
              f"(score={scores[best_idx]:.3f})")
        print(f"[INFO] mask 已保存 -> {mask_path}")

        # 释放 SAM 显存，给下游让位
        del sam, predictor
        torch.cuda.empty_cache()

        # ---------- 6) SAM3D 重建 mesh（子进程，参考时长） ----------
        sam3d_env = os.environ.copy()
        sam3d_env["XFORMERS_DISABLED"] = "1"
        sam3d_env["DATA_ROOT"] = data_root
        sam3d_env["DATA_ID"] = str(new_id)

        print(f"\n[STAGE] SAM3D 重建 mesh ...")
        t2 = time.perf_counter()
        r = subprocess.run(
            [SAM3D_PY, os.path.join(SAM3D_DIR, "run_sam3d.py"),
             "--data_dir", data_dir],
            cwd=SAM3D_DIR, env=sam3d_env)
        sam3d_time = time.perf_counter() - t2
        if r.returncode != 0:
            raise RuntimeError(f"SAM3D 返回码 {r.returncode}")
        print(f"[TIME] SAM3D 子进程总耗时: {sam3d_time:.3f}s")

        # ---------- 7) FoundationPose（子进程，计时 T3） ----------
        fp_env = os.environ.copy()
        # FP 单进程，强制 cuda:0；本机 cuda:2 闲置可作备援
        fp_env.setdefault("CUDA_VISIBLE_DEVICES",
                          str(args.cuda_device))
        mesh_file = os.path.join(data_dir, "mesh", "model.obj")
        debug_dir = os.path.join(data_dir, "debug")

        print(f"\n[STAGE] FoundationPose ...")
        t3 = time.perf_counter()
        r = subprocess.run(
            [FP_PY, os.path.join(FP_DIR, "run_fp.py"),
             "--test_scene_dir", data_dir,
             "--mesh_file", mesh_file,
             "--debug_dir", debug_dir,
             "--debug", "3"],
            cwd=FP_DIR, env=fp_env)
        fp_time = time.perf_counter() - t3
        if r.returncode != 0:
            raise RuntimeError(f"FoundationPose 返回码 {r.returncode}")
        print(f"[TIME] FoundationPose 子进程总耗时: {fp_time:.3f}s")

        # ---------- 8) 汇总 ----------
        print("\n" + "=" * 60)
        print("  TIMING SUMMARY")
        print("=" * 60)
        print(f"  模型权重加载         : {weight_load_time:8.3f} s")
        print(f"  SAM best_mask        : {sam_mask_time:8.3f} s")
        print(f"  FoundationPose       : {fp_time:8.3f} s")
        print(f"  (SAM3D 参考)          : {sam3d_time:8.3f} s")
        print("=" * 60)

    except Exception as e:
        cleanup_on_failure(str(e))
        raise

    # ---------- 9) save_data 处理 ----------
    if args.save_data:
        print(f"[INFO] save_data=1, 保留: {data_dir}")
    else:
        print(f"[INFO] save_data=0, 删除: {data_dir}")
        shutil.rmtree(data_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
