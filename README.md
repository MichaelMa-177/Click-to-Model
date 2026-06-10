# Click-to-Model

端到端"点一下即得目标物体的6D 位姿"pipeline：
**RealSense 采集 → SAM 交互分割 → SAM-3D 重建带纹理 mesh → 基于形状颜色感知的尺度配准 → ICP + FoundationPose 联合追踪机制输出每帧 6D 位姿**。

---

## 流水线

```
demo_from_notebook.py
  ├─ [T0]  预加载 SAM ViT-H 权重到 GPU (cuda:0)
  ├─ ----  RealSense 采集 N 帧 → rgb/ depth/ cam_K.txt / camera_params.json
  ├─ ----  在第一帧上点击标注（左键正点、右键负点）
  ├─ [T1]  SAM set_image + predict → best_mask
  ├─ [--]  SAM-3D 子进程：mesh + 纹理（ICP + 度量尺度对齐）
  └─ [T3]  FoundationPose 子进程：每帧 6D 位姿 + 可视化
```

数据布局：每次运行在 `data_online/<max+1>/` 下生成
```
<id>/
  rgb/       <- RealSense 录制
  depth/
  masks/     <- SAM 输出
  mesh/      <- SAM-3D 输出 (model.obj/.mtl/_texture.png)
  debug/     <- FoundationPose 输出 (ob_in_cam/, track_vis/)
  cam_K.txt
  camera_params.json
```

---

## 一键部署

```bash
git clone https://github.com/MichaelMa-177/Click-to-Model.git
cd Click-to-Model
bash scripts/setup.sh
```

`setup.sh` 顺序：
1. `scripts/clone_upstream.sh` — clone 6 个上游仓库到 Click-to-Model 下
2. 把 `patches/` 里的定制脚本覆盖到上游
3. `scripts/install_envs.sh` — `conda env create -f environments/*.yml` 创建两个 env
4. `scripts/download_models.sh` — 下载 SAM / SAM-3D / Dinov2 / FoundationPose 权重

> ⚠️ pytorch3d C++/CUDA 扩展首次会编译 5-15 分钟，需要 `nvcc` 12.1。
> ⚠️ FoundationPose 权重在 Google Drive，需要 `gdown`，且首次可能需要手动鉴权。

---

## 子项目来源

| 目录 | 上游 | 用途 |
|---|---|---|
| `segment-anything/` | `facebookresearch/segment-anything` | SAM ViT-H 推理 |
| `sam-3d-objects/` | `facebookresearch/sam-3d-objects` | 单图重建带纹理 mesh |
| `sam-3d-objects/pytorch3d/` | `facebookresearch/pytorch3d` | sam-3d-objects 的 mesh 依赖 |
| `FoundationPose/` | `MichaelMa-177/FoundationPose` (fork) | 6D 物体位姿估计与跟踪 |
| `dinov2/` | `facebookresearch/dinov2` | sam-3d-objects 的 backbone |
| `nvdiffrast/` | `NVlabs/nvdiffrast` | 可微分光栅化（重建中用） |

---

## 运行

接入 RealSense 相机后：

```bash
/data/ubuntu_data/miniconda3/envs/sam3d-objects/bin/python \
    Click-to-Model/demo_from_notebook.py
```

### 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--save_data {0,1}` | 1 | 1=保留本次目录；0=运行结束/失败时整目录删除 |
| `--n_frames N` | 300 | RealSense 录制帧数（给 FoundationPose 跟踪） |
| `--no_capture` | False | 跳过相机采集（调试，需自备 rgb/depth） |
| `--data_root PATH` | `<repo>/data_online` | 数据根目录，也可用 `$DATA_ROOT` |
| `--cuda_device N` | 0 | SAM 使用的 GPU id |
| `--no_gui` | False | 禁用 OpenCV 预览（无头环境） |

### 计时输出

每次运行结束会打印：
- **模型权重加载** — `cuda:0` 上 SAM ViT-H load + to(device)
- **SAM best_mask** — `set_image + predict + argmax + 写盘`
- **FoundationPose** — 子进程总耗时
- **SAM3D 参考** — 子进程总耗时

### GPU 分配

| 阶段 | 设备 |
|---|---|
| SAM（主进程） | `cuda:0`，结束后立即 `empty_cache()` 让出 |
| SAM3D（子进程） | `cuda:0` + `cuda:1`（双卡 encoder/decoder 分卡） |
| FoundationPose（子进程） | `cuda:0`（`CUDA_VISIBLE_DEVICES` 显式锁定） |

串行执行无冲突。

---

## 文件结构（仓库内）

```
Click-to-Model/
├── demo_from_notebook.py     # 主入口
├── run.sh                    # 等价命令行入口（无相机采集）
├── environments/
│   ├── sam3d-objects.yml     # conda env: SAM-3D + SAM 主进程 (Python 3.11, CUDA 12.1)
│   └── foundationpose.yml    # conda env: FoundationPose (Python 3.9)
├── scripts/
│   ├── setup.sh              # 编排：clone → patches → envs → models
│   ├── clone_upstream.sh     # 拉 6 个上游仓库
│   ├── install_envs.sh       # conda env create × 2
│   └── download_models.sh    # SAM/SAM-3D/Dinov2/FoundationPose 权重
├── patches/
│   ├── run_sam3d.py          # 定制版（含 ICP+尺度对齐），覆盖上游
│   └── run_fp.py             # 定制版（含 argparse），覆盖上游
├── .gitignore
└── README.md
```

---

## 排错

- **相机找不到**：`pip install pyrealsense2`；`lsusb | grep -i intel` 看是否插好。
- **pytorch3d 编译失败**：检查 `nvcc -V` 是 12.1；或换 wheel `pip install --no-build-isolation pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py311_cu121_pyt251/download.html`。
- **SAM3D OOM**：双卡是默认，单卡可设 `CUDA_VISIBLE_DEVICES=0` 但容易 OOM；推荐 24GB+。
- **FoundationPose 慢**：300 帧 4090 上约 5-15 分钟，跟物体复杂度与 `--debug` 级别有关。
