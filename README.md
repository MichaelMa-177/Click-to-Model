# Click-to-Model

从 RealSense 第一帧交互点击开始，依次完成 SAM2 分割、SAM3D 单物体重建、
RGB-D 米制尺度恢复，以及 SPARK-6D/FoundationPose 6D 位姿跟踪。

```text
RealSense RGB-D → SAM2 mask → SAM3D mesh → bounded scale ICP → 6D tracking
```

## 快速运行

```bash
# 默认布局：Click-to-Model 与 SPARK-6D 位于同一工作区
git clone --recurse-submodules \
  https://github.com/MichaelMa-177/Click-to-Model.git
cd Click-to-Model

conda activate foundationpose
source scripts/activate_click_to_model.sh

python -m click_to_model \
  --online \
  --distilled \
  --scale-mode icp \
  --require-icp-confidence
```

无需指定相机序列号时会选择第一台 RealSense；多相机环境可增加
`--camera-serial <SERIAL>`。首次部署、权重放置和 CUDA 扩展编译见
[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)。

相机窗口按 `s` 保存第一帧。SAM2 窗口中左键标目标、右键标背景，`u` 撤销、
`c` 清空、Enter/Space 接受。左上角的有效深度比例变红时，建议调整目标距离、
视角或照明后重新采集。

保存带位姿框、坐标轴和 FPS 的跟踪视频：

```bash
python -m click_to_model --online --distilled --save-tracking-video
```

只运行 SAM3D 和尺度恢复：

```bash
python -m click_to_model.reconstruction.cli \
  --data-dir /path/to/sequence \
  --scale-mode icp \
  --require-icp-confidence
```

原命令 `python run_click_to_model.py` 和 `python run_sam3d_rgbd.py` 仍可使用，
但这两个顶层文件现在只是兼容入口。

## 工程目录

```text
Click-to-Model/
├── click_to_model/                 # 维护中的应用代码
│   ├── camera.py                   # RealSense 采集与内参
│   ├── config.py                   # 环境变量和外部路径
│   ├── dataset.py                  # 数据集目录与首帧归档
│   ├── segmentation.py             # SAM2 会话和点击界面
│   ├── processes.py                # GPU 子进程/预加载生命周期
│   ├── tracking.py                 # 跟踪后端命令构建
│   ├── pipeline.py                 # 端到端编排入口
│   └── reconstruction/
│       ├── cli.py                  # 仅重建入口
│       ├── geometry.py             # RGB-D 投影、网格与可视化
│       ├── metric_scale.py         # 有界尺度搜索与刚体 ICP
│       └── pipeline.py             # SAM3D 推理和结果导出
├── tests/
│   ├── unit/                       # 无相机的快速回归测试
│   └── integration/                # SPARK-6D/视频集成测试
├── docs/                           # 架构与完整运行说明
├── tools/                          # 独立工具，不被主流程导入
├── scripts/                        # 当前环境激活和检查脚本
├── legacy/                         # 旧版原型，仅保留作复现
├── sam-3d-objects/                 # 第三方定制子模块
├── FoundationPose/                 # 第三方子模块
└── segment-anything/               # 第三方子模块
```

模块设计与依赖边界见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)，完整参数、
数据复用和输出说明见 [docs/USAGE.md](docs/USAGE.md)。

## 核心约定

- `rgb/` 与 `depth/` 必须同名且已经对齐，共用 `cam_K.txt`。
- 深度 PNG 为 RealSense `z16`，通过 `camera_params.json.depth_scale` 转为米。
- 在线初始化帧单独归档在 `masks/rgb/` 和 `masks/depth/`，中断后可继续运行。
- `mesh/model.obj` 是以物体为中心、单位为米的跟踪模型。
- SAM3D 在独立子进程运行，结束后释放显存再启动跟踪器。
- 第三方子模块保持独立，不在应用模块中复制上游实现。

## 尺度恢复

默认 `--scale-mode icp` 使用 mask 轮廓、对齐深度点云和 SAM3D 姿态建立有限尺度
候选，然后对每个尺度只运行刚体 ICP。这样避免单视角无约束 Sim(3) ICP 将完整网格
错误缩到局部可见表面。

以下情况会降低或拒绝尺度结果：

- ICP 点云覆盖率不足或 RMSE 过高；
- mask 内有效深度低于 `--min-depth-coverage`；
- mask 接触图像边缘，导致物体轮廓不完整。

诊断结果写入 `mesh/model_metadata.json`、`registration_overlay.png`、
`model_aligned_camera.ply` 和 `depth_object.ply`。

## 外部组件

默认使用可移植的仓库相对布局：

- Python：当前已激活 Conda 环境的 `python`
- SPARK-6D：主仓库同级的 `../SPARK-6D`
- SAM2：`../SPARK-6D/third_party/sam2`
- SAM3D：本仓库 `sam-3d-objects` fork 及官方权重
- GPU 路径：RTX 4070 SUPER 12GB 的分阶段低显存 mesh-only 推理

路径均可通过 `SPARK6D_REPO`、`SAM2_REPO`、`SAM2_CHECKPOINT`、
`SAM3D_REPO` 和 `SAM3D_CHECKPOINT_DIR` 覆盖，参考
[.env.example](.env.example)。仓库不保存用户主目录、相机序列号、令牌、
权重或采集数据。旧版双环境及覆盖式 patch 部署已移到 `legacy/`，
不要用于当前工程。

## 检查

```bash
bash scripts/check.sh
python scripts/doctor.py --strict
```

`check.sh` 运行快速回归测试和语法检查；`doctor.py` 验证当前环境、
子模块、外部仓库及权重布局。两者都不会启动相机或执行完整 SAM3D 推理。
