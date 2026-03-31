#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ================== Block 1 ==================
# 显示图像 + 交互点击记录点（改为 OpenCV 窗口点击）
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys
import subprocess

# ================== 1. 读取图像 ==================
image_path = "/home/data_online/1/rgb/0000.png"

image_bgr = cv2.imread(image_path)
if image_bgr is None:
    raise FileNotFoundError(f"图像未找到: {image_path}")
image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

print("[INFO] 已加载图像，准备点击标注")
print("[INFO] OpenCV窗口: 左键=正点(绿), 右键=负点(红), Enter/Space=完成, c=清空重选, q=退出")

# ================== 2. 记录点 ==================
clicked_points = []
point_labels = []

# OpenCV 显示缓冲
canvas = image_bgr.copy()
win_name = "SAM Point Annotation"

def redraw_canvas():
    global canvas
    canvas = image_bgr.copy()
    for (x, y), lbl in zip(clicked_points, point_labels):
        color = (0, 255, 0) if lbl == 1 else (0, 0, 255)  # BGR: 正点绿、负点红
        cv2.drawMarker(canvas, (x, y), color, markerType=cv2.MARKER_STAR, markerSize=18, thickness=2)

def on_mouse(event, x, y, flags, param):
    global clicked_points, point_labels
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append([x, y])
        point_labels.append(1)
        print(f"[CLICK] 正点: ({x}, {y})")
        redraw_canvas()
    elif event == cv2.EVENT_RBUTTONDOWN:
        clicked_points.append([x, y])
        point_labels.append(0)
        print(f"[CLICK] 负点: ({x}, {y})")
        redraw_canvas()

cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(win_name, on_mouse)
redraw_canvas()

while True:
    cv2.imshow(win_name, canvas)
    k = cv2.waitKey(20) & 0xFF
    if k in (13, 32):  # Enter / Space
        break
    elif k == ord("c"):
        clicked_points.clear()
        point_labels.clear()
        redraw_canvas()
        print("[INFO] 已清空标注点")
    elif k == ord("q"):
        cv2.destroyAllWindows()
        raise RuntimeError("用户取消标注")

cv2.destroyAllWindows()

if len(clicked_points) == 0:
    raise ValueError("你还没有点击任何点！")

# ================== Block 2 ==================
# 使用 clicked_points / point_labels 进行 SAM 推理并保存最佳 mask

# ================== 2. SAM 初始化 ==================
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

file_path = os.getcwd()
project_root = os.path.dirname(file_path)
project_root = os.path.dirname(project_root)

sam_checkpoint = os.path.join(
    project_root,
    "segment-anything/checkpoints/sam_vit_h_4b8939.pth"
)

print("[INFO] 自动生成的 SAM 路径为：", sam_checkpoint)
sam_checkpoint = "/home/segment-anything/checkpoints/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)
predictor.set_image(image)

# 转 numpy
input_points = np.array(clicked_points)
input_labels = np.array(point_labels)

print("[INFO] 开始 SAM 多点推理 ...")

# ================== 3. SAM 推理 ==================
masks, scores, logits = predictor.predict(
    point_coords=input_points,
    point_labels=input_labels,
    multimask_output=True
)

# ================== 4. 选取最佳 mask ==================
best_idx = np.argmax(scores)
best_mask = masks[best_idx]
best_score = scores[best_idx]

print(f"[INFO] 最佳 mask index={best_idx}, score={best_score:.4f}")

save_path2 = "/home/data_online/1/masks/0000.png"
mask_to_save = (best_mask.astype(np.uint8) * 255)
cv2.imwrite(save_path2, mask_to_save)
print(f"[INFO] 最佳 Mask 已保存至: {save_path2}")
print("[DEBUG] mask real path:", os.path.abspath(save_path2))

# ================== 5. 可视化最佳 mask ==================
def show_mask(mask, ax, random_color=False):
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    pos = coords[labels == 1]
    neg = coords[labels == 0]
    if len(pos) > 0:
        ax.scatter(pos[:, 0], pos[:, 1], color='green', s=marker_size, marker='*')
    if len(neg) > 0:
        ax.scatter(neg[:, 0], neg[:, 1], color='red', s=marker_size, marker='*')

plt.figure(figsize=(10, 10))
plt.imshow(image)
show_mask(best_mask, plt.gca())
show_points(input_points, input_labels, plt.gca())
plt.title(f"Best Mask (score={best_score:.3f})")
plt.axis("off")
plt.show()

# ================== Block 3（你 notebook 最后一段） ==================
import os
os.environ["REQUESTS_CA_BUNDLE"] = "/etc/ssl/certs/ca-certificates.crt"
os.environ["SSL_CERT_FILE"] = "/etc/ssl/certs/ca-certificates.crt"
os.environ["CURL_CA_BUNDLE"] = "/etc/ssl/certs/ca-certificates.crt"
os.environ["XFORMERS_DISABLED"] = "1"

result = subprocess.run(
    'bash -c "source /etc/network_turbo && env | grep -iE \\"^(http|https|all|no)_proxy=\\""',
    shell=True,
    capture_output=True,
    text=True,
    check=True,
)

output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value

os.chdir("/home")
run_sh = "run.sh"
subprocess.run(["bash", "-x", run_sh], check=True)