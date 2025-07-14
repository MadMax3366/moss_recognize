import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm


def find_and_crop_quadrat(image, filename=None, output_size=(500, 500)):
    """识别样方区域并透视裁剪，若失败则裁剪图片中心正方形区域"""
    DEBUG = False
    output_dir = "processing_results"
    os.makedirs(output_dir, exist_ok=True)

    image_h, image_w = image.shape[:2]
    image_area = image_h * image_w

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    l_channel = lab[:, :, 0]
    _, binary = cv2.threshold(l_channel, 200, 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    min_side_ratio = 0.5
    min_side_len = min(image_w, image_h) * min_side_ratio

    max_area = 0
    best_box = None
    for i in range(1, num_labels):  # skip background
        x, y, w, h, area = stats[i]
        aspect_ratio = w / h
        if w < min_side_len or h < min_side_len:
            continue
        if not (0.8 < aspect_ratio < 1.2):
            continue
        if area > max_area:
            max_area = area
            best_box = (x, y, w, h)

    if best_box is None:
        print("⚠️ 未找到样方，自动裁剪中心区域")
        side_ratio = 0.88
        side = int(min(image_w, image_h) * side_ratio)
        x0 = max((image_w - side) // 2, 0)
        y0 = max((image_h - side) // 2, 0)
        fallback_crop = image[y0:y0 + side, x0:x0 + side]

        if DEBUG and filename:
            cv2.imwrite(os.path.join(output_dir, f"fallback_cropped_{filename}"), fallback_crop)
        return fallback_crop

    x, y, w, h = best_box
    box_corners = np.array([
        [x, y],
        [x + w - 1, y],
        [x + w - 1, y + h - 1],
        [x, y + h - 1]
    ], dtype="float32")

    dst = np.array([
        [0, 0],
        [output_size[0] - 1, 0],
        [output_size[0] - 1, output_size[1] - 1],
        [0, output_size[1] - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(box_corners, dst)
    warped = cv2.warpPerspective(image, M, output_size)

    if DEBUG and filename:
        debug_img = image.copy()
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 255), 3)
        cv2.imwrite(os.path.join(output_dir, f"debug_detected_box_{filename}"), debug_img)
        cv2.imwrite(os.path.join(output_dir, f"cropped_{filename}"), warped)

    return warped

def detect_moss_by_rgb_gray(image):
    """根据采样RGB规律 + 灰度规则提取多类苔藓区域"""

    B, G, R = cv2.split(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 绿色苔藓（明亮绿色，G>R>B，且数值不高）
    green_moss = (
        (G > R) & (R > B) &
        (G < 140) & (R < 130) & (B < 110) &
        (G - R > 2) & (R - B > 5) &
        (gray > 60) & (gray < 160)
    )
    # 高亮绿色苔藓：控制 G - R 差异，避免太绿的叶片
    bright_green_moss = (
        (G > R) & (R > B) &
        (G >= 140) & (G <= 230) &
        (R >= 120) & (R <= 200) &
        (B < 120) &
        (gray >= 150) & (gray <= 230) &
        ((G - R) < 50)  # 🌿 避免非常绿的嫩叶子
    )

    # 黄褐色苔藓：限制 R-G/B 差值，控制灰度避免枯叶
    brownish_moss = (
        (R > G) & (G > B) &
        (R >= 140) & (R <= 180) &
        (G >= 110) & (G <= 150) &
        (B >= 80) & (B <= 120) &
        (gray >= 100) & (gray <= 160) &
        ((R - G) < 40) & ((G - B) < 40)  # 🍂 避免强烈颜色阶梯
    )
    # 深色苔藓（黑褐色调，RGB都低，G≈R≈B或G略大于R）
    dark_moss = (
        (G >= R - 10) & (R >= B - 10) &
        (R < 95) & (G < 100) & (B < 85) &
        (gray >= 20) & (gray <= 110)
    )

    # 橄榄绿/灰绿色苔藓（R、G差距小但都偏低，略偏绿）
    olive_moss = (
        (G > R - 10) & (R > B) &
        (R < 120) & (G < 135) & (B < 100) &
        (gray >= 30) & (gray <= 130)
    )

    # 红枯叶或土壤：R > G > B（我们反选）
    non_moss = (
        (
            (G > R) & (R > B) & (G > 130) & (gray > 140)   # 杂草、绿叶
        ) |
        (
            (B > R) & (R > G) & (B > 100)                  # 紫花蓝花
        ) |
        (
            (R > G) & (G > B) & (R > 120)                  # 枯叶、裸地
        )
    )
    moss_mask = ((green_moss | dark_moss | olive_moss | bright_green_moss | brownish_moss ) & ~non_moss).astype(np.uint8) * 255
    return moss_mask


def visualize_results(image, moss_mask, image_path):
    """Overlay a semi-transparent Windows-style selection blue on moss regions."""

    # Windows-style selection blue (BGR format)
    overlay_color = (0, 0, 255)   # BGR = RGB(51, 153, 255)
    # 创建调试输出目录
    debug_dir = "debug_pic"
    os.makedirs(debug_dir, exist_ok=True)
    # Create a blank color mask
    color_mask = np.zeros_like(image, dtype=np.uint8)
    color_mask[moss_mask == 255] = overlay_color

    # Alpha blend
    alpha = 0.5
    blended = cv2.addWeighted(image, 1.0, color_mask, alpha, 0)

    # Output path
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(debug_dir, f"{base_name}_overlay_mask.png")

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Original")

    axs[1].imshow(moss_mask, cmap='gray')
    axs[1].set_title("Moss Mask")

    axs[2].imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    axs[2].set_title("Overlay (Selection Blue)")

    for ax in axs:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"✅ Saved: {output_path}")

def calculate_coverage(moss_mask):
    """计算苔藓覆盖率（白色像素占比）"""
    total_pixels = moss_mask.size
    moss_pixels = np.sum(moss_mask == 255)
    return moss_pixels / total_pixels



CSV_PATH = "moss_coverage_full.csv"
WRITE_EVERY = 10  # 每处理10张写入一次，防止中途丢数据

def safe_append_to_csv(results, csv_path):
    """将结果列表写入csv，自动创建或追加"""
    df = pd.DataFrame(results)
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False, encoding="utf-8-sig")

def imread_unicode(path):
    import cv2
    import numpy as np
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

if __name__ == "__main__":
    root_folder = "./调查样方"
    image_files = []

    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg")):
                # 1. 拼出绝对路径
                abs_path = os.path.abspath(os.path.join(root, file))
                # 2. 统一用正斜杠（兼容 OpenCV、终端、Pandas 等）
                abs_path = abs_path.replace('\\', '/')
                image_files.append(abs_path)

    print(f"共找到 {len(image_files)} 张图片")
    processed_paths = set()
    if os.path.exists(CSV_PATH):
        # 读取已存在的路径，避免重复处理
        df_existing = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
        processed_paths = set(df_existing["图片路径"])

    results_buffer = []

    for idx, image_path in enumerate(tqdm(image_files, desc="处理中", unit="张"), 1):
        if image_path in processed_paths:
            tqdm.write(f" 已处理，跳过：{image_path}")
            continue

        if not os.path.exists(image_path):
            tqdm.write(f" 跳过：图像未找到: {image_path}")
            continue

        image = imread_unicode(image_path)
        if image is None:
            tqdm.write(f" 图像读取失败: {image_path}")
            continue

        try:
            cropped = find_and_crop_quadrat(image, filename=os.path.basename(image_path))
            moss_mask = detect_moss_by_rgb_gray(cropped)
            coverage = calculate_coverage(moss_mask)
            visualize_results(cropped, moss_mask, image_path)

            result = {
                "图片路径": image_path,
                "是否找到样方": "是" if cropped is not None else "否",
                "苔藓覆盖率 (%)": round(coverage * 100, 2)
            }
            tqdm.write(f" {image_path} 覆盖率: {coverage:.2%}")
            results_buffer.append(result)
        except Exception as e:
            tqdm.write(f" 处理出错: {image_path}，原因: {e}")
            result = {
                "图片路径": image_path,
                "是否找到样方": "否",
                "苔藓覆盖率 (%)": 0.0
            }
            results_buffer.append(result)

        if len(results_buffer) >= WRITE_EVERY or idx == len(image_files):
            safe_append_to_csv(results_buffer, CSV_PATH)
            tqdm.write(f" 已写入 {len(results_buffer)} 条结果到 {CSV_PATH}")
            results_buffer.clear()

