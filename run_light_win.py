import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm


def find_and_crop_quadrat(image, filename=None, output_size=(500, 500)):
    """Detect quadrat region and perform perspective crop. If detection fails, crop center square region."""
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
        print("⚠️ Quadrat not found. Cropping center region instead.")
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
    """Detect moss regions using RGB pattern + grayscale rules"""

    B, G, R = cv2.split(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Green moss (G > R > B, and relatively low intensity)
    green_moss = (
        (G > R) & (R > B) &
        (G < 140) & (R < 130) & (B < 110) &
        (G - R > 2) & (R - B > 5) &
        (gray > 60) & (gray < 160)
    )
    # Bright green moss: limit G - R difference
    bright_green_moss = (
        (G > R) & (R > B) &
        (G >= 140) & (G <= 230) &
        (R >= 120) & (R <= 200) &
        (B < 120) &
        (gray >= 150) & (gray <= 230) &
        ((G - R) < 50)
    )

    # Brownish moss: restrict R-G/B difference and gray range
    brownish_moss = (
        (R > G) & (G > B) &
        (R >= 140) & (R <= 180) &
        (G >= 110) & (G <= 150) &
        (B >= 80) & (B <= 120) &
        (gray >= 100) & (gray <= 160) &
        ((R - G) < 40) & ((G - B) < 40)
    )
    # Dark moss: low RGB values, G≈R≈B or G slightly greater
    dark_moss = (
        (G >= R - 10) & (R >= B - 10) &
        (R < 95) & (G < 100) & (B < 85) &
        (gray >= 20) & (gray <= 110)
    )

    # Olive / grayish moss (R and G are close and low)
    olive_moss = (
        (G > R - 10) & (R > B) &
        (R < 120) & (G < 135) & (B < 100) &
        (gray >= 30) & (gray <= 130)
    )

    # Exclude non-moss (weeds, flowers, soil)
    non_moss = (
        (
            (G > R) & (R > B) & (G > 130) & (gray > 140)
        ) |
        (
            (B > R) & (R > G) & (B > 100)
        ) |
        (
            (R > G) & (G > B) & (R > 120)
        )
    )
    moss_mask = ((green_moss | dark_moss | olive_moss | bright_green_moss | brownish_moss ) & ~non_moss).astype(np.uint8) * 255
    return moss_mask


def visualize_results(image, moss_mask, image_path):
    """Overlay a semi-transparent blue mask on moss regions"""
    overlay_color = (0, 0, 255)  # BGR
    debug_dir = "debug_pic"
    os.makedirs(debug_dir, exist_ok=True)

    color_mask = np.zeros_like(image, dtype=np.uint8)
    color_mask[moss_mask == 255] = overlay_color

    alpha = 0.5
    blended = cv2.addWeighted(image, 1.0, color_mask, alpha, 0)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(debug_dir, f"{base_name}_overlay_mask.png")

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Original")

    axs[1].imshow(moss_mask, cmap='gray')
    axs[1].set_title("Moss Mask")

    axs[2].imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    axs[2].set_title("Overlay")

    for ax in axs:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"✅ Saved: {output_path}")

def calculate_coverage(moss_mask):
    """Calculate moss coverage percentage (white pixels ratio)"""
    total_pixels = moss_mask.size
    moss_pixels = np.sum(moss_mask == 255)
    return moss_pixels / total_pixels

CSV_PATH = "moss_coverage_full.csv"
WRITE_EVERY = 10  # Save every 10 images to avoid data loss

def safe_append_to_csv(results, csv_path):
    """Append results to CSV, create file if not exists"""
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
    root_folder = "./moss_quadrats"
    image_files = []

    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg")):
                abs_path = os.path.abspath(os.path.join(root, file))
                abs_path = abs_path.replace('\\', '/')
                image_files.append(abs_path)

    print(f"Found {len(image_files)} images.")
    processed_paths = set()
    if os.path.exists(CSV_PATH):
        df_existing = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
        processed_paths = set(df_existing["Image Path"])

    results_buffer = []

    for idx, image_path in enumerate(tqdm(image_files, desc="Processing", unit="image"), 1):
        if image_path in processed_paths:
            tqdm.write(f"✅ Already processed, skipping: {image_path}")
            continue

        if not os.path.exists(image_path):
            tqdm.write(f"⚠️ Skipped: file not found: {image_path}")
            continue

        image = imread_unicode(image_path)
        if image is None:
            tqdm.write(f"❌ Failed to read image: {image_path}")
            continue

        try:
            cropped = find_and_crop_quadrat(image, filename=os.path.basename(image_path))
            moss_mask = detect_moss_by_rgb_gray(cropped)
            coverage = calculate_coverage(moss_mask)
            visualize_results(cropped, moss_mask, image_path)

            result = {
                "Image Path": image_path,
                "Quadrat Detected": "Yes" if cropped is not None else "No",
                "Moss Coverage (%)": round(coverage * 100, 2)
            }
            tqdm.write(f"🌿 {image_path} coverage: {coverage:.2%}")
            results_buffer.append(result)
        except Exception as e:
            tqdm.write(f"❌ Error: {image_path}, Reason: {e}")
            result = {
                "Image Path": image_path,
                "Quadrat Detected": "No",
                "Moss Coverage (%)": 0.0
            }
            results_buffer.append(result)

        if len(results_buffer) >= WRITE_EVERY or idx == len(image_files):
            safe_append_to_csv(results_buffer, CSV_PATH)
            tqdm.write(f"📥 Saved {len(results_buffer)} results to {CSV_PATH}")
            results_buffer.clear()

