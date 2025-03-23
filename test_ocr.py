import cv2
import pytesseract
import numpy as np
from rapidfuzz.distance import Levenshtein

# Set Tesseract path if needed (Windows)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# === CONFIGURATION ===
IMAGE_PATH1 = "/home/miza/Magisterka/src/data/images/1.jpg"  # Input image path
GROUND_TRUTH1 = "SALATKA JARZY 250G-D 1 szt. x4,50 4,50D \
JOG ALE PITNY 290g-D 1szt. 3,60 3,60D \
KAJZERKA PREM 60g-D 1szt. x0,49 0,49 \
KAJZERKA PREM 60g-D 1szt. x0,49 0,49 \
"  # Replace with actual expected text
IMAGE_PATH2 = "/home/miza/Magisterka/src/data/images/10.jpg"  # Input image path
GROUND_TRUTH2 = "SC BAGIETKA SZYN/SER (B) K:4724 1 szt. x16.99 16.99B\
OBNIŻKA -11.24B\
SC KAJZERKA SZYN/SER (B) K:4738 1 szt. x12.99 12.99B\
OBNIŻKA -7.24B\
SC KAJZERKA MOZZARELLA/POMIDOR (B) K:8167 1 szt. x12.99 12.99B\
OBNIŻKA -7.24B\
SC MUFFIN BLACK&WHITE 100G (B) K:7879 1 szt. x7.99 7.99B\
OBNIŻKA -2.24B"
LANGUAGE = "pol"  # Polish language model


from utils.utils import (
    grayscale,
    otsu_threshold,
    adaptive_threshold,
    sauvola_threshold,
    gaussian_blur,
    gaussian_blur1,
    gaussian_blur2,
    median_blur,
    sharpen,
    resize,
    contrast_stretching,
    histogram_equalization,
    clahe,
    morphological_closing,
    morphological_opening,
    dilation,
    erosion,
    bilateral_filter,
    unsharp_mask,
)


from annotate_data import get_text


def evaluate_ocr(image, truth):
    text = get_text(image)
    full_text = " ".join(text)
    return Levenshtein.normalized_distance(truth, full_text)  # Lower score = better match


if __name__ == "__main__":
    preprocessing_steps = {
        "Otsu Thresholding": otsu_threshold,
        "Adaptive Thresholding": adaptive_threshold,
        "Sauvola Thresholding": sauvola_threshold,
        # "Gaussian Blur (5x5)": gaussian_blur,
        "Gaussian Blur (3x3)": gaussian_blur1,
        # "Gaussian Blur (1x1)": gaussian_blur2,
        "Median Blur": median_blur,
        "Sharpening": sharpen,
        "Resize": resize,
        "Contrast Stretching": contrast_stretching,
        "Histogram Equalization": histogram_equalization,
        "CLAHE": clahe,
        "Morphological Closing": morphological_closing,
        "Morphological Opening": morphological_opening,
        "Dilation": dilation,
        "Erosion": erosion,
        "Bilateral Filtering": bilateral_filter,
        "Unsharp Masking": unsharp_mask,
    }

    import cv2
    from itertools import permutations
    from tqdm import tqdm
    import concurrent.futures

    # Function to run the pipeline for a given combination of steps
    def run_pipeline(combination):
        img1 = original_img1.copy()
        img2 = original_img2.copy()
        for step in combination:
            img1 = preprocessing_steps[step](img1)
            img2 = preprocessing_steps[step](img2)
        score1 = evaluate_ocr(img1, GROUND_TRUTH1)
        score2 = evaluate_ocr(img2, GROUND_TRUTH2)
        score = (score1 + score2) / 2
        return combination, score

    # Load and prepare the original image
    original_img1 = cv2.imread(IMAGE_PATH1)
    original_img1 = grayscale(original_img1)

    original_img2 = cv2.imread(IMAGE_PATH2)
    original_img2 = grayscale(original_img2)

    # Create a list of all combinations (5-step pipelines)
    import random

    all_combinations = list(permutations(preprocessing_steps.keys(), 4))
    sampled_combinations = random.sample(all_combinations, 250000)
    best_score = float("inf")
    best_pipeline = None

    # Use ThreadPoolExecutor to evaluate pipelines in parallel using 6 threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Map the run_pipeline function over all combinations
        for combination, score in tqdm(executor.map(run_pipeline, all_combinations), total=len(all_combinations)):
            if score < best_score:
                best_score = score
                best_pipeline = combination
                print(f"Pipeline: {combination}, Score: {score:.4f}")

    print("\nBest Preprocessing Pipeline:", best_pipeline, "with score:", best_score)

    from pathlib import Path

    with open(Path(__file__).parent / "best_pipeline.txt", "w") as f:
        f.write("\n".join(best_pipeline))
