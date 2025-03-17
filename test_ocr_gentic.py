import cv2
import pytesseract
import numpy as np
from skimage.filters import threshold_sauvola
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


from annotate_data import get_text


def evaluate_ocr(image, truth):
    text = get_text(image)
    full_text = " ".join(text)
    return Levenshtein.normalized_distance(truth, full_text)  # Lower score = better match


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

# List of preprocessing steps to test
preprocessing_steps = {
    "Otsu Thresholding": otsu_threshold,
    "Adaptive Thresholding": adaptive_threshold,
    "Sauvola Thresholding": sauvola_threshold,
    "Gaussian Blur (5x5)": gaussian_blur,
    "Gaussian Blur (3x3)": gaussian_blur1,
    "Gaussian Blur (1x1)": gaussian_blur2,
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


import math
import random

# Main loop (run for a number of generations)
original_img1 = cv2.imread(IMAGE_PATH1)
original_img1 = grayscale(original_img1)

original_img2 = cv2.imread(IMAGE_PATH2)
original_img2 = grayscale(original_img2)


import cv2
import random
import math
from tqdm import tqdm

# Assume:
# - preprocessing_steps is a dictionary mapping step names to functions.
# - evaluate_ocr is a function that returns an OCR score (lower is better).
# - grayscale is a function that converts an image to grayscale.

# Load and prepare benchmark images
original_img1 = cv2.imread(IMAGE_PATH1)
original_img1 = grayscale(original_img1)

original_img2 = cv2.imread(IMAGE_PATH2)
original_img2 = grayscale(original_img2)


def evaluate_pipeline(pipeline, image, gt):
    """Apply the pipeline to a given image and return the OCR score."""
    img = image.copy()
    for step in pipeline:
        img = preprocessing_steps[step](img)
    return evaluate_ocr(img, gt)


def evaluate_pipeline_both(pipeline, img1, img2):
    """Evaluate the pipeline on both images and return the average OCR score."""
    score1 = evaluate_pipeline(pipeline, img1, GROUND_TRUTH1)
    score2 = evaluate_pipeline(pipeline, img2, GROUND_TRUTH2)
    return (score1 + score2) / 2.0


def fitness(pipeline):
    """
    Since lower OCR scores are better, we define fitness as the negative average score.
    This way, maximizing fitness corresponds to minimizing the OCR score.
    """
    return -evaluate_pipeline_both(pipeline, original_img1, original_img2)


# GA Parameters
population_size = 50
num_generations = 10
tournament_size = 3
mutation_rate = 0.2

# Create initial population (each individual is a permutation of 5 steps)
steps = list(preprocessing_steps.keys())
population = [random.sample(steps, 5) for _ in range(population_size)]


def tournament_selection(pop, fit, k):
    """Select one individual via tournament selection."""
    selected = random.sample(list(zip(pop, fit)), k)
    # higher fitness is better
    selected.sort(key=lambda x: x[1], reverse=True)
    return selected[0][0]


def order_crossover(parent1, parent2):
    """
    Order Crossover (OX) for permutations.
    This function creates a child by copying a random slice from parent1 and filling
    the remainder from parent2 in order.
    """
    size = len(parent1)
    child = [None] * size
    start, end = sorted(random.sample(range(size), 2))
    # Copy slice from parent1
    child[start:end] = parent1[start:end]
    # Fill the remaining positions with genes from parent2 in order
    fill_pos = end
    for gene in parent2[end:] + parent2[:end]:
        if gene not in child:
            if fill_pos >= size:
                fill_pos = 0
            child[fill_pos] = gene
            fill_pos += 1
    return child


def mutate(pipeline, rate):
    """Swap mutation: swap two random positions with a given probability."""
    new_pipeline = pipeline.copy()
    if random.random() < rate:
        i, j = random.sample(range(len(new_pipeline)), 2)
        new_pipeline[i], new_pipeline[j] = new_pipeline[j], new_pipeline[i]
    return new_pipeline


# Evaluate initial fitnesses
fitnesses = [fitness(ind) for ind in population]

best_individual = None
best_fitness = -math.inf

# GA Main Loop with progress tracking via tqdm
for gen in tqdm(range(num_generations), desc="GA Generations"):
    new_population = []
    for _ in range(population_size):
        # Selection
        parent1 = tournament_selection(population, fitnesses, tournament_size)
        parent2 = tournament_selection(population, fitnesses, tournament_size)
        # Crossover
        child = order_crossover(parent1, parent2)
        # Mutation
        child = mutate(child, mutation_rate)
        new_population.append(child)
    population = new_population
    fitnesses = [fitness(ind) for ind in population]

    current_best_fitness = max(fitnesses)
    if current_best_fitness > best_fitness:
        best_fitness = current_best_fitness
        best_individual = population[fitnesses.index(current_best_fitness)]

print("Best Preprocessing Pipeline:", best_individual, "with average OCR score:", -best_fitness)
