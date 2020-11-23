import pickle
import os
import cv2
from params import *
import ladder_detection
import snake_detection

img = cv2.imread(board_image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur_gray = cv2.GaussianBlur(gray, (GAUSS_KERNEL_SIZE, GAUSS_KERNEL_SIZE), 0)
img_shape = img.shape

ladder_cache_path = os.path.join(results_dir, "ladders.pkl")

if os.path.exists(ladder_cache_path) and not FORCE_CALCULATE_LADDER:
    ladders, ladder_end_points, lines = pickle.load(open(ladder_cache_path, "rb"))
else:

    print('Deteting ladders...')

    ladders, ladder_end_points, lines = ladder_detection.detect_ladders(img, blur_gray, colors,
                                                                 results_dir)

    pickle.dump([ladders,ladder_end_points, lines], open(ladder_cache_path, "wb"))


print('Deteting snakes...')
snake_head_segmentation,snake_heads,snake_tails = snake_detection.detect_snakes(img,blur_gray,
                                                                             ladders, lines)

print("Done")
