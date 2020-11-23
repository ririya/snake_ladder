import os
from sklearn.cluster import KMeans,Birch,MiniBatchKMeans
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltColors

board_image_path = 'snake_ladder_board.png'
results_dir = "results"

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

FORCE_CALCULATE_LADDER = True
FORCE_CALCULATE_LINES = True
FORCE_SEGMENT_IMAGE = True
DEBUG = False
GAUSS_KERNEL_SIZE = 5
MIN_LINE_LENGTH = 30
MAX_DIST_LINE = 20
MERGE_THRES = 15
ANGLE_TOL = 3
GAP_THRES = 100000
PATCH_SIZE = 10
LADDER_INT_THRES = 100
MIN_LADDER_LENGTH = 100
EPS = 0.05
INTERSECTION_THRES = 15
GAP_THRES = 100
GAP_THRES_2 = 30
CONTOUR_AREA_THRES_ARTIFACTS = 350
CONTOUR_AREA_THRES = 1000

HSVweights = [2,1,1]
Labweights = [1,4,4]

N_CLUSTERS = 6

otherClass = [2,3,0]

clusteringFunction = KMeans

COLOR_MAP = ['Black','Red','Blue','Green','Magenta', 'Cyan', 'Yellow', 'White']
COLOR_RGB_VALUES =  {'Black':np.array([0, 0, 0,255])/255,
                     'Red':np.array([255, 0, 0,255])/255,
                     'Blue':np.array([0, 0, 255,255])/255,
                    'Green':np.array([0, 255, 0,255])/255,
                    'Magenta':np.array([255, 0, 255,255])/255,
                     'Cyan':np.array([0,255,255,255])/255,
                     'Yellow': np.array([255, 255, 0, 255]) / 255,
                    'White':np.array([235,235,235,255])/255,
                     }

colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
          (255, 0, 255)]

viridis = plt.cm.get_cmap('viridis', N_CLUSTERS)
newcolors = viridis(np.linspace(0, 1, N_CLUSTERS))

for cind in range(N_CLUSTERS):
    c = COLOR_MAP[cind]
    newcolors[cind,:] = COLOR_RGB_VALUES[c]
newcmp = pltColors.ListedColormap(newcolors)