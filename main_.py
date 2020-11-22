
from collections import defaultdict
import glob
import pickle
import os
import time
import math
import tqdm

from sklearn.cluster import KMeans,Birch,MiniBatchKMeans
import matplotlib.pyplot as plt
import matplotlib.colors as pltColors
from PIL import Image
import numpy as np
import cv2

import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import skimage.color as skcolor


def shortest_distance(x1, y1, a, b, c):
    d = abs((a * x1 + b * y1 + c)) / (math.sqrt(a * a + b * b))
    return d

def getPixelSequence(im, allPixels, COLOR_SPACE):
    imNp = np.asarray(im)
    imNp.reshape((-1, 3))
    pixelSequence = imNp.reshape((-1, 3))
    pixelSequence = np.array(pixelSequence)
    if COLOR_SPACE == 'HSV':
        for i in range(len(HSVweights)):
            pixelSequence[:,i] = pixelSequence[:,i]*HSVweights[i]
    elif COLOR_SPACE == 'Lab':
        for i in range(len(Labweights)):
            pixelSequence[:,i] *= Labweights[i]
    pixelSequence = np.float32(pixelSequence) / 255
    allPixels = np.append(allPixels, pixelSequence, axis=0)
    return imNp,pixelSequence,allPixels


def find_merging_group(groups, angle, b1, MERGE_THRES, ANGLE_TOL):

    min_dist = float('inf')
    best_ind = -1
    for ind_group, group in enumerate(groups):

        line_set2 = group[0]

        p1line2 = line_set2[0][0][0]
        p2line2 = line_set2[0][0][1]

        m2 = (p2line2[1] - p1line2[1]) / (p2line2[0] - p1line2[0])
        angle2 = math.degrees(math.atan2(p2line2[1] - p1line2[1], (p2line2[0] - p1line2[0])))

        if abs(angle - angle2) < ANGLE_TOL:

            b2 = p1line2[1] - m2 * p1line2[0]
            d = abs(b2 - b1) / ((m2 * m2) + 1)
            if d < MERGE_THRES:
                if d < min_dist:
                    min_dist = d
                    best_ind = ind_group
    return best_ind

def euclid_distance(p1,p2):

    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

board_image_path = 'snake_ladder_board.png'

# HSVweights = [2,0,1]
# Labweights = [1,2,2]

HSVweights = [2,1,1]
Labweights = [1,4,4]
# Labweights = [1,1,1]

# COLOR_SPACE = "HSV"
N_CLUSTERS = 4

otherClass = [2,3,0]

clusteringFunction = KMeans

COLOR_MAP = [['White','Brown','Gray','Black']]
COLOR_RGB_VALUES =  {'White':np.array([235,235,235,255])/255, 'Brown':np.array([176, 108, 73,255])/255, 'Gray':np.array([180, 180, 180,255])/255, 'Black':np.array([0, 0, 0,255])/255}

viridis = plt.cm.get_cmap('viridis', len(COLOR_MAP[0]))
newcolors = viridis(np.linspace(0, 1, len(COLOR_MAP[0])))

for cind, c in enumerate(COLOR_MAP[0]):
    newcolors[cind,:] = COLOR_RGB_VALUES[c]
newcmp = pltColors.ListedColormap(newcolors)


allPercentage = np.empty((0,N_CLUSTERS))

allPixels = np.empty((0,3))
allPixelsLab = np.empty((0,3))
allPixelsHSV = np.empty((0,3))


# im = Image.open(board_image_path)
# im = im.convert(mode="RGB")


img = cv2.imread("snake_ladder_board.png")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

detector = cv2.createLineSegmentDetector()

lines = detector.detect(blur_gray)

# edges = cv2.Canny(blur_gray,50,150,apertureSize = 3)
# edges = cv2.Canny(gray,0,255)
#
#
#
#
# plt.figure()
# implot = plt.imshow(edges,cmap = newcmp)
# plt.show()



# minLineLength = 100
# maxLineGap = 10
# lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
# lines = [lines]
#

# ANGLE_TOL = 3
#
# line_image = np.zeros_like(img)
# for line in lines[0]:
#     x1,y1,x2,y2 = line[0]
#
#     angle = math.atan2(y2 - y1, x2 - x1)
#     angle = math.degrees(angle)
#
#     if abs(abs(angle) - 90) > ANGLE_TOL and abs(abs(angle) - 0) > ANGLE_TOL and abs(abs(\
#             angle) - 180) > ANGLE_TOL and abs(abs(angle) - 360) > ANGLE_TOL:
#             cv2.line(line_image,(x1,y1),(x2,y2),(255,255,255),5)
#
# line_image = cv2.cvtColor(line_image,cv2.COLOR_BGR2GRAY)
# #
# lines = detector.detect(line_image)
#
line_image = np.zeros_like(img)

for line in lines[0]:
    x1,y1,x2,y2 = line[0]
    cv2.line(line_image,(x1,y1),(x2,y2),(255,255,255),2)

cv2.imwrite('lines2.png',line_image)
#
# plt.figure()
# implot = plt.imshow(line_image,cmap = newcmp)
# plt.show()


MIN_LINE_LENGTH = 30
MAX_DIST_LINE = 15
MERGE_THRES = 20
ANGLE_TOL = 3
GAP_THRES = 100000

line_sets = []

# lines = np.sort(lines[0])

lines = lines[0].tolist()

lines.sort()
n_lines = len(lines)

line_sets_path = "line_sets.pkl"

FORCE_CALCULATE_LINE_SETS = True

if os.path.exists(line_sets_path) and not FORCE_CALCULATE_LINE_SETS:
    line_sets = pickle.load(open(line_sets_path, "rb" ))

else:

    for ind_line in tqdm.tqdm(range(n_lines)):

        line = lines[ind_line]
        # for x1,y1,x2,y2 in line:

        x1,y1,x2,y2 = line[0]

        line_length = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)

        if line_length < MIN_LINE_LENGTH:
            continue

        angle = math.atan2(y2 - y1, x2 - x1)
        angle = math.degrees(angle)

        if abs(abs(angle) - 90) < ANGLE_TOL or abs(abs(angle) - 0) < ANGLE_TOL or abs(abs( \
                            angle) - 180) < ANGLE_TOL or abs(abs(angle) - 360) < ANGLE_TOL:
            continue

        found_line = False

        p1 = np.array([x1, y1], dtype=int)
        p2 = np.array([x2, y2], dtype=int)

        for ind_line_set, line_set in enumerate(line_sets):

            p1line = line_set[0][-1][0]
            p2line = line_set[0][-1][1]

            # angle2 = math.atan2(p2line[1] - p1line[1], p2line[0] - p1line[0])
            # angle2 = math.degrees(angle)

            # X = line_set[1]
            # Y = line_set[2]

            # coef = np.polyfit(X, Y, 1)
            # a = coef[0]
            # b = coef[1]
            # c = -a * X[0] - b * Y[0]
            #
            # angle = math.atan(a)
            # angle = math.degrees(angle)
            #
            # a_ = (p2line[1] - p1line[1]) / (p2line[0] - p1line[0])
            # b_ = p1line[1] - a*p1line[0]
            # c_ = -a * p1line[0] - b * p1line[1]

            # d1_ = shortest_distance(x1, y1, a, b, c)
            # d2_ = shortest_distance(x2, y2, a, b, c)

            d1_ = abs(np.cross(p2line - p1line, p1 - p1line) / np.linalg.norm(p2line - p1line))
            d2_ = abs(np.cross(p2line - p1line, p2 - p1line) / np.linalg.norm(p2line - p1line))

            # if abs(angle - angle2) < ANGLE_TOL and  d1_ < MAX_DIST_LINE and d2_ < MAX_DIST_LINE and \
            #         d1_ + d2_ < min_dist:

            gap_dist = euclid_distance(p1, p2line)

            if d1_ < MAX_DIST_LINE and d2_ < MAX_DIST_LINE:

                found_line = True
                break

        if not found_line:
            line_sets.append([[[p1, p2]],[x1,x2],[y1,y2]])

    colors = [(255, 255, 255), (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
              (255, 0,255)]
    line_image_clusters = img.copy()

    ind_color = 0

    for line_set in line_sets:

        for point in line_set[0]:
            x1, y1 = point[0]
            x2, y2 = point[1]

            cv2.line(line_image_clusters, (x1, y1), (x2, y2), colors[ind_color], 2)

        ind_color = (ind_color + 1) % len(colors)

    cv2.imwrite('orig_line_clusters.png', line_image_clusters)

    MIN_LINE_LENGTH = 0

    for ind_line in tqdm.tqdm(range(n_lines)):

        line = lines[ind_line]
        # for x1,y1,x2,y2 in line:

        x1,y1,x2,y2 = line[0]

        line_length = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)

        if line_length < MIN_LINE_LENGTH:
            continue

        angle = math.atan2(y2 - y1, x2 - x1)
        angle = math.degrees(angle)

        if abs(abs(angle) - 90) < ANGLE_TOL or abs(abs(angle) - 0) < ANGLE_TOL or abs(abs( \
                            angle) - 180) < ANGLE_TOL or abs(abs(angle) - 360) < ANGLE_TOL:
            continue

        p1 = np.array([x1, y1], dtype=int)
        p2 = np.array([x2, y2], dtype=int)

        found_line = False

        min_dist = float('inf')

        for ind_line_set, line_set in enumerate(line_sets):

            p1line = line_set[0][0][0]
            p2line = line_set[0][0][1]

            angle2 = math.atan2(p2line[1] - p1line[1], p2line[0] - p1line[0])
            # if angle2 < 0:
            #     angle2 = 2 * math.pi + angle2
            angle2 = math.degrees(angle2)

            # X = line_set[1]
            # Y = line_set[2]

            # coef = np.polyfit(X, Y, 1)
            # a = coef[0]
            # b = coef[1]
            # c = -a * X[0] - b * Y[0]
            #
            # angle = math.atan(a)
            # angle = math.degrees(angle)
            #
            # a_ = (p2line[1] - p1line[1]) / (p2line[0] - p1line[0])
            # b_ = p1line[1] - a*p1line[0]
            # c_ = -a * p1line[0] - b * p1line[1]

            # d1_ = shortest_distance(x1, y1, a, b, c)
            # d2_ = shortest_distance(x2, y2, a, b, c)

            d1_ = abs(np.cross(p2line - p1line, p1 - p1line) / np.linalg.norm(p2line - p1line))
            d2_ = abs(np.cross(p2line - p1line, p2 - p1line) / np.linalg.norm(p2line - p1line))

            angle_diff = abs(angle - angle2)

            if angle_diff < ANGLE_TOL and  d1_ < MAX_DIST_LINE and d2_ < MAX_DIST_LINE and \
                    d1_ + d2_ < min_dist:

            # gap_dist = euclid_distance(p1, p2line)
            #
            # if gap_dist < GAP_THRES and d1_ < MAX_DIST_LINE and d2_ < MAX_DIST_LINE and d1_ + d2_ < \
            #         min_dist:

                min_dist = d1_ + d2_
                min_ind = ind_line_set
                found_line = True

        if found_line:
            line_sets[min_ind][0].append([p1,p2])
            line_sets[min_ind][1].extend([x1, x2])
            line_sets[min_ind][2].extend([y1, y2])

        # else:
        #     line_sets.append([[[p1, p2]],[x1,x2],[y1,y2]])

    pickle.dump(line_sets, open( "line_sets.pkl", "wb" ))

GAP_THRES = 700
#
for ind_line_set in tqdm.tqdm(range(len(line_sets))):

    line_set = line_sets[ind_line_set]
    line_segs = line_set[0]

    new_line_segs = []

    for ind_line_seg, line_seg in enumerate(line_segs):

        min_dist = float('inf')

        for ind_line_seg_2, line_seg_2 in enumerate(line_segs):

            if ind_line_seg_2 <= ind_line_seg:
                continue

            gap_dist = euclid_distance(line_seg[1], line_seg_2[0])
            min_dist = min(gap_dist, min_dist)
            gap_dist = euclid_distance(line_seg[1], line_seg_2[1])
            min_dist = min(gap_dist, min_dist)
            gap_dist = euclid_distance(line_seg[0], line_seg_2[1])
            min_dist = min(gap_dist, min_dist)
            gap_dist = euclid_distance(line_seg[0], line_seg_2[0])
            min_dist = min(gap_dist, min_dist)

        # gap_dist = euclid_distance(last_point, point[0])
        if min_dist > GAP_THRES:
            continue

        new_line_segs.append(line_seg)

    line_sets[ind_line_set][0] = new_line_segs


#merging lines that are too close

groups = []

non_empty_lines = 0

for line_set in line_sets:

    if not line_set[0]:
        continue

    non_empty_lines += 1

    p1line = line_set[0][0][0]
    p2line = line_set[0][0][1]

    m = (p2line[1] - p1line[1])/ (p2line[0] - p1line[0])
    angle = math.degrees(math.atan2(p2line[1] - p1line[1], (p2line[0] - p1line[0])))

    b1 = p1line[1] - m * p1line[0]

    merge_group = find_merging_group(groups, angle, b1, MERGE_THRES,ANGLE_TOL)

    # merge_group = -1

    if merge_group == -1:
        groups.append([line_set])
    else:
        groups[merge_group].append(line_set)

line_image_clusters = img
stair_start_end = img.copy()
line_image_clusters2 = np.zeros_like(img)
line_image_clusters3 = np.zeros_like(img)


group_len = []

for group in groups:

    min_x = float('inf')
    max_x = -1

    curr_group = np.zeros_like(img)

    for line_set in group:

        for point in line_set[0]:
            x1,y1 = point[0]
            x2,y2 = point[1]

            if x1 < min_x:
                min_point = (x1,y1)
                min_x = x1
            if x2 > max_x:
                max_point = (x2,y2)
                max_x = x2

            cv2.line(line_image_clusters, (x1, y1), (x2, y2), colors[ind_color], 2)
            cv2.line(line_image_clusters2, (x1, y1), (x2, y2), colors[ind_color], 2)
            cv2.line(curr_group, (x1, y1), (x2, y2), colors[ind_color], 2)

    # plt.figure()
    # implot = plt.imshow(curr_group)
    # plt.show()

    cv2.circle(stair_start_end, min_point, 20, colors[ind_color], 5)
    cv2.circle(stair_start_end, max_point, 20, colors[ind_color], 5)

    group_len.append(math.sqrt((max_point[0] - min_point[0])**2 + (max_point[1] - min_point[1])**2))

    ind_color = (ind_color + 1) % len(colors)

for ind_group, group in enumerate(groups):

    if group_len[ind_group] < 200:
        continue

    for line_set in group:

        for point in line_set[0]:
            x1,y1 = point[0]
            x2,y2 = point[1]

            cv2.line(line_image_clusters3, (x1, y1), (x2, y2), colors[ind_color], 2)

    ind_color = (ind_color + 1) % len(colors)

cv2.imwrite('_line_clusters.png', line_image_clusters)
cv2.imwrite('_line_clusters2.png', line_image_clusters2)
cv2.imwrite('_line_clusters3.png', line_image_clusters3)
cv2.imwrite('stair_locations.png', stair_start_end)


    #
    # cv2.imwrite('lines.png',line_image)
    #
    # plt.figure()
    # implot = plt.imshow(line_image,cmap = newcmp)
    # plt.show()



                # segments_per_angle(angle).append(x1,y1,x2,y2)

            # cv2.line(line_image,(x1,y1),(x2,y2),(255,255,255),5)

# kernel = np.ones((10,10),np.uint8)
# line_image = cv2.morphologyEx(line_image, cv2.MORPH_CLOSE, kernel)




# line_image = np.zeros(img.shape)
# for line in lines[0]:
#     for x1,y1,x2,y2 in line:
#         cv2.line(line_image,(x1,y1),(x2,y2),(255,255,255),2)
#
# cv2.imwrite('lines.png',line_image)

#
#
# imHSV = im.convert(mode="HSV")
# imLab = skcolor.rgb2lab(im)
#
# imNp, pixelSequence, allPixels = getPixelSequence(im, allPixels,'RGB')
# imNpLab, pixelSequenceLab, allPixelsLab = getPixelSequence(imLab, allPixelsLab,'Lab')
# imNpHSV, pixelSequenceHSV, allPixelsHSV = getPixelSequence(imHSV, allPixelsHSV, 'HSV')
#
# print('Computing clustering RGB')
# # kmeansRGB = clusteringFunction(n_clusters=N_CLUSTERS, threshold=0.2).fit(allPixels)
# kmeansRGB = clusteringFunction(n_clusters=N_CLUSTERS).fit(allPixels)
# print('Computing clustering HSV')
# # kmeansHSV = clusteringFunction(n_clusters=N_CLUSTERS, threshold=0.2).fit(allPixelsHSV)
# kmeansHSV = clusteringFunction(n_clusters=N_CLUSTERS).fit(allPixelsHSV)
#
# print('Computing clustering Lab')
# # kmeansLab = clusteringFunction(n_clusters=N_CLUSTERS, threshold=0.2).fit(allPixelsLab)
# kmeansLab = clusteringFunction(n_clusters=N_CLUSTERS).fit(allPixelsLab)
#
# Images = [imNp, imNpHSV, imNpLab]
# pixelsFormats = [allPixels, allPixelsHSV,allPixelsLab ]
# colorFormats = ['RGB', 'HSV', 'LAB']
# kmeans = [kmeansLab, kmeansHSV,kmeansLab]
#
# subPlotInd = 1
#
# for indFormat in range(len(colorFormats)):
#
#     colorFormat = colorFormats[indFormat]
#
#     startKmeans = time.time()
#     labels = kmeans[indFormat].predict(pixelsFormats[indFormat])
#     endKmeans = time.time()
#     print('Elapsed Time Kmeans: ' + str(endKmeans - startKmeans) + ' seconds')
#
#     # percentage = np.zeros(N_CLUSTERS)
#     #
#     # for c in range(N_CLUSTERS):
#     #    ind_c = np.where(labels==c)
#     #    percentage[c] = len(ind_c[0])/len(labels)*100
#     #
#     # if colorFormats[indFormat] == 'Lab':
#     #     allPercentage = np.append(allPercentage,np.expand_dims(percentage,axis=0), axis = 0)
#
#     labels = labels.reshape((im.size[1], im.size[0]))
#
#     titles = [colorFormat, 'Segmented Img']
#     images = [Images[indFormat], labels]
#
#     for j in range(len(titles)):
#         plt.subplot(len(colorFormats), 2, subPlotInd)
#         implot = plt.imshow(images[j],cmap = newcmp)
#         plt.axis('off')
#         plt.title(titles[j])
#         subPlotInd += 1
#
# plt.show()
# pass