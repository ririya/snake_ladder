
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

def get_angle(x1,x2,y1,y2):

    angle = math.atan2(y2 - y1, x2 - x1)
    angle = math.degrees(angle)

    # if angle < 0:
    #     angle += 360

    return angle

def find_merging_group(groups, ind_line,line, MERGE_THRES, ANGLE_TOL):


    m1,b1,angle,p1_1,p2_1 = get_linear_coefs(line[0][0])

    min_dist = float('inf')
    best_ind = -1
    for ind_group, group in enumerate(groups):

        line2 = group[0]

        m2, b2, angle2, p1_2, p2_2 = get_linear_coefs(line2[0][0])

        if ind_group == 34 and ind_line == 63:
            if DEBUG:
                img_debug = img.copy()
                cv2.line(img_debug, (p1_1[0], p1_1[1]), (p2_1[0], p2_1[1]), (0, 255, 0),
                         5)
                cv2.line(img_debug, (p1_2[0], p1_2[1]), (p2_2[0], p2_2[1]), (0, 0, 255),
                         5)

                curr_fig = plt.figure()
                plt.imshow(img_debug)
                plt.show()
                plt.close(curr_fig)



        if abs(angle - angle2) < ANGLE_TOL:

            min_dist = float('inf')

            for line_seg1 in line[0]:

                for line_seg2 in line2[0]:

                    d,_p1_1,_p2_1, _p1_2, _p2_2 = get_line_seg_distance(line_seg1, line_seg2)



                    if d < MERGE_THRES:

                        if DEBUG:
                            img_debug = img.copy()
                            cv2.line(img_debug, (p1_1[0], p1_1[1]), (p2_1[0], p2_1[1]), (0, 255, 0),
                                     5)
                            cv2.line(img_debug, (p1_2[0], p1_2[1]), (p2_2[0], p2_2[1]), (0, 0, 255),
                                     5)

                            # cv2.line(img_debug, (_p1_1[0], _p1_1[1]), (_p2_1[0], _p2_1[1]), (255,
                            #                                                         255, 0),5)
                            # cv2.line(img_debug, (_p1_2[0], _p1_2[1]), (_p2_2[0], _p2_2[1]), (0,255,
                            #                                                              255),5)


                            curr_fig = plt.figure()
                            plt.imshow(img_debug)
                            plt.show()
                            plt.close(curr_fig)

                        return ind_group

    return best_ind

def reorder_points(p1,p2):

    combined = [p1.tolist(), p2.tolist()]
    combined.sort()

    return np.array(combined[0]),np.array(combined[1])


def get_linear_coefs(line_seg):

    p1 = line_seg[0]
    p2 = line_seg[1]

    # p1,p2 = reorder_points(p1,p2)

    m = (p2[1] - p1[1]) / (p2[0] - p1[0])
    angle = get_angle(p1[0],p2[0],p1[1],p2[1])

    b = p1[1] - m * p1[0]

    return m,b,angle,p1,p2


def get_line_seg_distance(line_seg1, line_seg2):

    m1,b1,_,p1_1,p2_1 = get_linear_coefs(line_seg1)
    m2, b2,_,p1_2,p2_2 = get_linear_coefs(line_seg2)
    #
    # d = abs(b2 - b1) / math.sqrt(((m2 * m2) + 1))

    d = euclid_distance(p1_1,p1_2)
    d = min(d,euclid_distance(p1_1, p2_2))
    d = min(d, euclid_distance(p2_1, p1_2))
    d = min(d, euclid_distance(p2_1, p2_2))

    if d < MERGE_THRES:

        if DEBUG:
            img_debug = img.copy()
            cv2.line(img_debug, (p1_1[0], p1_1[1]), (p2_1[0], p2_1[1]), (0, 255, 0),
                     5)
            cv2.line(img_debug, (p1_2[0], p1_2[1]), (p2_2[0], p2_2[1]), (0, 0, 255),
                     5)
            curr_fig = plt.figure()
            plt.imshow(img_debug)
            plt.show()
            plt.close(curr_fig)

    return d,p1_1,p2_1, p1_2, p2_2



def euclid_distance(p1,p2):

    d = np.linalg.norm(np.array(p1)-np.array(p2))
    # d2 = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    if d == 0:
        return float('inf')
    return d

def is_shadow(p1line,p2line, img):

    img_shape = img.shape

    min_int = []
    avg_int = []

    for pline in [p1line, p2line]:

        patch_x1 = max(1, pline[0] - int(PATCH_SIZE / 2))
        patch_x2 = min(img_shape[1], pline[0] + int(PATCH_SIZE / 2))

        patch_y1 = max(1, pline[1] - int(PATCH_SIZE / 2))
        patch_y2 = min(img_shape[0], pline[1] + int(PATCH_SIZE / 2))

        patch = blur_gray[patch_y1:patch_y2, patch_x1:patch_x2]

        avg_int.append(np.average(patch[:]))
        min_int.append(np.min(patch[:]))


    # if DEBUG:
    #     img_debug = img.copy()
    #     cv2.line(img_debug, (p1line[0], p1line[1]), (p2line[0], p2line[1]), (0,255,0), 5)
    #     curr_fig = plt.figure()
    #     plt.imshow(img_debug)
    #     plt.show()
    #     plt.close(curr_fig)

    return max(min_int) > LADDER_INT_THRES


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

img = cv2.imread(board_image_path)

img_shape = img.shape

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

detector = cv2.createLineSegmentDetector()

line_segs = detector.detect(blur_gray)

line_image = np.zeros_like(img)

for line_seg in line_segs[0]:
    x1,y1,x2,y2 = line_seg[0]
    cv2.line(line_image,(x1,y1),(x2,y2),(255,255,255),2)

cv2.imwrite('line_segs2.png',line_image)


DEBUG = True
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


lines = []

line_segs = line_segs[0].tolist()

line_segs.sort()
n_line_segs = len(line_segs)

lines_path = "lines.pkl"

FORCE_CALCULATE_LINES = True

if os.path.exists(lines_path) and not FORCE_CALCULATE_LINES:
    lines = pickle.load(open(lines_path, "rb" ))

else:

    for ind_line_seg in tqdm.tqdm(range(n_line_segs)):

        line_seg = line_segs[ind_line_seg]

        x1,y1,x2,y2 = line_seg[0]

        line_length = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)

        if line_length < MIN_LINE_LENGTH:
            continue

        p1 = np.array([x1, y1], dtype=int)
        p2 = np.array([x2, y2], dtype=int)

        p1, p2 = reorder_points(p1, p2)

        angle = get_angle(p1[0],p2[0],p1[1],p2[1])

        if abs(abs(angle) - 90) < ANGLE_TOL or abs(abs(angle) - 0) < ANGLE_TOL or abs(abs( \
                            angle) - 180) < ANGLE_TOL or abs(abs(angle) - 360) < ANGLE_TOL:
            continue

        found_line = False

        if is_shadow(p1, p2, img):
            continue

        for ind_line, line in enumerate(lines):

            p1line = line[0][-1][0]
            p2line = line[0][-1][1]

            angle2 = get_angle(p1line[0], p2line[0], p1line[1], p2line[1])

            d1_ = abs(np.cross(p2line - p1line, p1 - p1line) / np.linalg.norm(p2line - p1line))
            d2_ = abs(np.cross(p2line - p1line, p2 - p1line) / np.linalg.norm(p2line - p1line))

            if abs(angle - angle2) < ANGLE_TOL and d1_ < MAX_DIST_LINE and d2_ < MAX_DIST_LINE:

                found_line = True
                break

        if not found_line:
            lines.append([[[p1, p2]],[x1,x2],[y1,y2],[],[]])

            # if DEBUG:
            #     curr_ind = len(lines) - 1
            #     img_debug = img.copy()
            #     cv2.line(img_debug, (p1[0], p1[1]), (p2[0], p2[1]), (0,255,0), 5)
            #     # curr_fig = plt.figure()
            #     # plt.imshow(img_debug)
            #     # plt.show()
            #     # plt.close(curr_fig)
            #     cv2.imwrite(str.format("line_stems/{}.png", curr_ind), img_debug)

    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
              (255, 0,255)]

    img_lines = img.copy()

    ind_color = 0

    for line in lines:

        for point in line[0]:
            x1, y1 = point[0]
            x2, y2 = point[1]

            cv2.line(img_lines, (x1, y1), (x2, y2), colors[ind_color], 2)

        ind_color = (ind_color + 1) % len(colors)

    cv2.imwrite('orig_line_clusters.png', img_lines)

    MIN_LINE_LENGTH = 0

    for ind_line_seg in tqdm.tqdm(range(n_line_segs)):

        line = line_segs[ind_line_seg]

        x1,y1,x2,y2 = line[0]

        line_length = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)

        if line_length < MIN_LINE_LENGTH:
            continue

        p1 = np.array([x1, y1], dtype=int)
        p2 = np.array([x2, y2], dtype=int)

        p1, p2 = reorder_points(p1, p2)

        angle = get_angle(p1[0], p2[0], p1[1], p2[1])

        if abs(abs(angle) - 90) < ANGLE_TOL or abs(abs(angle) - 0) < ANGLE_TOL or abs(abs( \
                            angle) - 180) < ANGLE_TOL or abs(abs(angle) - 360) < ANGLE_TOL:
            continue

        if is_shadow(p1, p2, img):
            continue

        found_line = False

        min_dist = float('inf')

        for ind_line, line in enumerate(lines):

            p1line = line[0][0][0]
            p2line = line[0][0][1]

            angle2 = get_angle(p1line[0],p2line[0],p1line[1],p2line[1])

            d1_ = abs(np.cross(p2line - p1line, p1 - p1line) / np.linalg.norm(p2line - p1line))
            d2_ = abs(np.cross(p2line - p1line, p2 - p1line) / np.linalg.norm(p2line - p1line))

            angle_diff = abs(angle - angle2)

            if angle_diff < ANGLE_TOL:
                if  d1_ < MAX_DIST_LINE and d2_ < MAX_DIST_LINE and d1_ + d2_ < min_dist:

                    min_dist = d1_ + d2_
                    min_ind = ind_line
                    found_line = True

        if found_line:
            lines[min_ind][0].append([p1,p2])

    pickle.dump(lines, open( "lines.pkl", "wb" ))


GAP_THRES = 100
GAP_THRES_2 = 30

# #
for ind_line in tqdm.tqdm(range(len(lines))):

    line = lines[ind_line]
    line_segs = line[0]

    new_line_segs = []

    # sorted_line_segs = np.array(line_segs).tolist()
    #
    # sorted_line_segs.sort()
    #
    # last_point = sorted_line_segs[0][1]
    #
    # new_line_segs.append(sorted_line_segs[0])
    #
    # for ind_line_seg in range(1,len(sorted_line_segs)):
    #
    #     line_seg = sorted_line_segs[ind_line_seg]
    #
    #     if euclid_distance(line_seg[0], last_point) < GAP_THRES:
    #         new_line_segs.append([np.array(line_seg[0]),np.array(line_seg[1])])
    #         last_point = line_seg[1]
    #     else:
    #         break
    #
    # line_segs_2 = new_line_segs

    for ind_line_seg, line_seg in enumerate(line_segs):

        min_dist = float('inf')

        for ind_line_seg_2, line_seg_2 in enumerate(line_segs):

            if ind_line_seg_2 == ind_line_seg:
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
        if min_dist > GAP_THRES_2:
            continue

        new_line_segs.append(line_seg)

    lines[ind_line][0] = new_line_segs

# img_lines_before_merge = img.copy()
# for line in lines:
# #
#     for seg in line[0]:
#         x1,y1 = seg[0]
#         x2,y2 = seg[1]
#
#         cv2.line(img_lines_before_merge, (x1, y1), (x2, y2), colors[ind_color], 2)
#
#     ind_color = (ind_color + 1) % len(colors)
#
# cv2.imwrite('lines_before_merge.png', img_lines_before_merge)
#
# # merging line_segs that are too close
#
# groups = []
#
# non_empty_lines = 0
#
# for ind_line, line in enumerate(lines):
#
#     if not line[0]:
#         continue
#
#     non_empty_lines += 1
#
#     if DEBUG and ind_line >= 30:
#         img_debug = img.copy()
#         cv2.line(img_debug, (line[0][0][0][0], line[0][0][0][1]), (line[0][0][1][0], line[0][0][1][
#             1]), (0,
#                                                                                              255, 0),
#                  5)
#
#         curr_fig = plt.figure()
#         plt.imshow(img_debug)
#         plt.show()
#         plt.close(curr_fig)
#
#     # merge_group = find_merging_group(groups, ind_line, line, MERGE_THRES,ANGLE_TOL=10)
#
#     merge_group = -1
#
#     if merge_group == -1:
#         groups.append([line])
#     else:
#         groups[merge_group].append(line)
#
#
# line_lens = []

for ind_line, line in enumerate(lines):

    min_x = float('inf')
    max_x = 0

    min_point = np.array([-1,-1])
    max_point = np.array([-1,-1])

    for seg in line[0]:
            x1, y1 = seg[0]
            x2, y2 = seg[1]

            if x1 < min_x:
                min_point = np.array([x1,y1])
                min_x = x1
            if x2 > max_x:
                max_point = np.array([x2,y2])
                max_x = x2

    line_len = euclid_distance(min_point, max_point)
    if line_len == float('inf'):
        line_len = 0

    lines[ind_line][1] = line_len
    lines[ind_line][2] = [min_point,max_point]
    # line_lens.append(line_len)

img_lines = img.copy()
ladder_start_end = img.copy()
img_lines_2 = np.zeros_like(img)
img_lines_3 = np.zeros_like(img)



#find group of 3 lines that form a ladder (2 parallel + perpendicular connection)
n_lines = len(lines)
already_in_ladder = set()

ladders = []

for ind_line1 in range(n_lines):

    line1 = lines[ind_line1]

    if  line1[1] < MIN_LADDER_LENGTH:
        continue

    m1, _, angle1, p1_1, p2_1 = get_linear_coefs(line1[0][0])

    # for ind_line2 in range(ind_line1+1, n_lines):
    for ind_line2 in range(n_lines):
        if ind_line2 == ind_line1:
            continue

        if ind_line1 in already_in_ladder:
            break

        line2 = lines[ind_line2]

        if line2[1] < MIN_LADDER_LENGTH or ind_line2 in already_in_ladder:
            continue

        m2,_,angle2,p1_2, p2_2= get_linear_coefs(line2[0][0])

        if abs(angle1 - angle2) < ANGLE_TOL:

            if ind_line1 == 109 and ind_line2 == 113:

                if DEBUG:
                    img_debug = img.copy()
                    cv2.line(img_debug, (line1[0][0][0][0], line1[0][0][0][1]), (line1[0][0][1][0],
                                                                                 line1[0][0][1][1]),
                             (0,
                              255, 0),
                             5)
                    cv2.line(img_debug, (line2[0][0][0][0], line2[0][0][0][1]), (line2[0][0][1][0],
                                                                                 line2[0][0][1][1]),
                             (0,
                              255, 0),
                             5)

                    curr_fig = plt.figure()
                    plt.imshow(img_debug)
                    plt.show()
                    plt.close(curr_fig)

            # for ind_line3 in range(ind_line2+1, n_lines):
            for ind_line3 in range(n_lines):

                if ind_line3 == ind_line1 or ind_line3 == ind_line2:
                    continue

                line3 = lines[ind_line3]

                if not line3[0] or ind_line3 in already_in_ladder:
                    continue

                m3, _, angle3, p1_3, p2_3 = get_linear_coefs(line3[0][0])

                # if abs(m1*m3 + 1) <= EPS: #perpendicular lines

                #now need to check if perdicular line extremities are close to the ladder
                # handles
                d1_13 = abs(
                    np.cross(p2_1 - p1_1, p1_3 - p1_1) / np.linalg.norm(p2_1 - p1_1))
                d1_23 = abs(
                    np.cross(p2_1 - p1_1, p2_3 - p1_1) / np.linalg.norm(p2_1 - p1_1))

                d1_3 = min(d1_13, d1_23)

                d2_13 = abs(
                    np.cross(p2_2 - p1_2, p1_3 - p1_2) / np.linalg.norm(p2_2 - p1_2))
                d2_23 = abs(
                    np.cross(p2_2 - p1_2, p2_3 - p1_2) / np.linalg.norm(p2_2 - p1_2))

                d2_3 = min(d2_13, d2_23)

                if d1_3 <= INTERSECTION_THRES and d2_3 <= INTERSECTION_THRES:

                    ladders.append((ind_line1,ind_line2,ind_line3))
                    already_in_ladder.add(ind_line1)
                    already_in_ladder.add(ind_line2)
                    already_in_ladder.add(ind_line3)
                    break

img_ladders = img.copy()
img_ladders2 = np.zeros_like(img)

# def intersection(L1, L2):
#     D  = L1[0] * L2[1] - L1[1] * L2[0]
#     Dx = L1[2] * L2[1] - L1[1] * L2[2]
#     Dy = L1[0] * L2[2] - L1[2] * L2[0]
#     if D != 0:
#         x = Dx / D
#         y = Dy / D
#         return x,y
#     else:
#         return False

# def find_intersection(a1,b1,c1,a2,b2,c2):
#     determinant = a1*b2 - a2*b1
#     x = (c1*b2 - c2*b1) / determinant
#     y = (a1*c2 - a2*c1) / determinant
#
#     return (x,y)
#
# def find_reciprocals(m1,b1,m2,b2,c2, point):
#
#     perp_m = -1/m1
#     perp_b = point[1] - perp_m*point[0]
#
#     perp_c = -perp_m * point[0] - perp_b * point[1]
#
#     return find_intersection(perp_m, perp_b, perp_c, m2,b2,c2)
#
#
# for ladder in ladders:
#
#     line_ind1 = ladder[0]
#     line_ind2 = ladder[1]
#
#     line1 = lines[ind_line1]
#     line2 = lines[ind_line2]
#
#     m1, b1, _, _, _ = get_linear_coefs(line1[0][0])
#     m2, b2, _, _, _ = get_linear_coefs(line2[0][0])
#     point_1 = line1[0][0][0]
#     point_2 = line2[0][0][0]
#
#     left_recip_12 = find_reciprocals(m1,-1, m2, -1, -b2, line1[2][0])
#     right_recip_12 = find_reciprocals(m1, -1, m2, -1, -b2, line1[2][1])
#
#     left_recip_21 = find_reciprocals(m2, -1, m1, -1, -b1, line2[2][0])
#     right_recip_21 = find_reciprocals(m2, -1, m1, -1, -b1, line2[2][1])

for ladder in ladders:

    for ind, ind_line in enumerate(ladder):
        # for line in lines:
        line = lines[ind_line]

        for seg in line[0]:
            x1, y1 = seg[0]
            x2, y2 = seg[1]

            # if x1 < min_x:
            #     min_point = (x1,y1)
            #     min_x = x1
            # if x2 > max_x:
            #     max_point = (x2,y2)
            #     max_x = x2

            cv2.line(img_ladders, (x1, y1), (x2, y2), colors[ind_color], 3)
            cv2.line(img_ladders2, (x1, y1), (x2, y2), colors[ind_color], 3)

        if ind < 2:

            cv2.circle(img_ladders, tuple(line[2][0].tolist()), 20, colors[ind_color], 5)
            cv2.circle(img_ladders, tuple(line[2][1].tolist()), 20, colors[ind_color], 5)
            cv2.circle(img_ladders2, tuple(line[2][0].tolist()), 20, colors[ind_color], 5)
            cv2.circle(img_ladders2, tuple(line[2][1].tolist()), 20, colors[ind_color], 5)

    ind_color = (ind_color + 1) % len(colors)

cv2.imwrite('img_ladders.png', img_ladders)
cv2.imwrite('img_ladders2.png', img_ladders2)



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