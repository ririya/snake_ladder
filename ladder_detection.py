import pickle
import os
import math
import tqdm
import numpy as np
import cv2

from params import*


def find_intersection(a,c,b,d):
    #find intersection between lines ax + c and bx + d

    x = int(np.round((d - c) / (a - b)))

    y = int(np.round(a*x + c))

    return (x,y)

def find_reciprocals(m, b, perp_m, point):

    #calculate intersection of base line and perpendicular line
    perp_b = point[1] - perp_m*point[0]

    #calculate intersection of parallel line and perpendicular line
    return find_intersection(perp_m, perp_b, m,b)

def find_all_reciprocal_end_points(ladder, lines):

    line_ind1 = ladder[0]
    line_ind2 = ladder[1]
    line_ind3 = ladder[2]

    line1 = lines[line_ind1]
    line2 = lines[line_ind2]
    line3 = lines[line_ind3]

    m1, b1, _, _, _ = get_linear_coefs(line1[0][0])
    m2, b2, _, _, _ = get_linear_coefs(line2[0][0])
    m3, _, _, _, _ = get_linear_coefs(line3[0][0])

    #find reciprocal end points of line 1 on line 2
    left_recip_12 = find_reciprocals(m2, b2, m3, line1[2][0])
    right_recip_12 = find_reciprocals(m2, b2, m3, line1[2][1])

    #find reciprocals end points of line 2 on line 1
    left_recip_21 = find_reciprocals(m1, b1, m3, line2[2][0])
    right_recip_21 = find_reciprocals(m1, b1, m3, line2[2][1])


    return (left_recip_12, right_recip_12, left_recip_21, right_recip_21)

def get_angle(x1, x2, y1, y2):
    angle = math.atan2(y2 - y1, x2 - x1)
    angle = math.degrees(angle)
    return angle


def reorder_points(p1, p2):

    combined = [p1.tolist(), p2.tolist()]
    combined.sort()

    return np.array(combined[0]), np.array(combined[1])


def get_linear_coefs(line_seg):
    p1 = line_seg[0]
    p2 = line_seg[1]

    m = (p2[1] - p1[1]) / (p2[0] - p1[0])
    angle = get_angle(p1[0], p2[0], p1[1], p2[1])

    b = p1[1] - m * p1[0]

    return m, b, angle, p1, p2


def euclid_distance(p1, p2):
    d = np.linalg.norm(np.array(p1) - np.array(p2))

    if d == 0:  # disregard when it's the same point
        return float('inf')
    return d


def is_shadow(p1line, p2line, img, blur_gray):

    """
    Since shadows are not as dark as the real ladder lines, we search a square patch around the
    segment points for dark spots. The line detection sometimes does not land exactly on the
    ladder pixels, that's why a patch is needed.
    """

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


def find_lines(img, blur_gray, lines, line_segs, n_line_segs, min_line_length, generate_stems, desc):

    """
    Two modes of operation:
    generate_stems = True: The algorithm finds base line segments larger than a threshold and
    ignores other line segments in the same line. The threshold is used to save computational
    time and reduces noise. We assume the ladder main segments are relatively large compared to
    other line segments in the image and at least one will be greater than the threshold.

    generate_stems = False: The algorithm adds other line segments to the base lines. In this
    case we should consider a smaller line_length threshold.

    line segments are determined to be in the same line if their angles are similar and the
    distance from the two line segments is small.
    """

    for ind_line_seg in tqdm.tqdm(range(n_line_segs), desc=desc):

        line = line_segs[ind_line_seg]

        x1, y1, x2, y2 = line[0]

        line_length = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

        if line_length < min_line_length:
            continue

        p1 = np.array([x1, y1], dtype=int)
        p2 = np.array([x2, y2], dtype=int)

        #reorder points in sorted order. this ensures the angle calculation is always consistent
        p1, p2 = reorder_points(p1, p2)

        angle = get_angle(p1[0], p2[0], p1[1], p2[1])

        #ignore grid lines, parallel to the image axis
        if abs(abs(angle) - 90) < ANGLE_TOL or abs(abs(angle) - 0) < ANGLE_TOL or abs(abs( \
                angle) - 180) < ANGLE_TOL or abs(abs(angle) - 360) < ANGLE_TOL:
            continue

        # determines if it is a real ladder line segment or a shadow
        if is_shadow(p1, p2, img, blur_gray):
            continue

        found_line = False

        min_dist = float('inf')

        #search for the closest line_stem to this line segment
        for ind_line, line in enumerate(lines):

            p1line = line[0][0][0]
            p2line = line[0][0][1]

            angle2 = get_angle(p1line[0], p2line[0], p1line[1], p2line[1])

            # Calculate the distance between lines and points using a linear algebra formula
            d1_ = abs(np.cross(p2line - p1line, p1 - p1line) / np.linalg.norm(p2line - p1line))
            d2_ = abs(np.cross(p2line - p1line, p2 - p1line) / np.linalg.norm(p2line - p1line))

            angle_diff = abs(angle - angle2)

            if angle_diff < ANGLE_TOL:  #angles must be similar
                if d1_ < MAX_DIST_LINE and d2_ < MAX_DIST_LINE and d1_ + d2_ < min_dist:
                    min_dist = d1_ + d2_
                    min_ind = ind_line
                    found_line = True

                    if generate_stems:
                        break

        if found_line:
            if not generate_stems:
                lines[min_ind][0].append([p1, p2])
        else:
            if generate_stems:
                lines.append([[[p1, p2]], [x1, x2], [y1, y2], [], []])

            # if DEBUG:
            #     curr_ind = len(lines) - 1
            #     img_debug = img.copy()
            #     cv2.line(img_debug, (p1[0], p1[1]), (p2[0], p2[1]), (0,255,0), 5)
            #     # curr_fig = plt.figure()
            #     # plt.imshow(img_debug)
            #     # plt.show()
            #     # plt.close(curr_fig)
            #     cv2.imwrite(str.format("line_stems/{}.png", curr_ind), img_debug)


def save_lines(lines, img, name, colors, results_dir):
    ind_color = 0

    img_lines = img.copy()

    for line in lines:

        for point in line[0]:
            x1, y1 = point[0]
            x2, y2 = point[1]

            cv2.line(img_lines, (x1, y1), (x2, y2), colors[ind_color], 2)

        ind_color = (ind_color + 1) % len(colors)

    cv2.imwrite(os.path.join(results_dir, name + '.png'), img_lines)

    pickle.dump(lines, open(os.path.join(results_dir, name + ".pkl"), "wb"))


def detect_lines(img, blur_gray, results_dir):
    """
    Detect lines using the Line Segment Detection algorithm
    """
    detector = cv2.createLineSegmentDetector()

    line_segs = detector.detect(blur_gray)

    line_image = np.zeros_like(img)

    for line_seg in line_segs[0]:
        x1, y1, x2, y2 = line_seg[0]
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 2)

    cv2.imwrite(os.path.join(results_dir, 'line_segs.png'), line_image)

    return line_segs


def eliminate_outliers(lines, img):

    """
    Look for line segments that are too far from the rest any other line segment in the line and
    remove them
    """

    for ind_line in tqdm.tqdm(range(len(lines)),desc="Outlier Removal"):

        line = lines[ind_line]
        line_segs = line[0]

        new_line_segs = []

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

            if min_dist > GAP_THRES_2:
                continue

            new_line_segs.append(line_seg)

        lines[ind_line][0] = new_line_segs


def get_line_lens(lines, img):

    #gets the total line length for each line, determining start and end points, from left to right

    for ind_line, line in enumerate(lines):

        min_x = float('inf')
        max_x = 0

        min_point = np.array([-1, -1])
        max_point = np.array([-1, -1])

        for seg in line[0]:
            x1, y1 = seg[0]
            x2, y2 = seg[1]

            if x1 < min_x:
                min_point = np.array([x1, y1])
                min_x = x1
            if x2 > max_x:
                max_point = np.array([x2, y2])
                max_x = x2

        line_len = euclid_distance(min_point, max_point)
        if line_len == float('inf'):
            line_len = 0

        lines[ind_line][1] = line_len
        lines[ind_line][2] = [min_point, max_point]


def find_ladders(lines, img):
    # find group of 3 lines that form a ladder (2 parallel + perpendicular connection in H shape)
    n_lines = len(lines)
    already_in_ladder = set()

    ladders = []

    print("Finding Ladder groups...")

    for ind_line1 in range(n_lines):

        line1 = lines[ind_line1]

        if line1[1] < MIN_LADDER_LENGTH:
            continue

        m1, _, angle1, p1_1, p2_1 = get_linear_coefs(line1[0][0])

        for ind_line2 in range(n_lines):
            if ind_line2 == ind_line1:
                continue

            if ind_line1 in already_in_ladder:
                break

            line2 = lines[ind_line2]

            if line2[1] < MIN_LADDER_LENGTH or ind_line2 in already_in_ladder:
                continue

            m2, _, angle2, p1_2, p2_2 = get_linear_coefs(line2[0][0])

            if abs(angle1 - angle2) < ANGLE_TOL:

                for ind_line3 in range(n_lines):

                    if ind_line3 == ind_line1 or ind_line3 == ind_line2:
                        continue

                    line3 = lines[ind_line3]

                    if not line3[0] or ind_line3 in already_in_ladder:
                        continue

                    m3, _, angle3, p1_3, p2_3 = get_linear_coefs(line3[0][0])

                    # now need to check if perpendicular line extremities are close to the ladder
                    # handles. In some cases the lines are not perpendicular (smallest ladder) so
                    # we do not enforce lines to be perpendicular

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
                        ladders.append((ind_line1, ind_line2, ind_line3))
                        already_in_ladder.add(ind_line1)
                        already_in_ladder.add(ind_line2)
                        already_in_ladder.add(ind_line3)
                        break

    return ladders


def plot_ladders(ladders, lines, img, colors, results_dir):

    """
    plot ladder points and find extreme points. This algorithm makes no distinction between start
    and end of the ladder
    """

    img_ladders = img.copy()
    img_ladders2 = np.zeros_like(img)

    ind_color = 0

    ladder_end_points = []

    for ladder in ladders:

        curr_ladder_end_points = []

        """
        If there are mistakes in the line detection, we correct them by finding the reciprocal end 
        points in the parallel line. We choose the point closest to the center of the ladder as the base and 
        substitute the point in the other line with its reciprocal
        """
        left_recip_12, right_recip_12, left_recip_21, right_recip_21 = \
            find_all_reciprocal_end_points(ladder, lines)

        line1 = lines[ladder[0]]
        line2 = lines[ladder[1]]

        line1_left = line1[2][0]
        line2_left = line2[2][0]

        line1_right = line1[2][1]
        line2_right = line2[2][1]

        if line1_left[0] < line2_left[0]:
            line1[2][0] = np.array(left_recip_21)
        else:
            line2[2][0] = np.array(left_recip_12)

        if line1_right[0] > line2_right[0]:
            line1[2][1] = np.array(right_recip_21)
        else:
            line2[2][1] = np.array(right_recip_12)

        for ind, ind_line in enumerate(ladder):

            line = lines[ind_line]

            for seg in line[0]:
                x1, y1 = seg[0]
                x2, y2 = seg[1]

                cv2.line(img_ladders, (x1, y1), (x2, y2), colors[ind_color], 3)
                cv2.line(img_ladders2, (x1, y1), (x2, y2), colors[ind_color], 3)

            if ind < 2:

                cv2.circle(img_ladders, tuple(line[2][0].tolist()), 20, colors[ind_color], 5)
                cv2.circle(img_ladders, tuple(line[2][1].tolist()), 20, colors[ind_color], 5)
                cv2.circle(img_ladders2, tuple(line[2][0].tolist()), 20, colors[ind_color], 5)
                cv2.circle(img_ladders2, tuple(line[2][1].tolist()), 20, colors[ind_color], 5)

                curr_ladder_end_points.append(line[2][0])
                curr_ladder_end_points.append(line[2][0])

        ladder_end_points.append(curr_ladder_end_points)

        ind_color = (ind_color + 1) % len(colors)

    plt.figure()
    plt.imshow(img_ladders)
    plt.show()

    cv2.imwrite(os.path.join(results_dir, 'ladders.png'), img_ladders)
    cv2.imwrite(os.path.join(results_dir, 'ladders_no_bg.png'), img_ladders2)

    return ladder_end_points, img_ladders


def detect_ladders(img, blur_gray, colors, results_dir):

    """
    Detect ladders based on lines that form a H shape
    """

    #Detect all line segments in the image
    line_segs = detect_lines(img, blur_gray, results_dir)

    #Caching for fast run. FORCE_CALCULATE_LINES ignores the cached results
    lines_path = os.path.join(results_dir, "lines.pkl")
    if os.path.exists(lines_path) and not FORCE_CALCULATE_LINES:
        lines = pickle.load(open(lines_path, "rb"))

    else:
        lines = []
        line_segs = line_segs[0].tolist()
        line_segs.sort()
        n_line_segs = len(line_segs)

        # Detect the base line segments aka stems for the stairs
        find_lines(img, blur_gray, lines, line_segs, n_line_segs,
                        min_line_length=MIN_LINE_LENGTH,
                        generate_stems=True, desc = "Line Stem Detection")

        save_lines(lines, img, "stems", colors, results_dir)

        # Add other segments that lie on the same lines
        find_lines(img, blur_gray, lines, line_segs, n_line_segs, min_line_length=0,
                        generate_stems=False, desc = "Line Detection")

        save_lines(lines, img, "lines", colors, results_dir)

    #eliminate isolated line segments (outliers)
    eliminate_outliers(lines, img)

    #get line start and end points
    get_line_lens(lines, img)

    #find ladders by combining groups of lines with a H shape
    ladders = find_ladders(lines, img)

    #find ladder extreme points
    ladder_end_points, img_ladder_results = plot_ladders(ladders, lines, img, colors, results_dir)

    return ladders, ladder_end_points, lines, img_ladder_results