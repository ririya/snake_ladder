import os
import time
import math

import cv2
import tqdm

from ladder_detection import get_angle,reorder_points,detect_lines, detect_ladders, euclid_distance

from params import*

def find_grid_lines(img, blur_gray, lines, min_line_length):
        
        """
        Similar to the algorithm used on ladder detection, but this time we look for the grid 
        lines, which are parallel to the image axes
        """
        
        line_segs = detect_lines(img, blur_gray, results_dir)
        line_segs = line_segs[0]
        n_line_segs = len(line_segs)

        for ind_line_seg in tqdm.tqdm(range(n_line_segs), desc="Grid Line Detection"):

            line = line_segs[ind_line_seg]

            x1, y1, x2, y2 = line[0]

            line_length = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

            if line_length < min_line_length:
                continue

            p1 = np.array([x1, y1], dtype=int)
            p2 = np.array([x2, y2], dtype=int)

            p1, p2 = reorder_points(p1, p2)

            angle = get_angle(p1[0], p2[0], p1[1], p2[1])

            if abs(abs(angle) - 90) < ANGLE_TOL or abs(abs(angle) - 0) < ANGLE_TOL or abs(abs( \
                    angle) - 180) < ANGLE_TOL or abs(abs(angle) - 360) < ANGLE_TOL:

                found_line = False

                min_dist = float('inf')

                for ind_line, line in enumerate(lines):

                    p1line = line[0][0][0]
                    p2line = line[0][0][1]

                    angle2 = get_angle(p1line[0], p2line[0], p1line[1], p2line[1])

                    d1_ = abs(np.cross(p2line - p1line, p1 - p1line) / np.linalg.norm(p2line - p1line))
                    d2_ = abs(np.cross(p2line - p1line, p2 - p1line) / np.linalg.norm(p2line - p1line))

                    angle_diff = abs(angle - angle2)

                    if angle_diff < ANGLE_TOL:
                        if d1_ < MAX_DIST_LINE and d2_ < MAX_DIST_LINE and d1_ + d2_ < min_dist:
                            min_dist = d1_ + d2_
                            min_ind = ind_line
                            found_line = True

                if found_line:
                        lines[min_ind][0].append([p1, p2])
                else:
                        lines.append([[[p1, p2]], [x1, x2], [y1, y2], [], []])

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
    # pixelSequence = pixelSequence**2
    allPixels = np.append(allPixels, pixelSequence, axis=0)
    return imNp,pixelSequence,allPixels

def check_color_spaces(img):
    allPixels = np.empty((0, 3))
    allPixelsLab = np.empty((0, 3))
    allPixelsHSV = np.empty((0, 3))

    imHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    imLab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    #
    imNp, pixelSequence, allPixels = getPixelSequence(img, allPixels, 'RGB')
    imNpLab, pixelSequenceLab, allPixelsLab = getPixelSequence(imLab, allPixelsLab, 'Lab')
    imNpHSV, pixelSequenceHSV, allPixelsHSV = getPixelSequence(imHSV, allPixelsHSV, 'HSV')

    print('Computing clustering RGB')
    # kmeansRGB = clusteringFunction(n_clusters=N_CLUSTERS, threshold=0.2).fit(allPixels)
    kmeansRGB = clusteringFunction(n_clusters=N_CLUSTERS).fit(allPixels)
    print('Computing clustering HSV')
    # kmeansHSV = clusteringFunction(n_clusters=N_CLUSTERS, threshold=0.2).fit(allPixelsHSV)
    kmeansHSV = clusteringFunction(n_clusters=N_CLUSTERS).fit(allPixelsHSV)

    print('Computing clustering Lab')
    # kmeansLab = clusteringFunction(n_clusters=N_CLUSTERS, threshold=0.2).fit(allPixelsLab)
    kmeansLab = clusteringFunction(n_clusters=N_CLUSTERS).fit(allPixelsLab)
    #
    Images = [imNp, imNpHSV, imNpLab]
    pixelsFormats = [allPixels, allPixelsHSV, allPixelsLab]
    colorFormats = ['RGB', 'HSV', 'LAB']
    kmeans = [kmeansLab, kmeansHSV, kmeansLab]

    subPlotInd = 1

    plt.figure(1)

    for indFormat in range(len(colorFormats)):

        colorFormat = colorFormats[indFormat]

        startKmeans = time.time()
        labels = kmeans[indFormat].predict(pixelsFormats[indFormat])
        endKmeans = time.time()
        print('Elapsed Time Kmeans: ' + str(endKmeans - startKmeans) + ' seconds')

        labels = labels.reshape((img.shape[0], img.shape[1]))

        titles = [colorFormat, 'Segmented Img']
        images = [Images[indFormat], labels]

        plt.figure(1)
        for j in range(len(titles)):
            plt.subplot(len(colorFormats), 2, subPlotInd)
            implot = plt.imshow(images[j])
            plt.axis('off')
            plt.title(titles[j])
            subPlotInd += 1

        plt.figure(2)
        plt.imshow(labels, cmap=newcmp)
        plt.savefig(os.path.join(results_dir, str.format("{}.png", colorFormat)))
        plt.close(2)

        plt.figure(1)
        plt.show()
        plt.close(1)
        pass

def segment_image(img, blur_gray, ladders, lines):
    
    """
    Segments the image by applying k-means and finding the label with smallest norm (darkest)
    Delete ladder and grid lines 
    """
    
    #Caching for fast run. FORCE_SEGMENT_IMAGE = True ignores the cached results.
    seg_img_path = os.path.join(results_dir, "seg_img.png")

    if not os.path.exists(seg_img_path) or FORCE_SEGMENT_IMAGE:

        print("Segmenting Images...")
        allPixels = np.empty((0, 3))

        imNp, pixelSequence, allPixels = getPixelSequence(img, allPixels, 'RGB')

        kmeansRGB = clusteringFunction(n_clusters=N_CLUSTERS).fit(allPixels)

        labels = kmeansRGB.labels_

        labels = labels.reshape((img.shape[0], img.shape[1]))
        
        cluster_norms = [np.linalg.norm(cluster) for cluster in kmeansRGB.cluster_centers_]
        ind_black = np.argmin(np.array(cluster_norms))  #label with smallest norm (darkest)

        seg_img = np.zeros_like(labels)
        seg_img = seg_img.astype(np.uint8)
        seg_img[labels == ind_black] = 255

        seg_img = np.stack((seg_img, seg_img, seg_img), axis=-1)  #converts to 3 channel to use 
        # cv2.line()

        if DEBUG:
            plt.figure(2)
            plt.imshow(seg_img)
            plt.show()
            plt.close(2)

        grid_img = np.zeros_like(seg_img)

        # delete ladder lines
        for ladder in ladders:

            for ind, ind_line in enumerate(ladder):

                line = lines[ind_line]

                for seg in line[0]:
                    x1, y1 = seg[0]
                    x2, y2 = seg[1]

                    cv2.line(seg_img, (x1, y1), (x2, y2), (0, 0, 0), 3)

        lines = []

        # find grid lines and delete them
        find_grid_lines(img, blur_gray, lines, min_line_length=0)

        for line in lines:

            for seg in line[0]:
                x1, y1 = seg[0]
                x2, y2 = seg[1]

                cv2.line(seg_img, (x1, y1), (x2, y2), (0, 0, 0), 3)
                cv2.line(grid_img, (x1, y1), (x2, y2), (255, 255, 255), 3)

        seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(seg_img_path, seg_img)

    else:
        seg_img = cv2.imread(seg_img_path)
        seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)

    if DEBUG:
        plt.figure(2)
        plt.imshow(seg_img)
        plt.show()
        plt.close(2)

    return seg_img

def find_snakes(seg_img):

    """
    
    Find snake connected components. First, delete extra elements from the scene (numbers, 
    line and ladder leftovers). The extra elements will become small after an "open" operation
    only connected components left should be the snakes. Snake conn components are assumed to be 
    relatively large.
    """

    print("Finding Snakes...")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    open = cv2.morphologyEx(seg_img, cv2.MORPH_OPEN, kernel, iterations=1)

    if DEBUG:
        plt.figure(4)
        plt.imshow(open)
        plt.show()
        plt.close(4)

    contours = cv2.findContours(seg_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours[1], key=cv2.contourArea, reverse=True)

    filtered_contours = np.zeros(seg_img.shape, dtype=np.uint8)

    for cnt in contours:

        conn_comp_full = np.zeros(seg_img.shape, dtype=np.uint8)
        cv2.drawContours(conn_comp_full, [cnt], 0, 1, -1)

        conn_comp = conn_comp_full * open

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))

        opened_comp = cv2.morphologyEx(conn_comp, cv2.MORPH_OPEN, kernel, iterations=1)

        opened_comp_area = np.sum(opened_comp[:] / 255)

        if DEBUG:
            plt.figure(3)
            plt.imshow(opened_comp)
            plt.show()
            plt.close(3)

        if opened_comp_area > CONTOUR_AREA_THRES_ARTIFACTS:  # filter out small elements
            filtered_contours |= conn_comp

    if DEBUG:
        plt.figure(3)
        plt.imshow(filtered_contours)
        plt.show()
        plt.close(3)

    # perform closing to close holes created when deleting the lines and ladders
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))

    closing = cv2.morphologyEx(filtered_contours, cv2.MORPH_CLOSE, kernel, iterations=1)

    if DEBUG:
        plt.figure(4)
        plt.imshow(closing)
        plt.show()
        plt.close(4)

    # find new connected components after closing
    contours = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours[1], key=cv2.contourArea, reverse=True)

    all_contours = np.zeros(seg_img.shape, dtype=np.uint8)

    #finally find the snakes
    snakes = []

    for cnt in contours:

        conn_comp_full = np.zeros(seg_img.shape, dtype=np.uint8)
        cv2.drawContours(conn_comp_full, [cnt], 0, 1, -1)

        conn_comp = conn_comp_full * closing  #this multiplication ensures we are not using any 
        # holes filled by draw contours

        area = np.sum(conn_comp[:] / 255)
        if area > CONTOUR_AREA_THRES:
            snakes.append(conn_comp)

        if DEBUG:
            plt.figure(3)
            plt.imshow(conn_comp)
            plt.show()
            plt.close(3)

    if DEBUG:
        plt.figure(3)
        plt.imshow(all_contours)
        plt.show()
        plt.close(3)

    return snakes, closing

def find_snake_heads(seg_img, img, snakes,closing):
    
    """
    Find snake heads by 
    1)identifying a marker for the head
    2)finding the body
    3)the head will be the greatest connect component from subtracting the snake and body 
    
    We are returning a segmented image of the head and also the centroid
    """

    print("Finding Snake Heads...")
 
    
    snake_heads =[]
    head_centers = []

    all_snake_heads = np.zeros(seg_img.shape, dtype=np.uint8)

   
    for comp in snakes:

        # for each snake perform an increasing open operation until the image vanishes, last part 
        # to disappear will be a marker for the head
        
        i = 1
        
        while True:

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            opening = cv2.morphologyEx(comp, cv2.MORPH_OPEN, kernel, iterations= i)
            if np.sum(opening[:]) == 0:
                break
            marker = opening
            i += 1

            if DEBUG:
                plt.figure(100)
                plt.imshow(opening)
                plt.show()
                plt.close(100)

        #find marker complement; marker divides the snake into 2 halves one containing the body and
        #one containing the remaining of the head

        marker_complement = comp.copy()
        marker_complement[marker == 255] = 0

        #the snake's body will be the biggest connected component of the complement
        contours = cv2.findContours(marker_complement, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours[1], key=cv2.contourArea, reverse=True)

        snake_body = np.zeros_like(marker_complement)

        cv2.drawContours(snake_body, [contours[0]], 0, 255, -1)

        snake_head = comp.copy()
        snake_head[snake_body == 255] = 0

        if DEBUG:
            plt.figure(1111)
            plt.imshow(marker)
            plt.show()
            plt.close(1111)

            plt.figure(100)
            plt.imshow(snake_head)
            plt.show()
            plt.close(100)

        contours = cv2.findContours(snake_head, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours[1], key=cv2.contourArea, reverse=True)

        #if there are extra connected components, ignore them and choose the biggest as the head
        snake_head2 = np.zeros_like(snake_head)

        cv2.drawContours(snake_head2, [contours[0]], 0, 255, -1)

        snake_head2 = snake_head2*closing #multiplication to ignore holes filled by drawContours

        if DEBUG:
            plt.figure(100)
            plt.imshow(snake_head2)
            plt.show()
            plt.close(100)

        # calculate head centroid by using Hu moments
        M = cv2.moments(snake_head2)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        head_centers.append(np.array([cX,cY]))

        if DEBUG:
            plt.figure(99)
            plt.imshow(snake_head2)
            plt.show()
            plt.close(99)

        all_snake_heads |= snake_head2

        snake_heads.append(snake_head2)

    all_snake_heads_2 = img.copy()
    all_snake_heads_2[all_snake_heads == 1] = [0, 255, 0]

    # if DEBUG:
        # plt.figure(4)
        # plt.imshow(all_snake_heads_2)
        # plt.show()
        # plt.close(4)

    cv2.imwrite(os.path.join(results_dir, "snake_heads.png"), all_snake_heads_2)

    return all_snake_heads, snake_heads,head_centers

def find_snake_tails(snakes, head_centers):

    """
    The snake tail is the farthest snake's pixel from the head centroid.
    """

    snake_tails = []

    for ind_snake in tqdm.tqdm(range(len(snakes)), desc="Snake Tail Detection"):
        snake = snakes[ind_snake]

        head_center = head_centers[ind_snake]

        tail_end = np.array([-1, -1])

        max_dist = 0

        for i in range(snake.shape[0]):
            for j in range(snake.shape[1]):

                if i == head_center[1] and j == head_center[0]:
                    continue

                if snake[i][j] != 0:

                    dist = euclid_distance(np.array([j, i]), head_center)

                    if dist > max_dist:
                        max_dist = dist
                        tail_end = np.array([j, i])

        snake_tails.append(tail_end)

    return snake_tails



def detect_snakes(img,blur_gray, img_results, ladders = [], lines = []):
    
    #start by segmenting the image and generating a binary image
    seg_img = segment_image(img,blur_gray,ladders,lines)
    
    #find snake connected components
    snakes,closing = find_snakes(seg_img)
    
    #find snake heads (segmentation and centroids)
    all_snake_heads, snake_heads, head_centers = find_snake_heads(seg_img,img,snakes, closing)
    
    #find snake end points (single pixel)
    snake_tails = find_snake_tails(snakes, head_centers)

    #Plot results
    img_snake_results = img.copy()

    for ind_snake in range(len(snakes)):
        cv2.circle(img_snake_results, tuple(snake_tails[ind_snake].tolist()), 20, (0, 255, 0), 5)
        cv2.circle(img_snake_results, tuple(head_centers[ind_snake].tolist()), 20, (255, 0, 0), 5)
        cv2.circle(img_results, tuple(snake_tails[ind_snake].tolist()), 20, (0, 255, 0), 5)
        cv2.circle(img_results, tuple(head_centers[ind_snake].tolist()), 20, (255, 0, 0), 5)

    plt.figure()
    plt.imshow(img_snake_results)
    plt.show()
    cv2.imwrite(os.path.join(results_dir, "snake_head_and_tails.png"), img_snake_results)

    plt.figure()
    plt.imshow(img_results)
    plt.show()
    cv2.imwrite(os.path.join(results_dir, "detected_ladders_and_snakes.png"), img_results)


    return snake_heads,head_centers, snake_tails









