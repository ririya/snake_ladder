# snake_ladder

## Intro
 
 This code detects snakes and ladders in a board image. Ladders are defined by their end point pixels with no distinction between start and end. 
 The snake head is defined by it's centroid (however we also compute the head segmentation) and the snake tail is defined by the extreme pixel, opposing the head.
 
 This is a pure OpenCV python code, using carefully designed algorithms. Supervised machine learning was avoided due to a lack of a dataset. 
 State of the art object detection neural networks would definitely outperform the developed algorithms and any other supervised machine learning algorithm. 
 
  

## Ladder Detection

### Line detection
This algorithm relies on LSD (Line Segment Detection) combined with linear algebra principles. First, we detect base line segments (stems) for the stairs, 
then we add other line segments that lie on the same line. Line segments are considered to be in the same line if they have similar angles and are close to each other. 

### Line Grouping
Stairs are detected by combining 3 lines that form a H shape (2 parallel lines + 1 perpendicular). However in the given board one stair does not have perpendicular lines, so we do not enforce this constraint. Instead we look for lines that connect the parallel lines, where the end points are close enough to one of the parallel lines. 

### End Point Correction
If there are mistakes in the line detection, we correct the end points by finding the reciprocal end 
points in the parallel line. We choose the point closest to the center of the ladder as the base and 
substitute the point in the other line with its reciprocal.


### Limitations

This algorithm works for the given image as there are no stairs parallel to the grid. The code would also fail in the case of colinear stairs. In this case, we would need some additional contraints on the size of the lines and distance between them, and design a more robust outlier algorithm for detecting the colinear stair case. We would also need to handle the special case of vertical lines, as some parts of the code assume x1 < x2. 

For simplicity, there's no distinction between ladder start and end. We could find this difference by comparing the size of the last "H" in both ends. Once again we use linear algebra to detect the intersection points of the last H and find their sizes. In this case we would need to properly identify all the perpendicular lines in the stair. This could be achieved by reapplying the LSD algorithm on bounding boxes enclosing the detected stairs. Since there are potentially less artifact elements in the bounding box we could relax on the algorithm constraints. 


## Snake Detection 

This algorithm makes use of the fact the snakes are solid, large, dark objects in the image. We start by segmenting the image using K-means and looking for the label with darkest color. We explored different color spaces like HSV and Lab for this task and determined RGB was the best choice. Next, we delete the stair lines and the grid lines to make snake detetion easier. We also make use of opening and closing operations to get rid of remaining objects. 

### Snake Head
The snake head is detected by finding a marker for the head, by applying increasing degrees of a morphological opening operation. Because of the shape of the head, it will be the last element to vanish with the increasing levels. Once the marker is found, it typically divides the snake object into two halves, one containing the body and one containing the rest of the head. Manipulating these 3 components, we can restore the head and separate it from the body. This way we obtain a connected component for the head (segmentation) and the head's centroid.

### Snake Tail
The snake tail is defined as a single pixel, located on the extreme opposing the head. We could also define a segmentation by adding the pixels within a radial distance from the tail pixel. The radial distance could be a proportion of the snake length, which can be obtained by calculating the distance between head and tail. If the radial shape is not the best choice, we could also truncate it with a line connecting the points where the circle intersects the snake's edges.

### Limitations
The algorithm works for the given image, as the snakes are solid, large and dark. The algorithm is also sensitive to the extra objects found in the image (ladders, numbers, grid) and might fail if the extra objects are not properly removed. 

PS: Due to the random nature of K-means, there is a small chance the labels will not be segmented correctly. If that happens, just rerun the algorithm.

## Possible Improvements

The algorithms were not developed seeking optimal computational performance, so parallelization and efficient data structures could further optimize the code. 
 
The algorithms could also be further improved by fine-tuning all global parameters. 
 
Another possible improvement is first identifying and removing the numbers in the image as it tends to have high accuracy. However, for now we tried to avoid any supervised machine learning. 
 
## Dependencies

The conda virtual env folder **venv** is included in the source. The full list of dependencies are included in the source in the file **dependencies.txt**. The main dependencies are:

- Python 3.7.7
- OpenCV 3.4.2
- Numpy 1.19
- Scikit-learn 0.22.1
- Matplotlib 3.2.2
- Tqdm 4.51.0
