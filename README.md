
# Self-Driving Car Engineer Nanodegree


## Project: **Finding Lane Lines on the Road** 

## Import Packages


```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
%matplotlib inline
```

## Helper Functions


```python
import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)
```

## Building a pipeline

Here I'll build a simple line detection and drawing pipeline using the helper functions provided and apply it to videos.


```python
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def find_lines(image):
    # convert image to grayscale
    gray = grayscale(image)
    # smooth edges using gaussian blur
    blur = gaussian_blur(gray, kernel_size=5)
    # use canny to detect edges
    edges = canny(blur, low_threshold=50, high_threshold=150)
    # mask lane area
    imshape = edges.shape
    vertices = np.int32([[
        (int(imshape[1] * 0.05), int(imshape[0])),
        (int(imshape[1] * 0.40), int(imshape[0] * 0.6)),
        (int(imshape[1] * 0.60), int(imshape[0] * 0.6)),
        (int(imshape[1] * 0.95), int(imshape[0])),
    ]])
    masked = region_of_interest(edges, vertices=vertices)
    # use Hough to detect and lines
    return cv2.HoughLinesP(masked,
                           rho=1,
                           theta=np.pi / 180,
                           threshold=25,
                           lines=np.array([]),
                           minLineLength=25,
                           maxLineGap=250)

    
def process_image(image):
    lines = find_lines(image)
    # draw lines
    line_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    # annotate original image
    result = weighted_img(image, line_img, α=0.8, β=1., λ=0.)
    return result

if not os.path.exists('test_videos_output'):
    os.mkdir('test_videos_output')
white_output = 'test_videos_output/swr_1.mp4'
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
%time white_clip.write_videofile(white_output, audio=False)

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))
```

    [MoviePy] >>>> Building video test_videos_output/swr_1.mp4
    [MoviePy] Writing video test_videos_output/swr_1.mp4


    100%|█████████▉| 221/222 [00:03<00:00, 65.51it/s]


    [MoviePy] Done.
    [MoviePy] >>>> Video ready: test_videos_output/swr_1.mp4 
    
    CPU times: user 1.54 s, sys: 175 ms, total: 1.71 s
    Wall time: 3.62 s






<video width="960" height="540" controls>
  <source src="test_videos_output/swr_1.mp4">
</video>




## Drawing two lines

Instead of drawing all lines found, including false positives, let's average these to draw to lines, one for each lane.

Let's hope the number of correctly identified lines is much larger than false positives, reducing their effect on the final results


```python
def coefficients(line):
    x1, y1, x2, y2 = line
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    l2 = (x2 - x1) ** 2 + (y2 - y1) ** 2
    return m, b, l2

def process_image(image):
    # find average lines for lanes
    LEFT = 0
    RIGHT = 1
    lines = [[], []]
    for line in find_lines(image):
        m, b, l2 = coefficients(line[0])
        side = LEFT if m < 0 else RIGHT
        lines[side].append((m, b, l2))
    slopes = [
        np.mean([l[0] for l in lines[LEFT]]),
        np.mean([l[0] for l in lines[RIGHT]]),
    ]
    intercepts = [
        np.mean([l[1] for l in lines[LEFT]]),
        np.mean([l[1] for l in lines[RIGHT]]),
    ]

    # draw lines found
    line_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for slope, intercept in zip(slopes, intercepts):
        y1, y2 = image.shape[0], image.shape[0] * 0.6
        pt1 = (int((y1 - intercept) / slope), int(y1))
        pt2 = (int((y2 - intercept) / slope), int(y2))
        cv2.line(line_img, pt1, pt2, color=[255, 0, 0], thickness=3)

    # annotate original image
    result = weighted_img(image, line_img, α=0.8, β=1., λ=0.)
    return result

white_output = 'test_videos_output/swr_2.mp4'
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
%time white_clip.write_videofile(white_output, audio=False)

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))
```

    [MoviePy] >>>> Building video test_videos_output/swr_2.mp4
    [MoviePy] Writing video test_videos_output/swr_2.mp4


    100%|█████████▉| 221/222 [00:03<00:00, 69.43it/s]


    [MoviePy] Done.
    [MoviePy] >>>> Video ready: test_videos_output/swr_2.mp4 
    
    CPU times: user 1.6 s, sys: 182 ms, total: 1.78 s
    Wall time: 3.71 s






<video width="960" height="540" controls>
  <source src="test_videos_output/swr_2.mp4">
</video>




# Time to try the yellow lane video

The video shows the lanes slightly jittery when false positives influence gets high, but let's check the other video before further improving this.


```python
yellow_output = 'test_videos_output/syl_1.mp4'
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
%time yellow_clip.write_videofile(yellow_output, audio=False)

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_output))
```

    [MoviePy] >>>> Building video test_videos_output/syl_1.mp4
    [MoviePy] Writing video test_videos_output/syl_1.mp4


    100%|█████████▉| 681/682 [00:10<00:00, 63.83it/s]


    [MoviePy] Done.
    [MoviePy] >>>> Video ready: test_videos_output/syl_1.mp4 
    
    CPU times: user 5.07 s, sys: 545 ms, total: 5.62 s
    Wall time: 11.2 s






<video width="960" height="540" controls>
  <source src="test_videos_output/syl_1.mp4">
</video>




# Filtering outliers

The results above aren't that good, as the average slope and intercept for the lanes are being heavily skewed by false positives.

The obvious next step should be filtering out outliers that are messing up the averages.

To do that I'll first remove outliers based on the median slope. Then I'll calculate a weighted average, using the lines squared length as weights, so longer lines have a much higher impact on the results.


```python
def draw_filtered_lines(image, lines):
    # find average lines for lanes
    LEFT = 0
    RIGHT = 1
    separated = [[], []]
    for line in lines:
        m, b, l2 = coefficients(line[0])
        side = LEFT if m < 0 else RIGHT
        separated[side].append((m, b, l2))
    
    # Filter outliers based on slope
    filtered = []
    for selected in separated:
        slopes = [l[0] for l in selected]
        median_slope = np.median(slopes)
        limit = np.std(slopes) * 1.5
        selected = [s for s in selected if abs(s[0] - median_slope) < limit]
        filtered.append(selected)
    
    # Use the squared length of the lines for a weighted average
    average = []
    for selected in filtered:
        weights = [s[2] for s in selected]
        weight_sum = np.sum(weights)
        if weight_sum == 0:
            average.append((None, None, None))
            continue
        average.append(np.dot(weights, selected) / weight_sum)

    # draw lines found
    line_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for slope, intercept, _ in average:
        if slope is None:
            continue
        y1, y2 = image.shape[0], image.shape[0] * 0.6
        pt1 = (int((y1 - intercept) / slope), int(y1))
        pt2 = (int((y2 - intercept) / slope), int(y2))
        cv2.line(line_img, pt1, pt2, color=[255, 0, 0], thickness=3)

    # annotate original image
    result = weighted_img(image, line_img, α=0.8, β=1., λ=0.)
    return result

def process_image(image):
    lines = find_lines(image)
    return draw_filtered_lines(image, lines)

yellow_output = 'test_videos_output/syl_2.mp4'
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
%time yellow_clip.write_videofile(yellow_output, audio=False)

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_output))
```

    [MoviePy] >>>> Building video test_videos_output/syl_2.mp4
    [MoviePy] Writing video test_videos_output/syl_2.mp4


    100%|█████████▉| 681/682 [00:10<00:00, 64.71it/s]


    [MoviePy] Done.
    [MoviePy] >>>> Video ready: test_videos_output/syl_2.mp4 
    
    CPU times: user 5.27 s, sys: 545 ms, total: 5.82 s
    Wall time: 11 s






<video width="960" height="540" controls>
  <source src="test_videos_output/syl_2.mp4">
</video>




## Optional Challenge

I'm pretty happy with the results above, the pipeline works great here.

The last step is checking the challenge video.


```python
challenge_output = 'test_videos_output/challenge_1.mp4'
clip3 = VideoFileClip('test_videos/challenge.mp4')
challenge_clip = clip3.fl_image(process_image)
%time challenge_clip.write_videofile(challenge_output, audio=False)

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(challenge_output))
```

    [MoviePy] >>>> Building video test_videos_output/challenge_1.mp4
    [MoviePy] Writing video test_videos_output/challenge_1.mp4


    100%|██████████| 251/251 [00:08<00:00, 31.30it/s]


    [MoviePy] Done.
    [MoviePy] >>>> Video ready: test_videos_output/challenge_1.mp4 
    
    CPU times: user 3.85 s, sys: 318 ms, total: 4.16 s
    Wall time: 9.04 s






<video width="960" height="540" controls>
  <source src="test_videos_output/challenge_1.mp4">
</video>




# Using color filters

Now that's some messy results. Too many lines that aren't part of the lanes are being detected as such.

To improve this I'll use a much more specific filter using the colors of the lanes before sending the image for line detection.


```python
def color_filter(original):
    image = cv2.cvtColor(original, cv2.COLOR_RGB2HLS)
    white_mask = cv2.inRange(
        image,
        np.uint8([0, 200, 0]),
        np.uint8([255, 255, 255]),
    )
    yellow_mask = cv2.inRange(
        image,
        np.uint8([10, 0, 100]),
        np.uint8([40, 255, 255]),
    )
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked_original = cv2.bitwise_and(original, original, mask=mask)
    return masked_original

def improved_pipeline(image):
    masked = color_filter(image)
    lines = find_lines(masked)
    return draw_filtered_lines(image, lines)

challenge_output = 'test_videos_output/challenge_2.mp4'
clip3 = VideoFileClip('test_videos/challenge.mp4')
challenge_clip = clip3.fl_image(improved_pipeline)
%time challenge_clip.write_videofile(challenge_output, audio=False)

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(challenge_output))
```

    [MoviePy] >>>> Building video test_videos_output/challenge_2.mp4
    [MoviePy] Writing video test_videos_output/challenge_2.mp4


    100%|██████████| 251/251 [00:08<00:00, 28.91it/s]


    [MoviePy] Done.
    [MoviePy] >>>> Video ready: test_videos_output/challenge_2.mp4 
    
    CPU times: user 4.84 s, sys: 355 ms, total: 5.19 s
    Wall time: 9.36 s






<video width="960" height="540" controls>
  <source src="test_videos_output/challenge_2.mp4">
</video>




## That's it for now.
