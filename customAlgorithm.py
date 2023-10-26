import numpy as np
import cv2 as cv
from ipywidgets import fixed, interact


def yuv_in_range_filter(frame, lower_y, lower_u, lower_v, upper_y, upper_u, upper_v):
    from matplotlib import pyplot as plt

    plt.figure(figsize=(9, 9))

    plt.subplot(1, 1, 1)
    plt.imshow(
        cv.inRange(
            frame,
            np.array([lower_y, lower_u, lower_v]),
            np.array([upper_y, upper_u, upper_v]),
        )
    )
    plt.title("YUV RANGE FILTER")


def rgb_in_range_filter(frame, lower_r, lower_g, lower_b, upper_r, upper_g, upper_b):
    from matplotlib import pyplot as plt

    plt.figure(figsize=(9, 9))

    plt.subplot(1, 1, 1)
    plt.imshow(
        cv.inRange(
            frame,
            np.array([lower_r, lower_g, lower_b]),
            np.array([upper_r, upper_g, upper_b]),
        )
    )
    plt.title("RGB RANGE FILTER")


def rgb_colour_segmentation(frame):
    rgbImg = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    interact(
        rgb_in_range_filter,
        frame=fixed(rgbImg),
        lower_r=(0, 255, 1),
        lower_g=(0, 255, 1),
        lower_b=(0, 255, 1),
        upper_r=(0, 255, 1),
        upper_g=(0, 255, 1),
        upper_b=(0, 255, 1),
    )

    return frame


def multiSpaceSegmentation(img):
    """Convert the image into the yuv and rgb colour spaces,
    and threshold the two copies using inRange with harcoded
    values (found thanks to interactive functions like the one
    above). Obtain the thresholded/segmented image by combining
    the two copies.
    The two colour spaces extract in better ways different parts
    of the interested regions, so merging them returns a good
    segmentation.

    Args:
        img (numpy array): input image.

    Returns:
        numpy array: segmented image.
    """

    # Convert to rgb from bgr
    rgbImg = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Convert to yuv from bgr
    yuvImg = cv.cvtColor(img, cv.COLOR_BGR2YUV)

    # The following lower and upper values extract the best
    # possible appearance of the object, but not the entire table.
    rgbRanged = cv.inRange(
        rgbImg,
        np.array([105, 0, 0]),
        np.array([255, 255, 255]),
    )
    # The following lower and upper values extract the best
    # table contours.
    yuvRanged = cv.inRange(
        yuvImg,
        np.array([24, 16, 118]),
        np.array([255, 255, 255]),
    )

    # Add the two filtered images, to obtain a full representation
    # of the object and the table.
    rgbYuvAddition = cv.add(rgbRanged, yuvRanged)

    # Maximize the intensity for every non-zero intensity.
    rgbYuvAddition[rgbYuvAddition > 0] = 255

    # Fill the background with an intensity of 50, in order
    # to make it distinguishable from the object, the table
    # and the space under the table.
    cv.floodFill(
        rgbYuvAddition,
        None,
        seedPoint=(0, 0),
        newVal=(50, 0, 0),
        loDiff=(0, 0, 0, 0),
        upDiff=(0, 0, 0, 0),
    )

    # Make the distinction between the background with intensity 50
    # and the rest.
    rgbYuvAddition[rgbYuvAddition != 50] = 255

    # The Open operation will remove any noise, which I usually found
    # on the background of the last 2 objects.
    rgbYuvAddition = cv.morphologyEx(
        rgbYuvAddition, cv.MORPH_OPEN, np.ones((3, 3)), iterations=4
    )

    # Use the found regions to segment the original image.
    img[rgbYuvAddition == 50] = (0, 0, 0)
    img[rgbYuvAddition != 50] = (255, 255, 255)

    return img
