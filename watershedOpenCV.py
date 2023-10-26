import cv2 as cv
import numpy as np


def watershedSegmentation(img):
    """Segment the image by distinguishing the foreground and the background
    regions, and then use them to compute a segmentation using watershed.

    Args:
        img (numpy array): input image.

    Returns:
        numpy array: segmented image.
    """
    # Convert to gray from bgr
    grayImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Apply a Gaussian Blur to the grayImg to remove noise
    grayImg = cv.GaussianBlur(grayImg, (3, 3), 0)

    # Threshold the gray image
    thresh = cv.threshold(
        grayImg,
        0,
        255,
        cv.THRESH_BINARY_INV + cv.THRESH_OTSU,
    )[1]

    # Finding sure foreground area
    distTransform = cv.distanceTransform(thresh, cv.DIST_L2, 5)
    sureForeground = cv.threshold(distTransform, 0.65 * distTransform.max(), 255, 0)[1]
    sureForeground = np.uint8(sureForeground)

    # Finding unknown region
    unknown = cv.subtract(thresh, sureForeground)

    # Marker labelling
    markers = cv.connectedComponents(sureForeground)[1]

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    # Apply watershed to modify the markers, fiding the final regions.
    markers = cv.watershed(img, markers)

    # Use the found regions and mark them with black and white on the input image.
    img[markers > 1] = [0, 0, 0]
    img[markers <= 1] = [255, 255, 255]

    return img
