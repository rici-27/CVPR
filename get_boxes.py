import numpy as np
import cv2
import matplotlib.pyplot as plt


""" Original Datei """


def show_box_cv2(image, box, color=(0, 0, 255), thickness=2):
    """
    Draws a rectangle on an image using OpenCV.
    Args:
        image: The input image (numpy array).
        box: A bounding box, either 2D ([x_min, y_min, x_max, y_max]) or 3D ([x_min, y_min, z_min, x_max, y_max, z_max]).
        color: Color of the rectangle in BGR (default is blue).
        thickness: Thickness of the rectangle border (default is 2).
    Returns:
        The image with the rectangle drawn.
    """
    if len(box) == 4:  # 2D bounding box
        x_min, y_min, x_max, y_max = box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
    else:  # 3D bounding box
        x_min, y_min, z_min, x_max, y_max, z_max = box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
    return image

def show_mask_cv2(mask, image):
    rgb_mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    masked_image = image.copy()
    masked_image = np.where(rgb_mask, np.array([255,0,0], dtype='uint8'), masked_image)
    return cv2.addWeighted(image, 0.3, masked_image, 0.7, 0)

def mask2D_to_bbox(gt2D):
    y_indices, x_indices = np.where(gt2D > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = gt2D.shape
    bbox_shift = np.random.randint(0, 6, 1)[0]
    scale_y, scale_x = gt2D.shape
    bbox_shift_x = int(bbox_shift * scale_x/256)
    bbox_shift_y = int(bbox_shift * scale_y/256)
    x_min = max(0, x_min - bbox_shift_x)
    x_max = min(W-1, x_max + bbox_shift_x)
    y_min = max(0, y_min - bbox_shift_y)
    y_max = min(H-1, y_max + bbox_shift_y)
    boxes = np.array([x_min, y_min, x_max, y_max])
    return boxes

def mask3D_to_bbox(gt3D):
    b_dict = {}
    z_indices, y_indices, x_indices = np.where(gt3D > 0)
    z_min, z_max = np.min(z_indices), np.max(z_indices)
    # middle of z_indices
    z_middle = z_indices[len(z_indices)//2]
    D, H, W = gt3D.shape
    b_dict['z_min'] = z_min
    b_dict['z_max'] = z_max
    b_dict['z_mid'] = z_middle

    gt_mid = gt3D[z_middle]

    box_2d = mask2D_to_bbox(gt_mid)
    x_min, y_min, x_max, y_max = box_2d
    b_dict['z_mid_x_min'] = x_min
    b_dict['z_mid_y_min'] = y_min
    b_dict['z_mid_x_max'] = x_max
    b_dict['z_mid_y_max'] = y_max

    assert z_min == max(0, z_min)
    assert z_max == min(D-1, z_max)
    return b_dict


