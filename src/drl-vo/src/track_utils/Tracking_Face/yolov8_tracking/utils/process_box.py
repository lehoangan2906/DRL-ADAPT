import numpy as np

def calculate_iou_and_center_distance(A, B):
    # IoU calculation
    x_left = np.maximum(A[:, None, 0], B[:, 0])
    y_top = np.maximum(A[:, None, 1], B[:, 1])
    x_right = np.minimum(A[:, None, 2], B[:, 2])
    y_bottom = np.minimum(A[:, None, 3], B[:, 3])

    intersection_area = np.maximum(x_right - x_left, 0) * np.maximum(y_bottom - y_top, 0)

    area_A = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1])
    area_B = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])

    union_area = (area_A[:, None] + area_B) - intersection_area

    iou = intersection_area / union_area

    # Center distance calculation
    center_A = np.array([(A[:, 0] + A[:, 2]) / 2, (A[:, 1] + A[:, 3]) / 2]).T
    center_B = np.array([(B[:, 0] + B[:, 2]) / 2, (B[:, 1] + B[:, 3]) / 2]).T

    center_dist = np.linalg.norm(center_A - center_B, axis=1)

    return iou, center_dist


def find_Center(box):
    return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]