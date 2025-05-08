# src/utils.py
import cv2
import yaml
import numpy as np
import os
import random


def load_config(config_path="config/config.yaml"):
    """Loads YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_image(image_path):
    """Loads an image using OpenCV."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    return cv2.imread(image_path)


def save_image(image_path, image_data):
    """Saves an image using OpenCV. Creates directory if it doesn't exist."""
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    cv2.imwrite(image_path, image_data)
    print(f"Processed image saved to: {image_path}")


def overlay_bbox_cv(img, dets, class_names, score_thresh):
    """
    Overlays bounding boxes on an image.
    dets: dictionary {label_index: [[x0, y0, x1, y1, score], ...]}
    """
    all_box = []
    # dets in the notebook is {img_id (0): [[x,y,x,y,s],...]}
    # but overlay_bbox_cv expects dets[label_idx]
    # The notebook passes det_result which is {0: boxes_for_class_0_if_only_one_class}
    # and class_names = ['object'], so label=0 maps to 'object'.
    # This adaptation assumes dets is the direct output for a single image (list of detections)
    # or it's already in the {label: boxes} format.
    # The notebook code does:
    # classes = det_labels.detach().cpu().numpy()
    # inds = classes == 0
    # det_result[img_id] = np.concatenate([scaled_bboxes[inds, :4], scores_for_concat], axis=1).tolist()
    # overlay_bbox_cv(image, det_result, ['object'], ...) -> here dets becomes det_result
    # So, the `dets` passed to this function should be det_result from the notebook.
    # det_result is {0: [[x,y,x,y,s], ...]}

    for (
        label_idx
    ) in (
        dets
    ):  # Iterates through image_ids or class_ids depending on how `dets` is structured
        if not isinstance(dets[label_idx], list):  # Skip if not a list of bboxes
            continue
        for bbox_with_score in dets[label_idx]:
            if len(bbox_with_score) == 5:  # [x0, y0, x1, y1, score]
                score = bbox_with_score[-1]
                if score > score_thresh:
                    # The label_idx from dets keys should map to class_names
                    # In the notebook, det_result is keyed by img_id (0), and class is implicitly 0.
                    # For simplicity, if class_names has one entry, we use label_idx 0 for class_names.
                    # If class_names has multiple entries, label_idx should correspond to class_names index.
                    # The notebook uses `classes = det_labels.detach().cpu().numpy()` then `inds = classes == 0`
                    # which means it filters for a specific class (0). The `det_result` is built for this class.
                    # So, effectively, the 'label' passed to this function via `dets` key is the class index.
                    current_label = label_idx if label_idx < len(class_names) else 0

                    all_box.append(
                        [current_label]
                        + [int(i) for i in bbox_with_score[:4]]
                        + [score]
                    )
            else:
                print(f"Warning: Bounding box data format incorrect: {bbox_with_score}")

    all_box.sort(key=lambda v: v[5], reverse=True)  # Sort by score, highest first

    img_copy = img.copy()
    for box in all_box:
        label, x0, y0, x1, y1, score = box

        # Ensure label is within bounds of class_names
        if label >= len(class_names):
            print(
                f"Warning: Label index {label} out of bounds for class_names (len={len(class_names)}). Skipping box."
            )
            continue

        color = [random.randint(0, 255) for _ in range(3)]  # Random color for each box
        text = "{}:{:.1f}%".format(class_names[label], score * 100)
        txt_color = (
            (0, 0, 0) if sum(color) > 382 else (255, 255, 255)
        )  # Black or white text based on box color
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.5, 2)[0]
        cv2.rectangle(img_copy, (x0, y0), (x1, y1), color, 2)

        # Background for text
        cv2.rectangle(
            img_copy,
            (x0, y0 - txt_size[1] - 2),
            (x0 + txt_size[0], y0 - 2),
            color,
            -1,
        )
        cv2.putText(img_copy, text, (x0, y0 - 5), font, 0.5, txt_color, thickness=1)
    return img_copy


def warp_boxes(boxes, M, width, height):
    n = len(boxes)
    if n:
        xy = np.ones((n * 4, 3))
        xy[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
        xy = xy @ M.T
        xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        return xy.astype(np.float32)
    else:
        return boxes
