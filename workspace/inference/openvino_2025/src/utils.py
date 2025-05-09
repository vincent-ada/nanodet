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

        color = (0, 255, 0)  # Green color
        text = "{}:{:.1f}%".format(class_names[label], score * 100)
        txt_color = (0, 0, 0)  # Black or white text based on box color
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


def get_video_properties(cap):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    return width, height, fps
