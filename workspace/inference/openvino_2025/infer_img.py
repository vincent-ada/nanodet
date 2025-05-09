# infer_img.py
import os
import cv2
from src.utils import load_config, load_image, save_image, overlay_bbox_cv
from src.inference_engine import OpenVinoInferencer
from src.postprocessor import PostProcessor


def main():
    # Determine project root directory dynamically
    project_root = os.path.dirname(os.path.abspath(__file__))

    config_path = os.path.join(project_root, "config/config.yaml")
    config = load_config(config_path)

    # Resolve paths from config relative to project root
    model_xml_path = os.path.join(project_root, config["model"]["xml_path"])
    input_image_path = os.path.join(
        project_root, config["image_processing"]["input_image_path"]
    )
    output_image_path = os.path.join(
        project_root, config["image_processing"]["output_image_path"]
    )

    # --- Inference ---
    inferencer = OpenVinoInferencer(model_xml_path, config["model"]["device"])

    original_image = load_image(input_image_path)
    if original_image is None:
        return

    input_data = inferencer.preprocess_image(
        original_image,
        config["image_processing"]["input_width"],
        config["image_processing"]["input_height"],
    )

    raw_output = inferencer.infer(input_data)

    # --- Post-processing ---
    postprocessor = PostProcessor(
        strides=config["postprocessing"]["strides"],
        reg_max=config["postprocessing"]["reg_max"],
        input_width=config["image_processing"]["input_width"],
        input_height=config["image_processing"]["input_height"],
        score_thr=config["postprocessing"]["score_thr"],
        nms_iou_threshold=config["postprocessing"]["nms_iou_threshold"],
        max_detections=config["postprocessing"]["max_detections_per_image"],
    )

    # Process output to get detections in the format {class_idx: [[x,y,x,y,s], ...]}
    detections = postprocessor.process(raw_output, original_image.shape)

    # --- Visualization and Saving ---
    processed_image_bgr = overlay_bbox_cv(
        original_image,
        detections,
        config["postprocessing"]["class_names"],
        score_thresh=config["postprocessing"]["score_thr"],
    )

    save_image(output_image_path, processed_image_bgr)

    # For display if running in an environment that supports it (optional)
    cv2.imshow("Processed Image", processed_image_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
