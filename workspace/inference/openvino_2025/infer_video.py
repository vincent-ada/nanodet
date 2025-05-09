# infer_video.py
import os
import cv2
from src.utils import (
    load_config,
    overlay_bbox_cv,
    get_video_properties,
)
from src.inference_engine import OpenVinoInferencer
from src.postprocessor import PostProcessor


def main():
    # Determine project root directory dynamically
    project_root = os.path.dirname(os.path.abspath(__file__))

    config_path = os.path.join(project_root, "config/config.yaml")
    config = load_config(config_path)

    # Resolve paths from config relative to project root
    model_xml_path = os.path.join(project_root, config["model"]["xml_path"])
    input_video_path = os.path.join(
        project_root, config["image_processing"]["input_video_path"]
    )
    output_video_path = os.path.join(
        project_root, config["image_processing"]["output_video_path"]
    )

    # --- Inference ---
    inferencer = OpenVinoInferencer(model_xml_path, config["model"]["device"])

    postprocessor = PostProcessor(
        strides=config["postprocessing"]["strides"],
        reg_max=config["postprocessing"]["reg_max"],
        input_width=config["image_processing"]["input_width"],
        input_height=config["image_processing"]["input_height"],
        score_thr=config["postprocessing"]["score_thr"],
        nms_iou_threshold=config["postprocessing"]["nms_iou_threshold"],
        max_detections=config["postprocessing"]["max_detections_per_image"],
    )

    cap = cv2.VideoCapture(input_video_path)
    width, height, fps = get_video_properties(cap)
    fourcc = cv2.VideoWriter.fourcc(*"FFV1")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while True:
        ret, original_frame = cap.read()
        if not ret:
            # cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            # continue
            break

        input_data = inferencer.preprocess_image(
            original_frame,
            config["image_processing"]["input_width"],
            config["image_processing"]["input_height"],
        )

        raw_output = inferencer.infer(input_data)

        detections = postprocessor.process(raw_output, original_frame.shape)

        processed_image_bgr = overlay_bbox_cv(
            original_frame,
            detections,
            config["postprocessing"]["class_names"],
            score_thresh=config["postprocessing"]["score_thr"],
        )

        cv2.imshow("Video Inference", processed_image_bgr)

        out.write(processed_image_bgr)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
