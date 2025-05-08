import os
import cv2
import torch
import time
import sys  # Import sys for exiting

from nanodet.util import cfg, load_config, overlay_bbox_cv
from predictor import Predictor
from argparse import ArgumentParser

from IPython.display import display
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}


def cv2_imshow(a, convert_bgr_to_rgb=True):
    """A replacement for cv2.imshow() for use in Jupyter notebooks.
    Args:
        a: np.ndarray. shape (N, M) or (N, M, 1) is an NxM grayscale image. shape
            (N, M, 3) is an NxM BGR color image. shape (N, M, 4) is an NxM BGRA color
            image.
        convert_bgr_to_rgb: switch to convert BGR to RGB channel.
    """
    a = a.clip(0, 255).astype("uint8")
    # cv2 stores colors as BGR; convert to RGB
    if convert_bgr_to_rgb and a.ndim == 3:
        if a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
        else:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    display(Image.fromarray(a))


def get_video_properties(cap):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    return width, height, fps


def validate_input_extension(input_path, input_type):
    """
    Validates if the input file extension matches the expected input type.

    Args:
        input_path (str): Path to the input file.
        input_type (str): Expected input type ('image' or 'video').

    Returns:
        bool: True if valid, False otherwise. Prints an error message on failure.
    """
    if not os.path.isfile(input_path):
        print(f"Error: Input file not found at '{input_path}'")
        return False

    _, ext = os.path.splitext(input_path)
    ext = ext.lower()

    if input_type == "video":
        if ext not in VIDEO_EXTENSIONS:
            print(
                f"Error: Config input_type is 'video', but file extension '{ext}' "
                f"is not a recognized video format ({', '.join(sorted(list(VIDEO_EXTENSIONS)))})."
            )
            return False
    elif input_type == "image":
        if ext not in IMAGE_EXTENSIONS:
            print(
                f"Error: Config input_type is 'image', but file extension '{ext}' "
                f"is not a recognized image format ({', '.join(sorted(list(IMAGE_EXTENSIONS)))})."
            )
            return False
    else:
        print(f"Error: Unknown input_type '{input_type}' specified in config.")
        return False

    return True


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    load_config(cfg, args.config)

    model_path = cfg.inference.model_path
    output_path = cfg.inference.output
    input_path = cfg.inference.input
    input_type = cfg.inference.input_type

    if not validate_input_extension(input_path, input_type):
        sys.exit(1)

    predictor = Predictor(cfg, model_path, device="cpu")

    if input_type == "video":

        cap = cv2.VideoCapture(input_path)
        width, height, fps = get_video_properties(cap)
        fourcc = cv2.VideoWriter.fourcc(*"FFV1")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_count += 1
            meta, results = predictor.inference(frame)
            processed_frame = overlay_bbox_cv(
                meta["raw_img"][0], results[0], cfg.class_names, score_thresh=0.7
            )

            cv2.imshow("Video Inference", processed_frame)

            if cfg.inference.write_output:
                if out.isOpened():
                    out.write(processed_frame)
                else:
                    print("Warning: Video writer is not open. Cannot write frame.")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        end_time = time.time()
        print(f"Processed {frame_count} frames in {end_time - start_time} seconds.")

        cap.release()

        if cfg.inference.get("write_output", False) and out.isOpened():
            out.release()
        cv2.destroyAllWindows()

    elif input_type == "image":
        meta, results = predictor.inference(input_path)
        processed_image = overlay_bbox_cv(
            meta["raw_img"][0], results[0], cfg.class_names, score_thresh=0.7
        )

        cv2.imshow("Image inference", processed_image)
        print("Press any key to close the image window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"Error: Unsupported input_type '{input_type}' found in config.")
        sys.exit(1)
