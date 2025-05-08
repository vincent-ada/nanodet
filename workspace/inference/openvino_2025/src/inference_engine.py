# src/inference_engine.py
import openvino as ov
import numpy as np
import cv2
import time


class OpenVinoInferencer:
    def __init__(self, model_xml_path, device="AUTO"):
        core = ov.Core()
        print(f"Compiling model: {model_xml_path} for device: {device}")
        self.compiled_model = core.compile_model(model_xml_path, device)
        self.infer_request = self.compiled_model.create_infer_request()
        print("Model compiled successfully.")

    def preprocess_image(self, image_bgr, input_width, input_height):
        """
        Preprocesses an image for the NanoDet model.
        image_bgr: OpenCV image in BGR format.
        """
        print(f"Original image shape: {image_bgr.shape}")
        resized_image = cv2.resize(image_bgr, (input_width, input_height))
        print(f"Resized image shape: {resized_image.shape}")

        # Transpose and expand dimensions: HWC to NCHW
        input_data = np.transpose(resized_image, (2, 0, 1))  # C, H, W
        input_data = np.expand_dims(input_data, axis=0)  # N, C, H, W
        input_data = input_data.astype(np.float32)
        # print(f"Input data shape for model: {input_data.shape}")
        return input_data

    def infer(self, input_data):
        """Performs inference on the preprocessed image data."""
        input_tensor = ov.Tensor(input_data)
        self.infer_request.set_input_tensor(input_tensor)

        print("Running inference...")
        start_time = time.perf_counter()
        self.infer_request.start_async()
        self.infer_request.wait()
        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000
        print(f"Inference finished in {inference_time_ms:.3f} ms.")

        output_tensor = self.infer_request.get_output_tensor()
        output_buffer = output_tensor.data
        print(f"Raw output_buffer shape: {output_buffer.shape}")
        return output_buffer
