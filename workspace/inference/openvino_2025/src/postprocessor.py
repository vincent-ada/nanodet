# src/postprocessor.py
import torch
import torch.nn.functional as F
from torchvision.ops import nms as torch_nms
import numpy as np


# Helper function from Integral class logic
def _integral_distribution_project(reg_preds_tensor, reg_max):
    """
    Calculates integral result from distribution.
    Equivalent to the forward pass of the Integral nn.Module.
    """
    project_tensor = torch.linspace(
        0,
        reg_max,
        reg_max + 1,
        device=reg_preds_tensor.device,
        dtype=reg_preds_tensor.dtype,
    )
    shape = reg_preds_tensor.size()
    reg_preds_reshaped = reg_preds_tensor.reshape(*shape[:-1], 4, reg_max + 1)
    softmax_preds = F.softmax(reg_preds_reshaped, dim=-1)
    integral_result = F.linear(softmax_preds, project_tensor)
    return integral_result


class PostProcessor:
    def __init__(
        self,
        strides,
        reg_max,
        input_width,
        input_height,
        score_thr,
        nms_iou_threshold,
        max_detections,
    ):
        self.strides = strides
        self.reg_max = reg_max
        self.input_width = input_width
        self.input_height = input_height
        self.score_thr = score_thr
        self.nms_cfg = dict(
            type="nms", iou_threshold=nms_iou_threshold
        )  # type is used by batched_nms
        self.max_detections = max_detections
        self.center_priors = self._generate_center_priors()

    def _get_single_level_center_point(
        self, featmap_size, stride, dtype, device, flatten=True
    ):
        h, w = featmap_size
        x_range = (torch.arange(w, dtype=dtype, device=device) + 0.5) * stride
        y_range = (torch.arange(h, dtype=dtype, device=device) + 0.5) * stride
        y, x = torch.meshgrid(y_range, x_range, indexing="ij")
        if flatten:
            y = y.flatten()
            x = x.flatten()
        return y, x

    def _generate_center_priors(self):
        feature_map_sizes = [
            (self.input_height // stride, self.input_width // stride)
            for stride in self.strides
        ]
        mlvl_center_priors = []
        for i, stride_val in enumerate(self.strides):
            y, x = self._get_single_level_center_point(
                feature_map_sizes[i], stride_val, torch.float32, "cpu"
            )
            current_strides_tensor = x.new_full((x.shape[0],), stride_val)
            priors = torch.stack(
                [x, y, current_strides_tensor, current_strides_tensor], dim=-1
            )
            mlvl_center_priors.append(priors.unsqueeze(0))
        center_priors = torch.cat(mlvl_center_priors, dim=1)
        return center_priors

    def _distance2bbox(self, points, distance, max_shape=None):
        x1 = points[..., 0] - distance[..., 0]
        y1 = points[..., 1] - distance[..., 1]
        x2 = points[..., 0] + distance[..., 2]
        y2 = points[..., 1] + distance[..., 3]
        if max_shape is not None:
            x1 = x1.clamp(min=0, max=max_shape[1])
            y1 = y1.clamp(min=0, max=max_shape[0])
            x2 = x2.clamp(min=0, max=max_shape[1])
            y2 = y2.clamp(min=0, max=max_shape[0])
        return torch.stack([x1, y1, x2, y2], -1)

    def _batched_nms(self, boxes, scores, idxs, nms_cfg, class_agnostic=False):
        nms_cfg_ = nms_cfg.copy()
        class_agnostic = nms_cfg_.pop("class_agnostic", class_agnostic)
        if class_agnostic:
            boxes_for_nms = boxes
        else:
            max_coordinate = boxes.max() if boxes.numel() > 0 else 0
            offsets = idxs.to(boxes) * (max_coordinate + 1)
            boxes_for_nms = boxes + offsets[:, None]

        iou_threshold = nms_cfg_.pop(
            "iou_threshold"
        )  # Use the correct key for torch_nms

        keep = torch_nms(boxes_for_nms, scores, iou_threshold)

        boxes = boxes[keep]
        scores = scores[keep]
        return torch.cat([boxes, scores[:, None]], -1), keep

    def _multiclass_nms(
        self,
        multi_bboxes,
        multi_scores,
        score_thr,
        nms_cfg,
        max_num=-1,
        score_factors=None,
    ):
        num_classes = multi_scores.size(1) - 1  # Exclude background
        if multi_bboxes.shape[1] > 4:
            bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
        else:
            bboxes = multi_bboxes[:, None].expand(multi_scores.size(0), num_classes, 4)

        scores = multi_scores[:, :-1]  # Assuming last column is background

        valid_mask = scores > score_thr

        # Filter bboxes and scores
        # For ONNX export, masked_select was used. For direct PyTorch, boolean indexing is cleaner.
        # However, to stick to the notebook's way as much as possible:
        selected_bboxes = torch.masked_select(
            bboxes, torch.stack((valid_mask, valid_mask, valid_mask, valid_mask), -1)
        ).view(-1, 4)

        if score_factors is not None:  # Not used in notebook example
            scores = scores * score_factors[:, None]

        selected_scores = torch.masked_select(scores, valid_mask)
        selected_labels = valid_mask.nonzero(as_tuple=False)[:, 1]

        if selected_bboxes.numel() == 0:
            return multi_bboxes.new_zeros((0, 5)), multi_bboxes.new_zeros(
                (0,), dtype=torch.long
            )

        dets, keep = self._batched_nms(
            selected_bboxes, selected_scores, selected_labels, nms_cfg
        )

        if max_num > 0:
            dets = dets[:max_num]
            keep = keep[:max_num]

        return dets, selected_labels[keep]

    def process(self, raw_output, original_image_shape):
        # raw_output shape: (1, num_proposals, 33)
        # 33 = 1 (class_score) + 32 (distribution for 4 box coords, (reg_max+1)*4)

        cls_preds_np = raw_output[0, :, 0]  # Shape: (num_proposals,)
        bbox_dist_prob_np = raw_output[0, :, 1:]  # Shape: (num_proposals, 32)

        cls_preds_tensor = torch.from_numpy(cls_preds_np).float()
        bbox_dist_prob_tensor = torch.from_numpy(bbox_dist_prob_np).float()

        # Integral projection for bounding box regression
        # center_priors shape: (1, num_total_priors, 4)
        # We need center_priors[0] for a single image in batch
        # dis_preds = distribution_project(reg_preds) * center_priors[..., 2, None]
        # reg_preds are bbox_dist_prob_tensor
        # center_priors[0, :, 2] is the stride for each prior

        dis_preds_dist = _integral_distribution_project(
            bbox_dist_prob_tensor, self.reg_max
        )  # (num_proposals, 4)
        # Multiply by strides: center_priors[0, :, 2] is stride for x, can be used for all 4 distances
        # Or center_priors[0, :, 2:3] to keep dimensions for broadcasting
        strides_for_priors = self.center_priors[0, :, 2:3]  # Shape: (num_proposals, 1)
        dis_preds = dis_preds_dist * strides_for_priors  # (num_proposals, 4)

        # Decode bboxes
        # center_priors[0, :, :2] gives (x,y) for each prior
        bboxes = self._distance2bbox(
            self.center_priors[0, :, :2],
            dis_preds,
            max_shape=(self.input_height, self.input_width),
        )

        scores = cls_preds_tensor.sigmoid()  # (num_proposals,)

        # Prepare for multiclass_nms:
        # The notebook processes scores for a single class model.
        # 'scores' should be (num_proposals, num_classes_including_background)
        # Here, it's (num_proposals,). We make it (num_proposals, 2) for one class + background.
        scores_for_nms = scores.unsqueeze(dim=-1)  # (num_proposals, 1)
        padding = scores_for_nms.new_zeros(
            scores_for_nms.shape[0], 1
        )  # Background class
        scores_for_nms = torch.cat(
            [scores_for_nms, padding], dim=1
        )  # (num_proposals, 2)

        # bboxes from distance2bbox are already (num_proposals, 4)

        det_bboxes_scores, det_labels = self._multiclass_nms(
            bboxes,
            scores_for_nms,
            score_thr=self.score_thr,
            nms_cfg=self.nms_cfg,
            max_num=self.max_detections,
        )
        # det_bboxes_scores shape: (k, 5) [x1, y1, x2, y2, score]
        # det_labels shape: (k,) [class_index]

        # Scaling to original image dimensions
        original_h, original_w = original_image_shape[:2]
        scale_x = original_w / self.input_width
        scale_y = original_h / self.input_height

        scaled_bboxes_np = det_bboxes_scores[:, :4].cpu().numpy().copy()
        scaled_bboxes_np[:, 0] *= scale_x  # x1
        scaled_bboxes_np[:, 1] *= scale_y  # y1
        scaled_bboxes_np[:, 2] *= scale_x  # x2
        scaled_bboxes_np[:, 3] *= scale_y  # y2

        # Clip to image dimensions
        scaled_bboxes_np[:, 0] = np.clip(scaled_bboxes_np[:, 0], 0, original_w)
        scaled_bboxes_np[:, 1] = np.clip(scaled_bboxes_np[:, 1], 0, original_h)
        scaled_bboxes_np[:, 2] = np.clip(scaled_bboxes_np[:, 2], 0, original_w)
        scaled_bboxes_np[:, 3] = np.clip(scaled_bboxes_np[:, 3], 0, original_h)

        scores_np = det_bboxes_scores[:, 4:5].cpu().numpy()  # Keep as (k,1)
        labels_np = det_labels.cpu().numpy()

        final_output_for_overlay = {}
        # Assuming all detections from NMS (det_labels) are for the classes in config.class_names
        # and det_labels are 0-indexed.
        for i in range(len(labels_np)):
            label_idx = labels_np[i]
            if label_idx not in final_output_for_overlay:
                final_output_for_overlay[label_idx] = []

            box_with_score = np.concatenate(
                [
                    scaled_bboxes_np[i, :4].astype(np.float32),
                    scores_np[i].astype(np.float32),
                ]
            ).tolist()
            final_output_for_overlay[label_idx].append(box_with_score)

        return final_output_for_overlay
