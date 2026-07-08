import torch
from torchvision.ops import boxes as box_ops
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Tuple
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss
from model.robot_pred import RobotPredictor
from model.loss_function import CenterLossFunction, OrientationLossFunction
from config import CENTER_LOSS_WEIGHT, ORIENTATION_LOSS_WEIGHT

class RobotRoIHeads(RoIHeads):
    def __init__(self, roi_heads, in_features):
        super().__init__(
            box_roi_pool=roi_heads.box_roi_pool,
            box_head=roi_heads.box_head,
            box_predictor=roi_heads.box_predictor,
            fg_iou_thresh=roi_heads.proposal_matcher.high_threshold,
            bg_iou_thresh=roi_heads.proposal_matcher.low_threshold,
            batch_size_per_image=roi_heads.fg_bg_sampler.batch_size_per_image,
            positive_fraction=roi_heads.fg_bg_sampler.positive_fraction,
            bbox_reg_weights=roi_heads.box_coder.weights,
            score_thresh=roi_heads.score_thresh,
            nms_thresh=roi_heads.nms_thresh,
            detections_per_img=roi_heads.detections_per_img,
        )
        self.robot_head = RobotPredictor(in_features)
        self.center_loss = CenterLossFunction()
        self.orientation_loss = OrientationLossFunction()

    def postprocess_detections(
        self,
        class_logits,  # type: Tensor
        box_regression,  # type: Tensor
        center_preds,  # type: Tensor
        orientation_preds,  # type: Tensor
        proposals,  # type: List[Tensor]
        image_shapes,  # type: List[Tuple[int, int]]
    ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)
        pred_center_list = center_preds.split(boxes_per_image, 0)
        pred_orientation_list = orientation_preds.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        all_centers = []
        all_orientations = []
        for boxes, scores, centers, orientations, image_shape in zip(pred_boxes_list, pred_scores_list, pred_center_list, pred_orientation_list, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            inds = torch.where(scores > self.score_thresh)[0]
            boxes = boxes[inds]
            scores = scores[inds]
            labels = labels[inds]
            centers = centers[inds]
            orientations = orientations[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]
            centers = centers[keep]
            orientations = orientations[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only top scoring predictions
            keep = keep[: self.detections_per_img]
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]
            centers = centers[keep]
            orientations = orientations[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            all_centers.append(centers)
            all_orientations.append(orientations)
        return all_boxes, all_scores, all_labels, all_centers, all_orientations

    def forward(self, features, proposals, image_shapes, targets=None):
        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                if not t["boxes"].dtype in floating_point_types:
                    raise TypeError(f"target boxes must of float type, instead got {t['boxes'].dtype}")
                if not t["labels"].dtype == torch.int64:
                    raise TypeError(f"target labels must of int64 type, instead got {t['labels'].dtype}")
                if self.has_keypoint():
                    if not t["keypoints"].dtype == torch.float32:
                        raise TypeError(f"target keypoints must of float type, instead got {t['keypoints'].dtype}")

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)
        center_preds, orientation_preds = self.robot_head(box_features)

        result: List[Dict[str, torch.Tensor]] = []
        losses = {}
        if self.training:
            if labels is None:
                raise ValueError("labels cannot be None")
            
            center_targets = []
            orientation_targets = []

            for img_id in range(len(targets)):
                center_targets.append(
                    targets[img_id]["centers"].unsqueeze(0).expand(len(matched_idxs[img_id]), -1)
                )

                orientation_targets.append(
                    targets[img_id]["orientations"][matched_idxs[img_id]]
                )

            center_targets = torch.cat(center_targets, dim=0)
            orientation_targets = torch.cat(orientation_targets, dim=0)

            loss_center = self.center_loss(center_preds, center_targets)
            loss_orientation = self.orientation_loss(orientation_preds, orientation_targets)
            loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            losses = {"loss_center": loss_center*CENTER_LOSS_WEIGHT, "loss_orientation": loss_orientation*ORIENTATION_LOSS_WEIGHT, "loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        else:
            boxes, scores, labels, centers, orientations = self.postprocess_detections(class_logits, box_regression, center_preds, orientation_preds, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                        "centers": centers[i],
                        "orientation_logits": orientations[i],
                        "orientations": torch.argmax(orientations[i], dim=1),  # Convert index back to degrees
                    }
                )

        return result, losses