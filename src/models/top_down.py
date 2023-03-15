# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch.nn as nn
import mmcv
from mmcv.image import imwrite
from mmcv.utils.misc import deprecated_api_warning
from mmcv.visualization.image import imshow
from mmpose.core import imshow_bboxes, imshow_keypoints

from .backbones.hrnet import HRNet as backbone
from .heads.topdown_heatmap_simple_head import TopdownHeatmapSimpleHead as head


class TopDown(nn.Module):
    """Top-down pose detectors.
    """

    def __init__(self, config, return_heatmap=False):
        super().__init__()
        self.backbone_cfg = config.model["backbone"]
        self.head_cfg = config.model["keypoint_head"]
        self.test_cfg = config.model["test_cfg"]
        self.return_heatmap = return_heatmap

        self.backbone = backbone(extra=self.backbone_cfg["extra"], in_channels=self.backbone_cfg["in_channels"])
        self.head = head(in_channels=self.head_cfg["in_channels"],
                         out_channels=self.head_cfg["out_channels"],
                         num_deconv_layers=self.head_cfg["num_deconv_layers"],
                         num_deconv_filters=(256, 256, 256),
                         num_deconv_kernels=(4, 4, 4),
                         extra=self.head_cfg["extra"],
                         in_index=0,
                         input_transform=None,
                         align_corners=False,
                         test_cfg=self.test_cfg)

    def forward(self, data):
        img = data["img"]
        batch_size, _, img_height, img_width = img.shape

        features = self.backbone(img)
        output_heatmap = self.head.inference_model(features, flip_pairs=None)

        if self.test_cfg["flip_test"]:
            img_flipped = img.flip(3)
            features_flipped = self.backbone(img_flipped)
            output_flipped_heatmap = self.head.inference_model(features_flipped, data["ann_info"][0]["flip_pairs"])
            output_heatmap = (output_heatmap + output_flipped_heatmap) * 0.5

        keypoint_result = self.head.decode(data, output_heatmap)

        result = {}
        result.update(keypoint_result)

        if not self.return_heatmap:
            output_heatmap = None
        result['output_heatmap'] = output_heatmap
        return result

    def forward_dummy(self, img):
        """Used for computing network FLOPs.

        See ``tools/get_flops.py``.

        Args:
            img (torch.Tensor): Input image.

        Returns:
            Tensor: Output heatmaps.
        """
        output = self.backbone(img)
        if self.with_neck:
            output = self.neck(output)
        if self.with_keypoint:
            output = self.keypoint_head(output)
        return output

    @deprecated_api_warning({'pose_limb_color': 'pose_link_color'},
                            cls_name='TopDown')
    def show_result(self,
                    img,
                    result,
                    skeleton=None,
                    kpt_score_thr=0.3,
                    bbox_color='green',
                    pose_kpt_color=None,
                    pose_link_color=None,
                    text_color='white',
                    radius=4,
                    thickness=1,
                    font_scale=0.5,
                    bbox_thickness=1,
                    win_name='',
                    show=False,
                    show_keypoint_weight=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (list[dict]): The results to draw over `img`
                (bbox_result, pose_result).
            skeleton (list[list]): The connection of keypoints.
                skeleton is 0-based indexing.
            kpt_score_thr (float, optional): Minimum score of keypoints
                to be shown. Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            pose_kpt_color (np.array[Nx3]`): Color of N keypoints.
                If None, do not draw keypoints.
            pose_link_color (np.array[Mx3]): Color of M links.
                If None, do not draw links.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            radius (int): Radius of circles.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            show (bool): Whether to show the image. Default: False.
            show_keypoint_weight (bool): Whether to change the transparency
                using the predicted confidence scores of keypoints.
            wait_time (int): Value of waitKey param.
                Default: 0.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            Tensor: Visualized img, only if not `show` or `out_file`.
        """
        img = mmcv.imread(img)
        img = img.copy()

        bbox_result = []
        bbox_labels = []
        pose_result = []
        for res in result:
            if 'bbox' in res:
                bbox_result.append(res['bbox'])
                bbox_labels.append(res.get('label', None))
            pose_result.append(res['keypoints'])

        if bbox_result:
            bboxes = np.vstack(bbox_result)
            # draw bounding boxes
            imshow_bboxes(
                img,
                bboxes,
                labels=bbox_labels,
                colors=bbox_color,
                text_color=text_color,
                thickness=bbox_thickness,
                font_scale=font_scale,
                show=False)

        if pose_result:
            imshow_keypoints(img, pose_result, skeleton, kpt_score_thr,
                             pose_kpt_color, pose_link_color, radius,
                             thickness)

        if show:
            imshow(img, win_name, wait_time)

        if out_file is not None:
            imwrite(img, out_file)

        return img
