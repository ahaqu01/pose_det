import cv2
import collections
import numpy as np
import mmcv
import torch
from torchvision.transforms import functional as F
from .models.top_down import TopDown
from .utils.bbox_func import xyxy2xywh, box2cs
from .utils.transforms import get_affine_transform


class pose_detecter(object):
    def __init__(self,
                 model_cfg="/workspace/huangniu_demo/pose_det/src/configs/hrnet_w32_coco_256x192.py",
                 model_weights="/workspace/huangniu_demo/pose_det/src/weights/hrnet_w32_coco_256x192-c78dce93_20200708.pth",
                 device=None):
        # get config
        if isinstance(model_cfg, str):
            self.config = mmcv.Config.fromfile(model_cfg)
        # set detecter device
        self.device = device
        # create model and load weights
        self.model = TopDown(config=self.config)
        weights_state_dict = torch.load(model_weights)["state_dict"]
        model_state_dict = collections.OrderedDict()
        for n_p_1, n_p_2 in zip(self.model.state_dict().items(), weights_state_dict.items()):
            if n_p_1[1].shape == n_p_2[1].shape:
                model_state_dict[n_p_1[0]] = n_p_2[1]
        self.model.load_state_dict(model_state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

    def pre_process(self, img_src, bboxs):
        # img: ndarray, (H, W, 3), RGB
        flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12],
                      [13, 14], [15, 16]]
        bboxs_xyxy = bboxs.cpu().numpy()
        bboxes_xywh = xyxy2xywh(bboxs_xyxy)
        batch_data = []
        for bbox in bboxes_xywh:
            # img transforms
            # do warpAffine
            img = img_src.copy()
            center, scale = box2cs(bbox, self.config.data_cfg["image_size"])
            trans = get_affine_transform(center, scale, 0, self.config.data_cfg["image_size"])
            img = cv2.warpAffine(img, trans, (int(self.config.data_cfg["image_size"][0]), int(self.config.data_cfg["image_size"][1])), flags=cv2.INTER_LINEAR)
            # to tensor
            img = F.to_tensor(img)
            # normalize
            img = F.normalize(img, mean=self.config.test_pipeline[3]["mean"], std=self.config.test_pipeline[3]["std"])

            joints_3d = np.zeros((self.config.data_cfg.num_joints, 3), dtype=np.float32)
            joints_3d_visible = np.zeros((self.config.data_cfg.num_joints, 3), dtype=np.float32)

            # prepare data
            data = {
                'img': img,
                'center': center,
                'scale': scale,
                'bbox_score': bbox[4],
                'bbox_id': 0,  # need to be assigned if batch_size > 1
                'joints_3d': joints_3d,
                'joints_3d_visible': joints_3d_visible,
                'rotation': 0,
                'ann_info': {
                    'image_size': np.array(self.config.data_cfg['image_size']),
                    'num_joints': self.config.data_cfg['num_joints'],
                    'flip_pairs': flip_pairs
                }
            }
            batch_data.append(data)
        processed_data = {"img": torch.stack([data_i["img"] for data_i in batch_data]).to(self.device),
                          "center": [data_i["center"] for data_i in batch_data],
                          "scale": [data_i["scale"] for data_i in batch_data],
                          "bbox_score": [data_i["bbox_score"] for data_i in batch_data],
                          "bbox_id": [data_i["bbox_id"] for data_i in batch_data],
                          "joints_3d": [data_i["joints_3d"] for data_i in batch_data],
                          "joints_3d_visible": [data_i["joints_3d_visible"] for data_i in batch_data],
                          "rotation": [data_i["rotation"] for data_i in batch_data],
                          "ann_info": [data_i["ann_info"] for data_i in batch_data],
                          }
        return processed_data

    def post_process(self, result, bboxs):
        preds, boxes, bbox_ids, output_heatmap = result['preds'], result['boxes'], result['bbox_ids'], result[
            'output_heatmap']
        person_results = {}
        for i in range(bboxs.shape[0]):
            person_results_value = {"keypoints": preds[i],
                                    "bbox": bboxs[i].cpu().numpy()}
            person_results[i] = person_results_value
        return person_results

    @torch.no_grad()
    def inference(self, img, bboxs):
        # img: ndarray, (H, W, 3), RGB
        # bboxs: torch, (N, 5), (:, :4)分别为bbox左上角右下角xy坐标, (:, 4)代表置信度
        if bboxs.shape[0] >= 1:
            # do data pre_process
            processed_data = self.pre_process(img, bboxs)
            # pure model inference
            result = self.model(processed_data)
            # do results post process
            result = self.post_process(result, bboxs)
            return result
        else:
            return {}
