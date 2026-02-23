import torch
from libreyolo.rtdetr import LIBREYOLORTDETR


rt_detr_weights = [
    "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth",
    "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r34vd_dec4_6x_coco_from_paddle.pth",
    "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_m_6x_coco_from_paddle.pth",
    "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth",
    "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r101vd_6x_coco_from_paddle.pth"
]
libre_rt_detr_weights = [
    "weights/libre_rtdetr_r18vd_dec3_6x_coco.pth",
    "weights/libre_rtdetr_r34vd_dec4_6x_coco.pth",
    "weights/libre_rtdetr_r50vd_m_6x_coco.pth",
    "weights/libre_rtdetr_r50vd_6x_coco.pth",
    "weights/libre_rtdetr_r101vd_6x_coco.pth"
]

for rt_detr_weight, libre_rt_detr_weight in zip(rt_detr_weights, libre_rt_detr_weights):
    weights = torch.hub.load_state_dict_from_url(rt_detr_weight)
    weights = weights["ema"]["module"]
    torch.save(weights, libre_rt_detr_weight)
