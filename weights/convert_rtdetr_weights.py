import torch
from libreyolo.rtdetr import LIBREYOLORTDETR


rt_detr_weights = ""
libre_rt_detr_weights = ""

weights = torch.load(rt_detr_weights)
weights = weights["ema"]["module"]

# save weights
torch.save(weights, libre_rt_detr_weights)
