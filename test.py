from libreyolo import LIBREYOLO9NMSFree

model = LIBREYOLO9NMSFree("/home/testdummy/Downloads/yolov9_nmsfree.pt", size="t")

# results = model.predict(
#     source="/home/testdummy/datasets/coco/images/val2017/000000000785.jpg",
#     save=True,
#     conf=0.35,
#     iou=0.45,
# )

results = model.val(
    data="coco.yaml",   # dataset config
    batch=64,
    imgsz=640,
    conf=0.001,            # low conf for mAP calculation
    iou=0.6,               # NMS IoU threshold
    split="val",           # "val", "test", or "train"
    save_json=True,       # save predictions as COCO JSON
    # plots=True,            # reserved for future visualization
    verbose=True,          # print per-class metrics
)


# import torch

# model = torch.load("/home/testdummy/Downloads/yolov9_nmsfree.pt")

# print(model.keys())
# # print all model keys and values
# for key, value in model.items():
#     if key == "model" or key == "optimizer":
#         continue
#     print(key, value)


# model["model_family"] = "v9_nms_free"
# torch.save(model, "/home/testdummy/Downloads/yolov9_nmsfree.pt")