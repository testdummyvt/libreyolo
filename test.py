from libreyolo import LIBREYOLORTDETR

model = LIBREYOLORTDETR("/home/testdummy/Downloads/libre_rtdetr_r18vd_dec3_6x_coco.pth", size="r18")

results = model.predict(
    source="/home/testdummy/Downloads/test/x.jpg",
    save=True,
    conf=0.4,
    iou=0.6,
)

# results = model.val(
#     data="coco.yaml",   # dataset config
#     batch=16,
#     imgsz=640,
#     conf_thres=0.001,            # low conf for mAP calculation
#     iou_thres=0.6,               # NMS IoU threshold
#     split="val",           # "val", "test", or "train"
#     save_json=True,       # save predictions as COCO JSON
#     # plots=True,            # reserved for future visualization
#     verbose=True,          # print per-class metrics
# )


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