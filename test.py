from libreyolo import LIBREYOLORTDETR


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/home/testdummy/Downloads/libre_rtdetr_r18vd_dec3_6x_coco.pth")
    parser.add_argument("--size", type=str, default="r18")
    args = parser.parse_args()

    model = LIBREYOLORTDETR(args.model_path, size=args.size)
    results = model.val(
        data="coco.yaml",
        batch=16,
        imgsz=640,
        conf_thres=0.001,
        iou_thres=0.6,
        split="val",
        save_json=True,
        verbose=True,
    )


"""
R18
==================================================
Validation Results
==================================================
  metrics/precision: 0.4565
  metrics/recall: 0.6943
  metrics/mAP50: 0.6285
  metrics/mAP50-95: 0.4466
--------------------------------------------------
  Images processed: 5000
  Total time: 68.13s
  Speed: 13.6ms/image
==================================================

R34
==================================================
Validation Results
==================================================
  metrics/precision: 0.4816
  metrics/recall: 0.7235
  metrics/mAP50: 0.6603
  metrics/mAP50-95: 0.4728
--------------------------------------------------
  Images processed: 5000
  Total time: 71.52s
  Speed: 14.3ms/image
==================================================

R50
==================================================
Validation Results
==================================================
  metrics/precision: 0.5100
  metrics/recall: 0.7690
  metrics/mAP50: 0.7067
  metrics/mAP50-95: 0.5170
--------------------------------------------------
  Images processed: 5000
  Total time: 90.50s
  Speed: 18.1ms/image
==================================================

R50M
==================================================
Validation Results
==================================================
  metrics/precision: 0.5086
  metrics/recall: 0.7579
  metrics/mAP50: 0.6999
  metrics/mAP50-95: 0.5072
--------------------------------------------------
  Images processed: 5000
  Total time: 89.26s
  Speed: 17.9ms/image
==================================================

R101
==================================================
Validation Results
==================================================
  metrics/precision: 0.5118
  metrics/recall: 0.7833
  metrics/mAP50: 0.7216
  metrics/mAP50-95: 0.5295
--------------------------------------------------
  Images processed: 5000
  Total time: 118.13s
  Speed: 23.6ms/image
==================================================
"""
# results = model.predict(
#     source="/home/testdummy/Downloads/test/x.jpg",
#     save=True,
#     conf=0.4,
#     iou=0.6,
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