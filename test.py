from libreyolo import LIBREYOLO9NMSFree

model = LIBREYOLO9NMSFree(model_path = None, size="t")

results = model.train(
    data="coco.yaml",     # path to data.yaml (required)

    # Training parameters
    epochs=100,              # default: 100
    batch=32,
    imgsz=640,

    # Optimizer
    lr0=0.001,                # initial learning rate
    optimizer="AdamW",         # "SGD", "Adam", "AdamW"

    # System
    device="cuda",              # GPU device ("", "cpu", "cuda", "0", "0,1")
    workers=8,
    seed=0,

    # Output
    project="runs/train",
    name="libreyolonmsfree",
    exist_ok=False,

    # Training features
    amp=True,                # automatic mixed precision
    patience=50,             # early stopping patience
    resume=False,            # resume from loaded checkpoint
)

print(f"Best mAP50-95: {results['best_mAP50_95']:.3f}")
print(f"Best checkpoint: {results['best_checkpoint']}")