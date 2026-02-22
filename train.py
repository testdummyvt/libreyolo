from libreyolo import LIBREYOLORTDETR

model = LIBREYOLORTDETR(model_path = None, size="r18")

results = model.train(
    data="coco.yaml",     # path to data.yaml (required)

    # Training parameters
    epochs=50,              # default: 100
    batch=24,
    imgsz=640,

    # Optimizer
    lr0=0.0001,                # initial learning rate
    optimizer="AdamW",         # "SGD", "Adam", "AdamW"

    # System
    device="cuda",              # GPU device ("", "cpu", "cuda", "0", "0,1")
    workers=8,
    seed=0,

    # Output
    project="runs/train",
    name="rtdetr_train",
    exist_ok=False,

    # Training features
    amp=False,                # automatic mixed precision
    patience=50,             # early stopping patience
    resume=False,            # resume from loaded checkpoint
    log_interval = 1,
    eval_interval = 5
)

print(f"Best mAP50-95: {results['best_mAP50_95']:.3f}")
print(f"Best checkpoint: {results['best_checkpoint']}")
