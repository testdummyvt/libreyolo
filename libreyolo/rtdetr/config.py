"""
Training configuration for RT-DETR.

Provides a dataclass-based configuration with RT-DETR-specific defaults.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Tuple, Union
import yaml


@dataclass
class RTDETRTrainConfig:
    """
    Configuration for RT-DETR training.
    """

    # Model configuration
    size: str = "r18"  # "r18", "r34", "r50", "r101", "x"
    num_classes: int = 80
    pretrained: Optional[str] = None

    # Data configuration
    data: Optional[str] = None  # Path to data.yaml
    data_dir: Optional[str] = None  # Direct path to dataset
    imgsz: int = 640

    # Training parameters
    epochs: int = 72
    batch: int = 4
    device: str = "auto"  # "auto", "cuda", "mps", "cpu"

    # Optimizer settings
    optimizer: str = "adamw"  # "sgd", "adam", "adamw"
    lr0: float = 0.0001  # Base learning rate usually lower for DETR
    momentum: float = 0.9  
    weight_decay: float = 0.0001
    nesterov: bool = False

    # Learning rate schedule 
    scheduler: str = "linear"  # "linear", "cos", "warmcos"
    warmup_epochs: int = 0
    warmup_lr_start: float = 0.0
    no_aug_epochs: int = 0
    min_lr_ratio: float = 0.01

    # Augmentation settings 
    mosaic_prob: float = 0.5
    mixup_prob: float = 0.0
    hsv_prob: float = 0.1
    flip_prob: float = 0.5
    degrees: float = 0.0
    translate: float = 0.1
    mosaic_scale: Tuple[float, float] = (0.5, 1.5)
    mixup_scale: Tuple[float, float] = (0.5, 1.5)
    shear: float = 0.0

    # Training features
    ema: bool = True
    ema_decay: float = 0.9999
    amp: bool = True  # Automatic mixed precision

    # Checkpointing
    project: str = "runs/train"
    name: str = "rtdetr_train"
    exist_ok: bool = False
    save_period: int = 10  # Save checkpoint every N epochs
    eval_interval: int = 1  # Evaluate every N epochs

    # Workers
    workers: int = 4

    # Early stopping
    patience: int = 50

    # Resume
    resume: bool = False

    # Logging
    log_interval: int = 1  # Log every N iterations

    # Reproducibility
    seed: int = 0

    def __post_init__(self):
        """Validate configuration after initialization."""
        valid_sizes = ["r18", "r34", "r50", "r101", "r18-vd", "r34-vd", "r50-vd", "r101-vd", "x", "l"]
        if self.size not in valid_sizes:
            raise ValueError(f"Invalid size '{self.size}'. Must be one of {valid_sizes}")

        valid_schedulers = ["linear", "cos", "warmcos"]
        if self.scheduler not in valid_schedulers:
            raise ValueError(
                f"Invalid scheduler '{self.scheduler}'. Must be one of {valid_schedulers}"
            )

        valid_optimizers = ["sgd", "adam", "adamw"]
        if self.optimizer not in valid_optimizers:
            raise ValueError(
                f"Invalid optimizer '{self.optimizer}'. Must be one of {valid_optimizers}"
            )

        # Ensure scale tuples are tuples
        if isinstance(self.mosaic_scale, list):
            self.mosaic_scale = tuple(self.mosaic_scale)
        if isinstance(self.mixup_scale, list):
            self.mixup_scale = tuple(self.mixup_scale)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "RTDETRTrainConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        # Handle old field names for backward compatibility
        renames = {
            "batch_size": "batch",
            "num_workers": "workers",
            "save_dir": "project",
            "lr": "lr0",
            "save_interval": "save_period",
        }
        for old, new in renames.items():
            if old in data and new not in data:
                data[new] = data.pop(old)
        return cls(**data)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    def update(self, **kwargs) -> "RTDETRTrainConfig":
        """Create a new config with updated values."""
        data = asdict(self)
        data.update(kwargs)
        return RTDETRTrainConfig(**data)

    @property
    def input_size(self) -> Tuple[int, int]:
        """Return input size as (height, width) tuple."""
        return (self.imgsz, self.imgsz)

    @property
    def effective_lr(self) -> float:
        """Calculate effective learning rate based on batch size."""
        # DETR doesn't strictly linear scale but we can leave it for now
        return self.lr0 * self.batch / 16.0

    def __repr__(self) -> str:
        lines = ["RTDETRTrainConfig("]
        for key, value in asdict(self).items():
            lines.append(f"  {key}={value!r},")
        lines.append(")")
        return "\n".join(lines)


# Preset configurations for different model sizes
RTDETR_CONFIGS = {
    "r18": RTDETRTrainConfig(size="r18", imgsz=640),
    "r34": RTDETRTrainConfig(size="r34", imgsz=640),
    "r50": RTDETRTrainConfig(size="r50", imgsz=640),
    "r101": RTDETRTrainConfig(size="r101", imgsz=640),
    "x": RTDETRTrainConfig(size="x", imgsz=640),
}


def get_rtdetr_config(size: str = "r18", **kwargs) -> RTDETRTrainConfig:
    """
    Get a preset RTDETR configuration with optional overrides.

    Args:
        size: Model size ("r18", "r34", "r50", "r101", "x")
        **kwargs: Override any configuration parameter

    Returns:
        RTDETRTrainConfig instance
    """
    if size not in RTDETR_CONFIGS:
        raise ValueError(f"Unknown size '{size}'. Available: {list(RTDETR_CONFIGS.keys())}")

    config = RTDETR_CONFIGS[size]
    if kwargs:
        config = config.update(**kwargs)
    return config
