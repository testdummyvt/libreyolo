from libreyolo.validation import DetectionValidator, ValidationConfig


class DABDETRValidator(DetectionValidator):
    """DAB-DETR validator — thin subclass of DetectionValidator.

    All detection logic is handled by the model's _postprocess method.
    """

    def __init__(self, model, config: ValidationConfig, **kwargs):
        super().__init__(model=model, config=config, **kwargs)
