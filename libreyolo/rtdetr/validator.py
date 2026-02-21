from libreyolo.validation import DetectionValidator, ValidationConfig

class RTDETRValidator(DetectionValidator):
    """
    RTDETR Validator that builds upon DetectionValidator but handles
    specific metric tracking or logging nuances if necessary.
    Uses generic DetectionValidator underneath heavily relying on model's _postprocess.
    """
    def __init__(self, model, config: ValidationConfig, **kwargs):
        super().__init__(model=model, config=config, **kwargs)
