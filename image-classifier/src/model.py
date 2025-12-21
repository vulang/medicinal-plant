try:
    import timm
    from timm.data import resolve_model_data_config
except ImportError:
    timm = None
    resolve_model_data_config = None

def _build_timm_model(model_name: str, num_classes: int, pretrained: bool):
    if timm is None:
        raise ImportError(f"timm is required for {model_name}. Install it with `pip install timm`.")
    if resolve_model_data_config is None:
        raise ImportError("timm>=0.9 is required for data config resolution.")
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    model.data_config = resolve_model_data_config(model)
    return model

def build_model(model_name: str, num_classes: int, pretrained: bool = True):
    if model_name == "convnextv2_base.fcmae_ft_in22k_in1k":
        return _build_timm_model(model_name, num_classes, pretrained)
    if model_name == "swin_base_patch4_window7_224.ms_in22k":
        return _build_timm_model(model_name, num_classes, pretrained)
    raise ValueError(f"Unknown model_name: {model_name}")
