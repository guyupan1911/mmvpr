from mmvpr.registry import MODELS

BACKBONES = MODELS

def build_backbone(cfg):
    """Build backbone"""
    return BACKBONES.build(cfg)