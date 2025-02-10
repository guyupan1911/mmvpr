from mmvpr.registry import MODELS

BACKBONES = MODELS
NECKS = MODELS
HEADS = MODELS
LOSSES = MODELS
CLASSIFIERS = MODELS
AGGREGATORS = MODELS

def build_backbone(cfg):
    """Build backbone"""
    return BACKBONES.build(cfg)

def build_neck(cfg):
    """Build neck"""
    return HEADS.build(cfg)

def build_head(cfg):
    """BUILD head"""
    return HEADS.build(cfg)

def build_aggregator(cfg):
    """Build aggregator"""
    return AGGREGATORS.build(cfg)

def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)

def build_classifier(cfg):
    """Build classifier"""
    return CLASSIFIERS.build(cfg)