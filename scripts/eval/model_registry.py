"""Central registry of all models and their checkpoints.

Add new models here. All evaluation scripts reference this registry
instead of maintaining their own hardcoded model lists.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "pose_estimation" / "models" / "configs"
TRAIN_DIR = PROJECT_ROOT / "experiments" / "train"

# Public pretrained models (no Bizarre Pose fine-tuning)
PRETRAINED = {
    "RTMPose-m COCO": (
        CONFIGS_DIR / "experiments/stages/rtmpose_m_stage_a.py",
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth",
    ),
    "RTMPose-l COCO": (
        CONFIGS_DIR / "models/rtmpose_l_humanart.py",
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.pth",
    ),
    "RTMPose-m HumanArt": (
        CONFIGS_DIR / "experiments/stages/rtmpose_m_stage_a.py",
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_8xb256-420e_humanart-256x192-8430627b_20230611.pth",
    ),
    "RTMPose-l HumanArt": (
        CONFIGS_DIR / "models/rtmpose_l_humanart.py",
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_8xb256-420e_humanart-256x192-389f2cb0_20230611.pth",
    ),
}

# Our fine-tuned models
FINETUNED = {
    "Stage A": (
        CONFIGS_DIR / "experiments/stages/rtmpose_m_stage_a.py",
        TRAIN_DIR / "rtmpose_m_stage_a/best_coco_AP_epoch_10.pth",
    ),
    "HumanArt->BP": (
        CONFIGS_DIR / "models/rtmpose_m_humanart_pretrained.py",
        TRAIN_DIR / "rtmpose_m_humanart_finetune/best_coco_AP_epoch_10.pth",
    ),
    "Curriculum S2": (
        CONFIGS_DIR / "experiments/curriculum/humanart_curriculum_s2.py",
        TRAIN_DIR / "curriculum_s2/best_coco_AP_epoch_10.pth",
    ),
    "3L+mask": (
        CONFIGS_DIR / "experiments/occluded_reasoning/occluded_3layer.py",
        TRAIN_DIR / "techniques/occluded_3layer/best_coco_AP_epoch_10.pth",
    ),
    "3L curriculum": (
        CONFIGS_DIR / "experiments/occluded_reasoning/occ3l_curriculum_s2.py",
        TRAIN_DIR / "techniques/occ3l_curriculum_s2/best_coco_AP_epoch_8.pth",
    ),
    "3L soft-attn": (
        CONFIGS_DIR / "experiments/occluded_reasoning/soft_attn_curriculum_s2.py",
        TRAIN_DIR / "techniques/soft_attn_s2/best_coco_AP_epoch_10.pth",
    ),
    "P1 crop-aug": (
        CONFIGS_DIR / "experiments/occluded_reasoning/p1_crop_augment.py",
        TRAIN_DIR / "presence/p1_crop_augment/best_coco_AP_epoch_6.pth",
    ),
}

# All models combined
ALL_MODELS = {**PRETRAINED, **FINETUNED}


def get_model(name):
    """Get (config_path, checkpoint_path) for a model name."""
    if name in ALL_MODELS:
        cfg, ckpt = ALL_MODELS[name]
        return str(cfg), str(ckpt)
    raise KeyError(f"Unknown model: {name}. Available: {list(ALL_MODELS.keys())}")


def list_models():
    """List all available model names."""
    print("Pretrained:")
    for name in PRETRAINED:
        print(f"  {name}")
    print("Fine-tuned:")
    for name in FINETUNED:
        print(f"  {name}")
