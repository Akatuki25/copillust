"""3L GAU + soft attention masking, curriculum S2 (all 3200 images)."""
_base_ = ["./occ3l_curriculum_s2.py"]

load_from = "experiments/train/techniques/soft_attn_s1/best_coco_AP_epoch_10.pth"

model = dict(
    head=dict(
        soft_attention_mask=True,
        occ_attenuation=0.3,
    ),
)
