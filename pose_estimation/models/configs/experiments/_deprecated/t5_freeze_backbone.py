"""T5: Freeze backbone (lr_mult=0) on curriculum S2."""
_base_ = ["../../experiments/curriculum/humanart_curriculum_s2.py"]

optim_wrapper = dict(
    optimizer=dict(type="AdamW", lr=1e-4, weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={"backbone": dict(lr_mult=0.0)},
    ),
)
