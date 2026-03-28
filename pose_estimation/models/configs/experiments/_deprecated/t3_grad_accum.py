"""T3: Gradient accumulation (effective batch=128) on curriculum S2."""
_base_ = ["../../experiments/curriculum/humanart_curriculum_s2.py"]

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=1e-4, weight_decay=0.05),
    accumulative_counts=4,
)
