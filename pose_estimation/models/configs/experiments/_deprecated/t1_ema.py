"""T1: EMA (Exponential Moving Average) on curriculum S2."""
_base_ = ["../../experiments/curriculum/humanart_curriculum_s2.py"]

custom_hooks = [
    dict(type="SyncBuffersHook"),
    dict(type="EMAHook", ema_type="ExpMomentumEMA", momentum=0.0002, update_buffers=True, priority=49),
]
