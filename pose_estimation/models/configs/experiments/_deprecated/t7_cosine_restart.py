"""T7: Cosine annealing with warm restarts on curriculum S2."""
_base_ = ["../../experiments/curriculum/humanart_curriculum_s2.py"]

train_cfg = dict(max_epochs=20, val_interval=2, by_epoch=True)

param_scheduler = [
    dict(type="LinearLR", begin=0, end=2, start_factor=1e-5, by_epoch=True),
    dict(
        type="CosineRestartLR",
        periods=[6, 6, 6],
        restart_weights=[1, 0.7, 0.5],
        eta_min=1e-7,
        begin=2,
        end=20,
        by_epoch=True,
    ),
]
