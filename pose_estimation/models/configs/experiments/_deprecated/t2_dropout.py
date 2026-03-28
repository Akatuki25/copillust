"""T2: Increased dropout in GAU on curriculum S2."""
_base_ = ["../../experiments/curriculum/humanart_curriculum_s2.py"]

model = dict(
    head=dict(
        gau_cfg=dict(dropout_rate=0.1, drop_path=0.1),
    ),
)
