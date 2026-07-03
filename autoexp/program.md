# autoresearch — Illustration Pose Estimation

This is an automated research loop for improving RTMPose-based illustration pose estimation.

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr8`). The branch `autoexp/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoexp/<tag>` from current main.
3. **Read the in-scope files** for full context:
   - `README.md` — current benchmark, confirmed findings, bottleneck analysis
   - `docs/model_modify.md` — RTMPose's broken assumptions for illustrations
   - `docs/model_modify_task.md` — priority order and past experiment results
   - `autoexp/evaluate.py` — fixed evaluation. Do not modify.
   - `pose_estimation/models/configs/experiments/curriculum/humanart_curriculum_s2.py` — baseline config (OKS@75=0.801)
   - `pose_estimation/models/heads/occluded_rtmcc_head.py` — custom head
   - `pose_estimation/codecs/visibility_aware_simcc.py` — custom codec
   - `pose_estimation/transforms/random_edges_blackout.py` — crop augmentation
   - `autoexp/hypotheses/bottleneck1_visibility.md` — hypothesis space for bottleneck 1
   - `autoexp/hypotheses/bottleneck2_relation.md` — hypothesis space for bottleneck 2
   - `autoexp/hypotheses/bottleneck3_presence.md` — hypothesis space for bottleneck 3
4. **Initialize results.tsv**: Create `autoexp/results.tsv` with just the header row. Leave it untracked by git.
5. **Confirm and go**.

## Experimentation

Each experiment trains for a **fixed budget of 10 epochs** (consistent with all past experiments in this project).

**What you CAN do:**
- Create or modify MMPose config files under `pose_estimation/models/configs/experiments/autoexp/`
- Modify existing custom Python modules: `pose_estimation/models/heads/*.py`, `pose_estimation/codecs/*.py`, `pose_estimation/transforms/*.py`
- Create new Python modules in those same directories
- Each experiment is one focused change. Compound changes are not allowed.

**What you CANNOT do:**
- Modify `autoexp/evaluate.py`. It is read-only. It is the ground truth metric.
- Modify `__init__.py` files in any directory — they are not needed and touching them risks breaking imports.
- Modify base configs under `pose_estimation/models/configs/base/`
- Modify the baseline config `humanart_curriculum_s2.py`
- Change training data (always use `data/merged_500_corrected/`, the 3200-image corrected set)
- Change the backbone (CSPNeXt), resolution (256×192), or base pretrained weights
- Install new packages
- Add Amateur Drawings data (known to degrade performance)
- Swap backbone to HRNet/UniFormer/ViTPose (already confirmed non-effective)

**Every new config must:**
- Set `_base_ = ["../../base/base_bizarre_pose.py"]` (or extend from an existing autoexp config via relative path)
- Set `load_from = "experiments/train/curriculum_s2/best_coco_AP_epoch_10.pth"`
- Set `data_root = "data/merged_500_corrected/"`

**Registering custom Python modules in a config:**

MMPose uses `custom_imports` to load non-MMPose Python modules. If your config uses a custom class (head, codec, or transform), you MUST declare it:

```python
# For a new or modified existing module:
custom_imports = dict(
    imports=["pose_estimation.models.heads.my_new_head"],
    allow_failed_imports=False,
)
```

For multiple custom modules:
```python
custom_imports = dict(
    imports=[
        "pose_estimation.models.heads.my_new_head",
        "pose_estimation.codecs.visibility_aware_simcc",
    ],
    allow_failed_imports=False,
)
```

The `@MODELS.register_module()` (or `@KEYPOINT_CODECS.register_module()`) decorator in the Python file does the actual registration when the module is imported. Without `custom_imports`, MMPose will raise `KeyError: 'MyClass is not in the models registry'`.

Existing custom classes and their import paths:
- `OccludedRTMCCHead` → `pose_estimation.models.heads.occluded_rtmcc_head`
- `VisibilityAwareSimCC` → `pose_estimation.codecs.visibility_aware_simcc`
- `VisibilityWeightControl` → `pose_estimation.transforms.visibility_weight_control`
- `RandomEdgesBlackout` → `pose_estimation.transforms.random_edges_blackout`

**The goal: maximize `bp_oks75`.**
Baseline (Curriculum S2): **0.801**. Paper best (Chen & Zwicker, WACV 2022): 0.793. We already beat it.

**Simplicity criterion**: All else being equal, simpler is better. A small gain that adds ugly complexity is not worth it. Removing something and getting equal or better results is a great outcome.

**The first run**: Establish the baseline by evaluating the existing Curriculum S2 checkpoint without any modification.

## Training a new config

```bash
python -m pose_estimation.training.trainer \
    --config pose_estimation/models/configs/experiments/autoexp/<name>.py \
    --work-dir experiments/train/autoexp/<name> \
    --device auto > train.log 2>&1
```

Wait for completion. Find the best checkpoint:

```bash
ls experiments/train/autoexp/<name>/best_coco_AP_epoch_*.pth | tail -1
```

## Evaluating

```bash
python -m autoexp.evaluate \
    --config pose_estimation/models/configs/experiments/autoexp/<name>.py \
    --checkpoint experiments/train/autoexp/<name>/best_coco_AP_epoch_<N>.pth \
    > eval.log 2>&1
grep "OKS@75\|OKS@50\|mydata\|Delta" eval.log
```

For the baseline evaluation (no training needed):

```bash
python -m autoexp.evaluate \
    --config pose_estimation/models/configs/experiments/curriculum/humanart_curriculum_s2.py \
    --checkpoint experiments/train/curriculum_s2/best_coco_AP_epoch_10.pth \
    > eval.log 2>&1
```

## Output format

`autoexp/evaluate.py` prints:

```
Evaluation Results
==================================================
Bizarre Pose test (487 images):
  OKS@50: 0.892
  OKS@75: 0.801  ← primary metric (baseline: 0.801)

mydata sanity (18 images):
  OKS@50: 0.xxx
  OKS@75: 0.xxx

Delta vs baseline: +0.000
```

Extract the key metric:
```bash
grep "OKS@75:" eval.log
```

## Logging results

Log every experiment to `autoexp/results.tsv` (tab-separated, NOT comma-separated).

Header and columns:

```
commit	bp_oks75	bp_oks50	mydata_oks75	status	description
```

1. git commit hash (short, 7 chars)
2. bp_oks75 achieved — use 0.000000 for crashes
3. bp_oks50 achieved — use 0.000000 for crashes
4. mydata_oks75 — use 0.000000 if sanity failed or crash
5. status: `keep`, `discard`, or `crash`
6. short description of what this experiment tried

Example:
```
commit	bp_oks75	bp_oks50	mydata_oks75	status	description
a1b2c3d	0.801000	0.892000	0.612000	keep	baseline (curriculum_s2)
b2c3d4e	0.805000	0.893000	0.618000	keep	v1_weight=0.5 in VisibilityAwareSimCC
c3d4e5f	0.798000	0.890000	0.600000	discard	gau hidden_dims=128 (worse)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoexp/apr8`).

LOOP FOREVER:

1. Look at the git state: current branch and commit
2. Decide the next experiment based on: hypothesis files, past results in results.tsv, domain knowledge from docs/
3. Write or modify the config and/or custom Python module
4. `git commit`
5. Run training: `python -m pose_estimation.training.trainer ... > train.log 2>&1`
6. If training crashes: `tail -50 train.log` to diagnose. Fix if trivial, otherwise skip and revert.
7. Find checkpoint: `ls experiments/train/autoexp/<name>/best_coco_AP_epoch_*.pth | tail -1`
8. Run evaluation: `python -m autoexp.evaluate ... > eval.log 2>&1`
9. Read result: `grep "OKS@75\|OKS@50\|mydata\|Delta" eval.log`
10. Log to results.tsv
11. If bp_oks75 improved AND mydata sanity passed: keep the commit (advance the branch)
12. If bp_oks75 did not improve OR mydata sanity failed: `git reset --hard HEAD~1`, delete the work dir

**Timeout**: Each experiment (training) takes approximately 5-15 minutes depending on device. If it runs longer than 30 minutes, kill it and treat as failure.

**Crashes**: Fix trivial bugs and retry. If the idea is fundamentally broken, log "crash" and move on.

**NEVER STOP**: Once the loop has begun, do NOT pause to ask the user if you should continue. You are autonomous. If you run out of ideas, consult the hypothesis files, re-read docs/model_modify.md, try combining near-misses, try more radical changes within the allowed scope. The loop runs until the user interrupts you.

## Bottleneck priority

Read the hypothesis files for the full hypothesis space. Priority order:

1. **Bottleneck 1 (visibility supervision)** — `autoexp/hypotheses/bottleneck1_visibility.md`
   - Confirmed culprit. Most promising first: H1.1 (v1_weight=0.5), then H1.3 (soft-gated attention)
2. **Bottleneck 2 (GAU relation prior)** — `autoexp/hypotheses/bottleneck2_relation.md`
   - H2.1 (reduced hidden_dims), H2.2 (dropout)
3. **Bottleneck 3 (tail failure / presence)** — `autoexp/hypotheses/bottleneck3_presence.md`
   - H3.3 (crop aug tuning), then H3.1 (presence head)
