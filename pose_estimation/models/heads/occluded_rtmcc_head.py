"""RTMCCHead with occlusion-aware reasoning (based on WACV 2024).

Extends RTMCCHead with:
1. VisNet: MLP that predicts per-keypoint visibility from backbone features
2. Masking: attenuates occluded keypoint features before GAU
3. The GAU then reasons from visible keypoints to infer occluded ones
4. Visibility BCE loss as auxiliary objective

Reference: "Rethinking Visibility in Human Pose Estimation:
Occluded Pose Reasoning via Transformers" (Sun et al., WACV 2024)
"""

import random
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from mmpose.codecs.utils import get_simcc_normalized
from mmpose.evaluation.functional import simcc_pck_accuracy
from mmpose.models.utils.rtmcc_block import RTMCCBlock, ScaleNorm
from mmpose.models.utils.tta import flip_vectors
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptSampleList)
from mmpose.models.heads.base_head import BaseHead

OptIntSeq = Optional[Sequence[int]]


@MODELS.register_module()
class OccludedRTMCCHead(BaseHead):
    """RTMCCHead with occlusion-aware visibility prediction and masking.

    Architecture:
        backbone features -> final_conv [B,K,H,W] -> flatten [B,K,HW]
                                |
                          VisNet (MLP) -> pred_vis [B,K]
                                |
                          Masking: visible*1.0, occluded*0.01
                                |
                          MLP -> [B,K,hidden]
                                |
                          GAU (self-attention with masked features)
                                |
                          SimCC cls_x, cls_y
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_size: Tuple[int, int],
        in_featuremap_size: Tuple[int, int],
        simcc_split_ratio: float = 2.0,
        final_layer_kernel_size: int = 7,
        gau_cfg: dict = dict(),
        loss: ConfigType = dict(type='KLDiscretLoss'),
        decoder: OptConfigType = None,
        init_cfg: OptConfigType = None,
        # Occlusion-specific params
        vis_loss_weight: float = 0.33,
        occ_attenuation: float = 0.01,
        mix_gt_prob: float = 0.5,
        num_gau_layers: int = 1,
        soft_attention_mask: bool = False,
    ):
        if init_cfg is None:
            init_cfg = [
                dict(type='Normal', layer=['Conv2d'], std=0.001),
                dict(type='Normal', layer=['Linear'], std=0.01),
            ]

        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.in_featuremap_size = in_featuremap_size
        self.simcc_split_ratio = simcc_split_ratio
        self.vis_loss_weight = vis_loss_weight
        self.occ_attenuation = occ_attenuation
        self.mix_gt_prob = mix_gt_prob
        self.soft_attention_mask = soft_attention_mask

        self.loss_module = MODELS.build(loss)
        if decoder is not None:
            self.decoder = KEYPOINT_CODECS.build(decoder)
        else:
            self.decoder = None

        flatten_dims = self.in_featuremap_size[0] * self.in_featuremap_size[1]

        # Final conv: backbone channels -> K keypoint channels
        self.final_layer = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=final_layer_kernel_size,
            stride=1,
            padding=final_layer_kernel_size // 2)

        # VisNet: predict per-keypoint visibility from spatial features
        self.visnet = nn.Sequential(
            nn.Linear(flatten_dims, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Visibility loss
        self.vis_criterion = nn.BCELoss(reduction='mean')

        # MLP: flatten -> hidden
        self.mlp = nn.Sequential(
            ScaleNorm(flatten_dims),
            nn.Linear(flatten_dims, gau_cfg['hidden_dims'], bias=False))

        W = int(self.input_size[0] * self.simcc_split_ratio)
        H = int(self.input_size[1] * self.simcc_split_ratio)

        # GAU (stacked layers for deeper inter-keypoint reasoning)
        self.gau_layers = nn.ModuleList([
            RTMCCBlock(
                self.out_channels,
                gau_cfg['hidden_dims'],
                gau_cfg['hidden_dims'],
                s=gau_cfg['s'],
                expansion_factor=gau_cfg['expansion_factor'],
                dropout_rate=gau_cfg['dropout_rate'],
                drop_path=gau_cfg['drop_path'],
                attn_type='self-attn',
                act_fn=gau_cfg['act_fn'],
                use_rel_bias=gau_cfg['use_rel_bias'],
                pos_enc=gau_cfg['pos_enc'])
            for _ in range(num_gau_layers)
        ])

        # SimCC classification heads
        self.cls_x = nn.Linear(gau_cfg['hidden_dims'], W, bias=False)
        self.cls_y = nn.Linear(gau_cfg['hidden_dims'], H, bias=False)

    def _apply_visibility_mask(self, feats, visibility):
        """Apply visibility mask to per-keypoint features (hard masking).

        Used when occ_attenuation < 1.0.
        """
        mask = visibility * (1.0 - self.occ_attenuation) + self.occ_attenuation
        return feats * mask.unsqueeze(-1)

    def _apply_soft_attention_mask(self, gau_layer, feats, visibility):
        """Run GAU with visibility-weighted attention (soft masking).

        Instead of attenuating features before GAU, we modify the attention
        kernel inside GAU: multiply each column of kernel[B,K,K] by the
        source keypoint's visibility. This means:
        - Attending FROM a visible keypoint: full weight
        - Attending FROM an occluded keypoint: reduced weight (occ_attenuation)
        - But the occluded keypoint's features are NOT destroyed

        This preserves the occluded keypoint's own features while reducing
        its influence on other keypoints' updates.
        """
        # Run GAU internals manually to inject visibility into attention
        x = gau_layer.ln(feats)
        uv = gau_layer.uv(x)
        uv = gau_layer.act_fn(uv)

        u, v, base = torch.split(uv, [gau_layer.e, gau_layer.e, gau_layer.s], dim=2)
        base = base.unsqueeze(2) * gau_layer.gamma[None, None, :] + gau_layer.beta
        q, k = torch.unbind(base, dim=2)

        qk = torch.bmm(q, k.permute(0, 2, 1))

        if gau_layer.use_rel_bias:
            bias = gau_layer.rel_pos_bias(q.size(1))
            qk += bias[:, :q.size(1), :k.size(1)]

        kernel = torch.square(torch.relu(qk / gau_layer.sqrt_s))  # [B, K, K]

        # Soft visibility weighting on attention kernel columns
        # visibility: (B, K) -> weight for each SOURCE keypoint
        vis_weight = visibility * (1.0 - self.occ_attenuation) + self.occ_attenuation  # (B, K)
        kernel = kernel * vis_weight.unsqueeze(1)  # [B, K, K] * [B, 1, K]

        if gau_layer.dropout_rate > 0.:
            kernel = gau_layer.dropout(kernel)

        x = u * torch.bmm(kernel, v)
        x = gau_layer.o(x)

        # Residual
        if gau_layer.shortcut:
            return gau_layer.res_scale(feats) + gau_layer.drop_path(x)
        else:
            return gau_layer.drop_path(x)

    def forward(self, feats: Tuple[Tensor], visibility: Tensor = None) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass.

        Args:
            feats: backbone features
            visibility: (B, K) optional visibility mask for training

        Returns:
            pred_x, pred_y: SimCC coordinate distributions
            pred_vis: (B, K) predicted visibility probabilities
        """
        feats = feats[-1]
        feats = self.final_layer(feats)  # B, K, H, W
        feats = torch.flatten(feats, 2)  # B, K, H*W

        # Predict visibility from spatial features
        pred_vis = self.visnet(feats).squeeze(-1)  # B, K

        # Determine which visibility to use for masking
        if visibility is not None:
            # Training: mix GT and predicted visibility
            if self.training and random.random() < self.mix_gt_prob:
                vis_mask = visibility
            else:
                vis_mask = (pred_vis.detach() >= 0.5).float()
        else:
            # Inference: use predicted visibility
            vis_mask = (pred_vis.detach() >= 0.5).float()

        if self.soft_attention_mask:
            # Soft masking: visibility applied inside GAU attention, not on features
            feats_hidden = self.mlp(feats)  # B, K, hidden (no feature masking)
            for gau in self.gau_layers:
                feats_hidden = self._apply_soft_attention_mask(gau, feats_hidden, vis_mask)
        else:
            # Hard masking: attenuate features before GAU (original approach)
            feats_masked = self._apply_visibility_mask(feats, vis_mask)
            feats_hidden = self.mlp(feats_masked)
            for gau in self.gau_layers:
                feats_hidden = gau(feats_hidden)

        pred_x = self.cls_x(feats_hidden)
        pred_y = self.cls_y(feats_hidden)

        return pred_x, pred_y, pred_vis

    def predict(
        self,
        feats: Tuple[Tensor],
        batch_data_samples: OptSampleList,
        test_cfg: OptConfigType = {},
    ) -> InstanceList:
        """Predict results from features."""

        if test_cfg.get('flip_test', False):
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            _feats, _feats_flip = feats

            _batch_pred_x, _batch_pred_y, _ = self.forward(_feats)
            _batch_pred_x_flip, _batch_pred_y_flip, _ = self.forward(_feats_flip)
            _batch_pred_x_flip, _batch_pred_y_flip = flip_vectors(
                _batch_pred_x_flip, _batch_pred_y_flip,
                flip_indices=flip_indices)

            batch_pred_x = (_batch_pred_x + _batch_pred_x_flip) * 0.5
            batch_pred_y = (_batch_pred_y + _batch_pred_y_flip) * 0.5
        else:
            batch_pred_x, batch_pred_y, _ = self.forward(feats)

        preds = self.decode((batch_pred_x, batch_pred_y))

        if test_cfg.get('output_heatmaps', False):
            batch_pred_x = get_simcc_normalized(batch_pred_x)
            batch_pred_y = get_simcc_normalized(batch_pred_y)
            B, K, _ = batch_pred_x.shape
            x = batch_pred_x.reshape(B, K, 1, -1)
            y = batch_pred_y.reshape(B, K, -1, 1)
            batch_heatmaps = torch.matmul(y, x)
            from mmengine.structures import PixelData
            pred_fields = [PixelData(heatmaps=hm) for hm in batch_heatmaps.detach()]
            for pred_instances, pred_x, pred_y in zip(
                    preds, to_numpy(batch_pred_x), to_numpy(batch_pred_y)):
                pred_instances.keypoint_x_labels = pred_x[None]
                pred_instances.keypoint_y_labels = pred_y[None]
            return preds, pred_fields
        else:
            return preds

    def loss(
        self,
        feats: Tuple[Tensor],
        batch_data_samples: OptSampleList,
        train_cfg: OptConfigType = {},
    ) -> dict:
        """Calculate losses."""

        # Get GT visibility for masking and vis loss
        gt_vis = torch.cat([
            d.gt_instance_labels.keypoint_weights
            for d in batch_data_samples
        ], dim=0)  # (B, K, 1) or (B, K)

        if gt_vis.dim() == 3:
            gt_vis = gt_vis.squeeze(-1)  # (B, K)

        # Convert to binary: v>=1 -> visible for masking, v==0 -> occluded
        # But for vis prediction target: v==2 -> visible(1), v==1 or v==0 -> occluded(0)
        vis_target = (gt_vis >= 2.0).float()  # only fully visible = 1
        vis_mask_for_gau = (gt_vis >= 1.0).float()  # occluded(v=1) still has position info

        pred_x, pred_y, pred_vis = self.forward(feats, visibility=vis_mask_for_gau)

        gt_x = torch.cat([
            d.gt_instance_labels.keypoint_x_labels for d in batch_data_samples
        ], dim=0)
        gt_y = torch.cat([
            d.gt_instance_labels.keypoint_y_labels for d in batch_data_samples
        ], dim=0)
        keypoint_weights = torch.cat([
            d.gt_instance_labels.keypoint_weights for d in batch_data_samples
        ], dim=0)

        pred_simcc = (pred_x, pred_y)
        gt_simcc = (gt_x, gt_y)

        losses = dict()

        # Keypoint loss
        loss_kpt = self.loss_module(pred_simcc, gt_simcc, keypoint_weights)
        losses.update(loss_kpt=loss_kpt)

        # Visibility loss
        loss_vis = self.vis_criterion(pred_vis, vis_target)
        losses.update(loss_vis=loss_vis * self.vis_loss_weight)

        # Accuracy
        _, avg_acc, _ = simcc_pck_accuracy(
            output=to_numpy(pred_simcc),
            target=to_numpy(gt_simcc),
            simcc_split_ratio=self.simcc_split_ratio,
            mask=to_numpy(keypoint_weights) > 0,
        )
        acc_pose = torch.tensor(avg_acc, device=gt_x.device, dtype=torch.float32)
        losses.update(acc_pose=acc_pose)

        return losses

    @property
    def default_init_cfg(self):
        return [
            dict(type='Normal', layer=['Conv2d'], std=0.001),
            dict(type='Normal', layer=['Linear'], std=0.01),
        ]
