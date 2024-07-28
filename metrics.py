import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class Metrics(nn.Module):
    def __init__(self, threshold=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold

    def forward(self, pred: torch.tensor, target: torch.tensor):
        tp, fp, fn, tn = smp.metrics.get_stats(pred, target, mode='binary', threshold=self.threshold)
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="none")
        precision = smp.metrics.precision(tp, fp, fn, tn, reduction="none")
        recall = smp.metrics.recall(tp, fp, fn, tn, reduction="none")
        f1score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="none")

        return {
            "batch_size": pred.shape[0],
            "background": {
                "iou": iou[:, 0].sum().item(),
                "precision": precision[:, 0].sum().item(),
                "recall": recall[:, 0].sum().item(),
                "f1score": f1score[:, 0].sum().item(),
            },
            "building": {
                "iou": iou[:, 1].sum().item(),
                "precision": precision[:, 1].sum().item(),
                "recall": recall[:, 1].sum().item(),
                "f1score": f1score[:, 1].sum().item(),
            }
        }
