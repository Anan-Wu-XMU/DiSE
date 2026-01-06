import torch
from torchmetrics import Metric, MetricCollection
from torch import Tensor
import wandb
import torch.nn as nn


class CEPerClass(Metric):
    full_state_update = False
    def __init__(self, class_id):
        super().__init__()
        self.class_id = class_id
        self.add_state('total_ce', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.softmax = torch.nn.Softmax(dim=-1)
        self.binary_cross_entropy = torch.nn.BCELoss(reduction='sum')

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets.
        Args:
            preds: Predictions from model   (batch, nodes, num_features) or (batch, nodes, nodes, num_features)
            target: Ground truth values     (batch, nodes, num_features) or (batch, nodes, nodes, num_features)
        """
        target = target.reshape(-1, target.shape[-1])  # [batch * nodes, num_fea] or [batch * nodes * nodes, num_fea]

        mask = (target != 0.).any(dim=-1)  # to mash the values that all are zero in the target.shape[-1]

        prob = self.softmax(preds)[..., self.class_id]
        # 1 get softmax of the predictions
        # 2 [batch, nodes, 1] or [batch, nodes, nodes, 1], 1: the class_id of the dimensions

        prob = prob.flatten()[mask]

        target = target[:, self.class_id]
        target = target[mask]

        output = self.binary_cross_entropy(prob, target)
        self.total_ce += output
        self.total_samples += prob.numel()  # numel: number of elements, batch * nodes or batch * nodes * nodes

    def compute(self):
        return self.total_ce / self.total_samples


class NoBondCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SingleCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class DoubleCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class TripleCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class AromaticCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)

class SingleAromaticCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)

class BondMetricsCE(MetricCollection):
    def __init__(self):
        ce_no_bond = NoBondCE(0)
        ce_SI = SingleCE(1)
        ce_DO = DoubleCE(2)
        ce_TR = TripleCE(3)
        ce_AR = AromaticCE(4)
        super().__init__([ce_no_bond, ce_SI, ce_DO, ce_TR, ce_AR])


class BondMetricsCE_pcq(MetricCollection):
    def __init__(self):
        ce_no_bond = NoBondCE(0)
        ce_SI = SingleCE(1)
        ce_DO = DoubleCE(2)
        ce_TR = TripleCE(3)
        ce_AR = AromaticCE(4)
        ce_SIAR = SingleAromaticCE(5)
        super().__init__([ce_no_bond, ce_SI, ce_DO, ce_TR, ce_AR, ce_SIAR])





class TrainMolecularMetricsDiscrete(nn.Module):
    def __init__(self):
        super().__init__()
        self.train_bond_metrics = BondMetricsCE()

    def forward(self, masked_pred_E, true_E, log: bool):
        # masked_pred_X: [Batch, Max_num_nodes, num_atom_features]
        # masked_pred_E: [Batch, Max_num_nodes, Max_num_nodes, num_edge_features]
        # true_X: [Batch, Max_num_nodes, num_atom_features]
        # true_E: [Batch, Max_num_nodes, Max_num_nodes, num_edge_features]

        self.train_bond_metrics(masked_pred_E, true_E)

        if log:
            to_log = {}
            for key, val in self.train_bond_metrics.compute().items():
                to_log['train/' + key] = val.item()
            if wandb.run:
                wandb.log(to_log, commit=False)

    def reset(self):
        for metric in [self.train_bond_metrics]:
            metric.reset()

    def log_epoch_metrics(self):
        epoch_bond_metrics = self.train_bond_metrics.compute()

        to_log = {}
        for key, val in epoch_bond_metrics.items():
            to_log['train_epoch/' + key] = val.item()
        if wandb.run:
            wandb.log(to_log, commit=False)

        for key, val in epoch_bond_metrics.items():
            epoch_bond_metrics[key] = val.item()

        return epoch_bond_metrics



class TrainMolecularMetricsDiscrete_pcq(nn.Module):
    def __init__(self):
        super().__init__()
        self.train_bond_metrics = BondMetricsCE_pcq()

    def forward(self, masked_pred_E, true_E, log: bool):
        # masked_pred_X: [Batch, Max_num_nodes, num_atom_features]
        # masked_pred_E: [Batch, Max_num_nodes, Max_num_nodes, num_edge_features]
        # true_X: [Batch, Max_num_nodes, num_atom_features]
        # true_E: [Batch, Max_num_nodes, Max_num_nodes, num_edge_features]

        self.train_bond_metrics(masked_pred_E, true_E)

        if log:
            to_log = {}
            for key, val in self.train_bond_metrics.compute().items():
                to_log['train/' + key] = val.item()
            if wandb.run:
                wandb.log(to_log, commit=False)

    def reset(self):
        for metric in [self.train_bond_metrics]:
            metric.reset()

    def log_epoch_metrics(self):
        epoch_bond_metrics = self.train_bond_metrics.compute()

        to_log = {}
        for key, val in epoch_bond_metrics.items():
            to_log['train_epoch/' + key] = val.item()
        if wandb.run:
            wandb.log(to_log, commit=False)

        for key, val in epoch_bond_metrics.items():
            epoch_bond_metrics[key] = val.item()

        return epoch_bond_metrics
