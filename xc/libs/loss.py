import torch
import torch.nn.functional as F
import numpy as np


class _Loss(torch.nn.Module):
    def __init__(self, reduction='mean', pad_ind=None):
        super(_Loss, self).__init__()
        self.reduction = reduction
        self.pad_ind = pad_ind

    def _reduce(self, loss):
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()

    def _mask_at_pad(self, loss):
        """
        Mask the loss at padding index, i.e., make it zero
        """
        if self.pad_ind is not None:
            loss[:, self.pad_ind] = 0.0
        return loss

    def _mask(self, loss, mask=None):
        """
        Mask the loss at padding index, i.e., make it zero
        * Mask should be a boolean array with 1 where loss needs
        to be considered.
        * it'll make it zero where value is 0
        """
        if mask is not None:
            mask = mask.to(loss.device)
            loss = loss * mask
        return loss


class BCEWithLogitsLoss(_Loss):
    r""" BCE loss (expects logits; numercial stable)
    This loss combines a `Sigmoid` layer and the `BCELoss` in one single
    class. This version is more numerically stable than using a plain `Sigmoid`
    followed by a `BCELoss` as, by combining the operations into one layer,
    we take advantage of the log-sum-exp trick for numerical stability.
    Arguments:
    ----------
    weight: torch.Tensor or None, optional (default=None))
        a manual rescaling weight given to the loss of each batch element.
        If given, has to be a Tensor of size batch_size
    reduction: string, optional (default='mean')
        Specifies the reduction to apply to the output:
        * 'none': no reduction will be applied
        * 'mean' or 'sum': mean or sum of loss terms
    pos_weight: torch.Tensor or None, optional (default=None)
        a weight of positive examples.
        it must be a vector with length equal to the number of classes.
    pad_ind: int/int64 or None (default=None)
        ignore loss values at this index
        useful when some index has to be used as padding index
    """
    __constants__ = ['weight', 'pos_weight', 'reduction']

    def __init__(self, weight=None, reduction='mean',
                 pos_weight=None, pad_ind=None):
        super(BCEWithLogitsLoss, self).__init__(reduction, pad_ind)
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, input, target, mask=None):
        """
        Arguments:
        ---------
        input: torch.FloatTensor
            real number pred matrix of size: batch_size x output_size
            typically logits from the neural network
        target:  torch.FloatTensor
            0/1 ground truth matrix of size: batch_size x output_size
        mask: torch.BoolTensor or None, optional (default=None)
            ignore entries [won't contribute to loss] where mask value is zero
        Returns:
        -------
        loss: torch.FloatTensor
            dimension is defined based on reduction
        """
        loss = F.binary_cross_entropy_with_logits(input, target,
                                                  self.weight,
                                                  pos_weight=self.pos_weight,
                                                  reduction='none')
        loss = self._mask_at_pad(loss)
        loss = self._mask(loss, mask)
        return self._reduce(loss).unsqueeze(0)


class CosineEmbeddingLoss(_Loss):
    r""" Cosine embedding loss (expects cosine similarity)
    Arguments:
    ----------
    reduction: string, optional (default='mean')
        Specifies the reduction to apply to the output:
        * 'none': no reduction will be applied
        * 'mean' or 'sum': mean or sum of loss terms
    pos_weight: float or None, optional (default=None)
        weight of loss with positive target
    """

    def __init__(self, margin=0.5, reduction='mean', pos_wt=1.0):
        super(CosineEmbeddingLoss, self).__init__(reduction=reduction)
        self.margin = margin
        self.pos_wt = pos_wt
        self.weight = None

    def forward(self, input, target, mask=None):
        """
        Arguments:
        ---------
        input: torch.FloatTensor
            real number pred matrix of size: batch_size x output_size
            cosine similarity b/w label and document
        target:  torch.FloatTensor
            0/1 ground truth matrix of size: batch_size x output_size
        mask: torch.BoolTensor or None, optional (default=None)
            ignore entries [won't contribute to loss] where mask value is zero
        Returns:
        -------
        loss: torch.FloatTensor
            dimension is defined based on reduction
        """
        target = target.to(input.device)
        loss = torch.where(target > 0, (1-input) * self.pos_wt,
                           F.relu(input - self.margin))
        loss = self._mask_at_pad(loss)
        loss = self._mask(loss, mask)
        return self._reduce(loss).unsqueeze(0)

    def extra_repr(self):
        return f"m={self.margin}, pos_wts={self.pos_wt}"


class CustomMarginLoss(_Loss):

    def __init__(self, margin=1, num_neg=3, num_pos=1,
                 mn_lim=-50.0, mx_lim=50.0, tau=0.1, reduction='mean'):
        super(CustomMarginLoss, self).__init__(reduction=reduction)
        self.tau = tau
        self.margin = margin
        self.num_neg = num_neg
        self.mn_lim = mn_lim
        self.mx_lim = mx_lim
        self.num_pos = num_pos

    def forward(self, sim_b, target, sim_p=None, mask=None):
        B = target.size(0)
        target = target.to(sim_b.device)
        if sim_p is None:
            MX_LIM = torch.full_like(sim_b, self.mx_lim)
            sim_p = sim_b.where(target == 1, MX_LIM)
            indices = sim_p.topk(largest=False, dim=1, k=self.num_pos)[1]
            sim_p = sim_p.gather(1, indices)
        MN_LIM = torch.full_like(sim_b, self.mn_lim)

        _, num_pos = sim_p.size()
        sim_p = sim_p.view(B, num_pos, 1)
        sim_m = MN_LIM.where(target == 1, sim_b)
        indices = sim_m.topk(largest=True, dim=1, k=self.num_neg)[1]
        sim_n = sim_b.gather(1, indices)
        sim_n = sim_n.unsqueeze(1).repeat_interleave(num_pos, dim=1)

        loss = F.relu(sim_n - sim_p + self.margin)
        prob = torch.softmax(sim_n/self.tau, dim=-1)
        loss = loss * prob
        loss = self._reduce(loss)
        return loss

    def extra_repr(self):
        return f"m={self.margin}, num_neg={self.num_neg}, num_pos={self.num_pos}"
    

class TripletMarginLossOHNM(_Loss):
    r""" Triplet Margin Loss with Online Hard Negative Mining

    * Applies loss using the hardest negative in the mini-batch
    * Assumes diagonal entries are ground truth (for multi-class as of now)

    Arguments:
    ----------
    reduction: string, optional (default='mean')
        Specifies the reduction to apply to the output:
        * 'none': no reduction will be applied
        * 'mean' or 'sum': mean or sum of loss terms
    margin: float, optional (default=0.8)
        margin in triplet margin loss
    k: int, optional (default=2)
        compute loss only for top-k negatives in each row 
    apply_softmax: boolean, optional (default=2)
        promotes hard negatives using softmax
    """

    def __init__(self, margin=0.8, k=3, apply_softmax=True,
                 tau=0.1, num_violators=False, reduction='mean'):
        super(TripletMarginLossOHNM, self).__init__(reduction=reduction)
        self.margin = margin
        self.k = k
        self.tau = tau
        self.num_violators = num_violators
        self.apply_softmax = apply_softmax

    def forward(self, input, target, mask=None):
        """
        Arguments:
        ---------
        input: torch.FloatTensor
            real number pred matrix of size: batch_size x output_size
            cosine similarity b/w label and document
        target:  torch.FloatTensor
            0/1 ground truth matrix of size: batch_size x output_size
        mask: torch.BoolTensor or None, optional (default=None)
            ignore entries [won't contribute to loss] where mask value is False

        Returns:
        -------
        loss: torch.FloatTensor
            dimension is defined based on reduction
        """
        sim_p = torch.diagonal(input).view(-1, 1)
        similarities = torch.where(target == 0, input, torch.full_like(input, -50))
        _, indices = torch.topk(similarities, largest=True, dim=1, k=self.k)
        sim_n = input.gather(1, indices)
        loss = torch.max(torch.zeros_like(sim_p), sim_n - sim_p + self.margin)
        if self.apply_softmax:
            sim_n[loss == 0] = -50
            prob = torch.softmax(sim_n/self.tau, dim=1)
            loss = loss * prob
        reduced_loss = self._reduce(loss)
        if self.num_violators:
            nnz = torch.sum((loss > 0), axis=1).float().mean()
            return reduced_loss, nnz
        else:
            return reduced_loss

