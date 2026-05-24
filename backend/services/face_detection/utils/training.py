# utils/training.py
"""
Training utilities.
===================

Epoch runner, logging, timing, and accuracy helpers for fine-tuning
InceptionResnetV1 on custom identity datasets.

Changes from legacy version
----------------------------
    - AMP (Automatic Mixed Precision) support via torch.cuda.amp.GradScaler.
      Enable with use_amp=True in pass_epoch(). Gives ~1.5-2x speedup on
      Ampere+ GPUs with negligible accuracy loss.

    - optimizer.zero_grad(set_to_none=True) replaces zero_grad().
      Setting gradients to None instead of zero saves memory and is slightly
      faster — the gradient tensor is deallocated rather than zeroed in-place.

    - Scheduler step moved to end of epoch (not end of batch), matching
      PyTorch's documented usage for LR schedulers.

    - EpochResult dataclass replaces bare tuple return, making calling code
      readable without magic index unpacking.

    - Type hints throughout.

    - writer.iteration initialization removed from this module — callers
      should initialize SummaryWriter state on their own writers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class EpochResult:
    """
    Return value from pass_epoch().

    Attributes
    ----------
    loss : torch.Tensor
        Mean loss over the epoch.

    metrics : dict[str, torch.Tensor]
        Mean of each tracked metric over the epoch.

    mode : str
        'Train' or 'Valid'.
    """
    loss: torch.Tensor
    metrics: Dict[str, torch.Tensor]
    mode: str

    def __repr__(self) -> str:
        metric_str = ', '.join(f'{k}={v:.4f}' for k, v in self.metrics.items())
        return f"EpochResult(mode={self.mode}, loss={self.loss:.4f}, {metric_str})"


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class Logger:
    """
    In-place batch progress printer.

    Writes a single updating line to stdout: mode, batch index, loss,
    and any additional metrics. Newline is printed at the end of the epoch.

    Parameters
    ----------
    mode : str
        'Train' or 'Valid'.
    length : int
        Total number of batches in the epoch.
    calculate_mean : bool
        If True, display running cumulative mean rather than per-batch values.
    """

    def __init__(
        self,
        mode: str,
        length: int,
        calculate_mean: bool = False,
    ) -> None:
        self.mode = mode
        self.length = length
        self.calculate_mean = calculate_mean
        self._scale = (lambda x, i: x / (i + 1)) if calculate_mean else (lambda x, i: x)

    def __call__(
        self,
        loss: torch.Tensor,
        metrics: Dict[str, torch.Tensor],
        i: int,
    ) -> None:
        track = f'\r{self.mode} | {i + 1:5d}/{self.length:<5d}| '
        loss_str = f'loss: {self._scale(loss, i):9.4f} | '
        metric_str = ' | '.join(
            f'{k}: {self._scale(v, i):9.4f}' for k, v in metrics.items()
        )
        print(track + loss_str + metric_str + '   ', end='')
        if i + 1 == self.length:
            print('')


# ---------------------------------------------------------------------------
# Batch timer
# ---------------------------------------------------------------------------

class BatchTimer:
    """
    Measures throughput or latency per batch or per sample.

    Call with (y_pred, y) at the end of each batch to get the elapsed time
    or rate since the previous call (or since construction for batch 0).

    Parameters
    ----------
    rate : bool
        If True, report samples/sec or batches/sec.
        If False, report sec/sample or sec/batch.
    per_sample : bool
        If True, normalize by batch size.
    """

    def __init__(self, rate: bool = True, per_sample: bool = True) -> None:
        self._start = time.perf_counter()
        self.rate = rate
        self.per_sample = per_sample

    def __call__(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        elapsed = time.perf_counter() - self._start
        self._start = time.perf_counter()

        if self.per_sample:
            elapsed /= len(y_pred)
        if self.rate:
            elapsed = 1.0 / elapsed

        return torch.tensor(elapsed)


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

def accuracy(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Top-1 classification accuracy.

    Parameters
    ----------
    logits : torch.Tensor, shape (B, C)
    y : torch.Tensor, shape (B,)
        Ground-truth class indices.

    Returns
    -------
    torch.Tensor
        Scalar accuracy in [0, 1].
    """
    _, preds = torch.max(logits, dim=1)
    return (preds == y).float().mean()


# ---------------------------------------------------------------------------
# Epoch runner
# ---------------------------------------------------------------------------

def pass_epoch(
    model: nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    batch_metrics: Optional[Dict[str, Callable]] = None,
    show_running: bool = True,
    device: Union[str, torch.device] = 'cpu',
    writer: Optional[SummaryWriter] = None,
    use_amp: bool = False,
) -> EpochResult:
    """
    Train or evaluate the model over one full epoch.

    Pass optimizer=None for evaluation; the function detects train vs eval
    mode from model.training.

    Parameters
    ----------
    model : nn.Module
        The model to train or evaluate.
        Call model.train() or model.eval() before passing.

    loss_fn : callable
        Loss function: (y_pred, y) -> scalar tensor.

    loader : DataLoader
        Yields (x, y) batches.

    optimizer : Optimizer or None
        Required when model.training is True. Ignored during eval.

    scheduler : LR scheduler or None
        If provided, stepped once at the end of the epoch (not per batch).

    batch_metrics : dict[str, callable] or None
        Dictionary of metric functions, each taking (y_pred, y) and returning
        a scalar tensor. Defaults to {'time': BatchTimer()}.

    show_running : bool
        If True, print running cumulative means. If False, print per-batch values.

    device : str or torch.device
        Device to move batches to. Default 'cpu'.

    writer : SummaryWriter or None
        Optional TensorBoard writer. Scalars are logged every batch during
        training and once at epoch end during validation.

    use_amp : bool
        If True, use Automatic Mixed Precision (fp16 forward + fp32 grad update).
        Requires CUDA. Gives ~1.5-2x speedup on Ampere+ GPUs.
        Has no effect (silently disabled) on CPU or MPS.

    Returns
    -------
    EpochResult
        Contains mean epoch loss and mean metric values.

    Raises
    ------
    ValueError
        If model is in train mode but no optimizer is provided.
    """
    if model.training and optimizer is None:
        raise ValueError(
            "optimizer must be provided when model is in training mode. "
            "Pass optimizer=None only for evaluation."
        )

    if batch_metrics is None:
        batch_metrics = {'time': BatchTimer()}

    mode = 'Train' if model.training else 'Valid'
    device = torch.device(device)

    # AMP is only meaningful on CUDA; disable silently otherwise to avoid
    # unexpected behavior on CPU or MPS where autocast has no benefit.
    amp_enabled = use_amp and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    logger = Logger(mode, length=len(loader), calculate_mean=show_running)
    epoch_loss = torch.tensor(0.0)
    epoch_metrics: Dict[str, torch.Tensor] = {}

    for i_batch, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)

        # ------------------------------------------------------------------
        # Forward pass — wrapped in autocast for AMP when enabled.
        # autocast is a no-op context manager when amp_enabled=False.
        # ------------------------------------------------------------------
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            y_pred = model(x)
            loss_batch = loss_fn(y_pred, y)

        if model.training:
            # set_to_none=True deallocates gradient tensors between steps
            # rather than zeroing them — reduces memory and is slightly faster.
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss_batch).backward()
            scaler.step(optimizer)
            scaler.update()

        # ------------------------------------------------------------------
        # Metric accumulation
        # ------------------------------------------------------------------
        batch_metric_vals: Dict[str, torch.Tensor] = {}
        for name, fn in batch_metrics.items():
            val = fn(y_pred, y).detach().cpu()
            batch_metric_vals[name] = val
            epoch_metrics[name] = epoch_metrics.get(name, torch.tensor(0.0)) + val

        if writer is not None and model.training:
            if hasattr(writer, 'iteration') and hasattr(writer, 'interval'):
                if writer.iteration % writer.interval == 0:
                    writer.add_scalars('loss', {mode: loss_batch.detach().cpu()}, writer.iteration)
                    for name, val in batch_metric_vals.items():
                        writer.add_scalars(name, {mode: val}, writer.iteration)
                writer.iteration += 1

        loss_batch_cpu = loss_batch.detach().cpu()
        epoch_loss = epoch_loss + loss_batch_cpu

        if show_running:
            logger(epoch_loss, epoch_metrics, i_batch)
        else:
            logger(loss_batch_cpu, batch_metric_vals, i_batch)

    # ------------------------------------------------------------------
    # Scheduler step — once per epoch, after all batches.
    # Stepping per-batch is only correct for specific schedulers (e.g.
    # OneCycleLR); the default epoch-level step is correct for StepLR,
    # CosineAnnealingLR, ReduceLROnPlateau, etc.
    # ------------------------------------------------------------------
    if model.training and scheduler is not None:
        scheduler.step()

    n_batches = i_batch + 1
    epoch_loss = epoch_loss / n_batches
    epoch_metrics = {k: v / n_batches for k, v in epoch_metrics.items()}

    # Log epoch-level validation metrics to TensorBoard.
    if writer is not None and not model.training:
        if hasattr(writer, 'iteration'):
            writer.add_scalars('loss', {mode: epoch_loss.detach()}, writer.iteration)
            for name, val in epoch_metrics.items():
                writer.add_scalars(name, {mode: val}, writer.iteration)

    return EpochResult(loss=epoch_loss, metrics=epoch_metrics, mode=mode)


# ---------------------------------------------------------------------------
# DataLoader collation helpers
# ---------------------------------------------------------------------------

def collate_pil(batch):
    """
    Collate a batch of (PIL Image, label) pairs without stacking images.

    Use this as DataLoader(collate_fn=collate_pil) when the transforms
    pipeline operates on PIL Images and you want to defer tensor conversion.

    Returns
    -------
    tuple(list[PIL.Image], list)
        Images as a plain list (not stacked), labels as a list.
    """
    out_x, out_y = [], []
    for x, y in batch:
        out_x.append(x)
        out_y.append(y)
    return out_x, out_y
