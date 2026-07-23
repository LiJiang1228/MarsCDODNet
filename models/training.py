

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from .convgru import ConvGRUModel
from .convlstm import ConvLSTMModel
from .convlstm_s2s import ConvLSTMS2SModel
from .marscdodnet import AttentionResidualModel, MarsCDODTerrainFiLMModel
from .predrnn import PredRNNModel
from .swinlstm import SwinLSTMModel


MODEL_TYPES = (
    "convlstm",
    "convlstm_s2s",
    "convgru",
    "predrnn",
    "swinlstm",
    "attention_residual",
    "marscdodnet",
)


class GradientDifferenceLoss(nn.Module):


    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction_dx = torch.abs(prediction[..., :, 1:] - prediction[..., :, :-1])
        target_dx = torch.abs(target[..., :, 1:] - target[..., :, :-1])
        prediction_dy = torch.abs(prediction[..., 1:, :] - prediction[..., :-1, :])
        target_dy = torch.abs(target[..., 1:, :] - target[..., :-1, :])
        loss_x = torch.mean(torch.abs(prediction_dx - target_dx) ** self.alpha)
        loss_y = torch.mean(torch.abs(prediction_dy - target_dy) ** self.alpha)
        return loss_x + loss_y


class CompositeLoss(nn.Module):


    def __init__(self, gradient_weight: float = 0.0) -> None:
        super().__init__()
        self.gradient_weight = gradient_weight
        self.mse = nn.MSELoss()
        self.gradient = GradientDifferenceLoss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.mse(prediction, target) + self.gradient_weight * self.gradient(
            prediction, target
        )


def parse_indices(value: str | None, channel_count: int) -> list[int]:

    if value is None:
        return list(range(channel_count))
    if value.strip().lower() in ("", "none"):
        return []
    indices = [int(part.strip()) for part in value.split(",") if part.strip()]
    if len(set(indices)) != len(indices):
        raise ValueError(f"Repeated channel indices: {indices}")
    invalid = [index for index in indices if not 0 <= index < channel_count]
    if invalid:
        raise ValueError(f"Channel indices outside [0,{channel_count - 1}]: {invalid}")
    return indices


def parse_hidden_dims(value: str) -> list[int]:
    dims = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not dims or any(dim <= 0 for dim in dims):
        raise ValueError("--hidden-dims must contain positive integers")
    return dims


def get_device(requested: str) -> torch.device:

    if requested != "auto":
        return torch.device(requested)
    try:
        import torch_npu  # noqa: F401

        if torch.npu.is_available():
            return torch.device("npu:0")
    except (ImportError, AttributeError):
        pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MarsSequenceDataset(Dataset):


    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        input_steps: int,
        input_channel_indices: list[int],
        cdod_raw_channel: int,
        decoder_aux_indices: list[int],
    ) -> None:
        if x.ndim != 5 or y.ndim != 5:
            raise ValueError(f"Expected X/y with five dimensions, got {x.shape} and {y.shape}")
        if x.shape[0] != y.shape[0] or x.shape[2:4] != y.shape[2:4]:
            raise ValueError("X and y sample/spatial dimensions do not match")
        if not input_channel_indices:
            raise ValueError("At least one encoder input channel is required")
        if cdod_raw_channel not in input_channel_indices:
            raise ValueError("The CDOD channel must be included in encoder input channels")
        if not 0 < input_steps <= x.shape[1]:
            raise ValueError(f"input_steps={input_steps} is invalid for X with {x.shape[1]} time steps")

        self.x = x.astype(np.float32, copy=False)
        self.y = y.astype(np.float32, copy=False)
        self.input_steps = input_steps
        self.output_steps = y.shape[1]
        self.input_channel_indices = input_channel_indices
        self.decoder_aux_indices = decoder_aux_indices
        self.cdod_input_position = input_channel_indices.index(cdod_raw_channel)
        self.future_aux_source = (
            "future_x" if decoder_aux_indices and x.shape[1] >= input_steps + self.output_steps
            else "repeat_last_x" if decoder_aux_indices
            else "none"
        )

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int):
        x = np.take(self.x[index, : self.input_steps], self.input_channel_indices, axis=-1)
        target = self.y[index, :, :, :, :1]
        if self.future_aux_source == "future_x":
            aux = np.take(
                self.x[index, self.input_steps : self.input_steps + self.output_steps],
                self.decoder_aux_indices,
                axis=-1,
            )
        elif self.future_aux_source == "repeat_last_x":
            last = np.take(self.x[index, self.input_steps - 1], self.decoder_aux_indices, axis=-1)
            aux = np.repeat(last[None], self.output_steps, axis=0)
        else:
            aux = np.empty((*target.shape[:1], *target.shape[1:3], 0), dtype=np.float32)
        x = np.ascontiguousarray(x.transpose(0, 3, 1, 2))
        aux = np.ascontiguousarray(aux.transpose(0, 3, 1, 2))
        target = np.ascontiguousarray(target.transpose(0, 3, 1, 2))
        return torch.from_numpy(x), torch.from_numpy(aux), torch.from_numpy(target)


def load_static_features(path: Path) -> np.ndarray:

    static = np.load(path, allow_pickle=False)
    if isinstance(static, np.lib.npyio.NpzFile):
        try:
            if "static_features" not in static:
                raise KeyError(f"{path} does not contain static_features")
            return np.asarray(static["static_features"], dtype=np.float32)
        finally:
            static.close()
    return np.asarray(static, dtype=np.float32)


def build_model(
    model_type: str,
    dataset: MarsSequenceDataset,
    sample_x: torch.Tensor,
    sample_aux: torch.Tensor,
    sample_y: torch.Tensor,
    hidden_dims: list[int],
    args: argparse.Namespace,
) -> nn.Module:
    """Construct one architecture using dimensions inferred from the dataset."""
    if args.static_path is None:
        raise ValueError("--static-path is required because every model uses static terrain inputs")
    static_features = load_static_features(args.static_path)
    common = {
        "input_timesteps": sample_x.shape[0],
        "output_timesteps": sample_y.shape[0],
        "input_height": sample_x.shape[2],
        "input_width": sample_x.shape[3],
        "input_channels": sample_x.shape[1],
        "cdod_input_channel_idx": dataset.cdod_input_position,
        "decoder_aux_channels": sample_aux.shape[1],
        "hidden_dims": hidden_dims,
        "kernel_size": (args.kernel_size, args.kernel_size),
        "dropout_prob": args.dropout,
        "detach_feedback": args.detach_feedback,
        "static_channels": static_features.shape[0],
        "static_features": static_features,
    }
    constructors = {
        "convlstm": ConvLSTMModel,
        "convlstm_s2s": ConvLSTMS2SModel,
        "convgru": ConvGRUModel,
        "predrnn": PredRNNModel,
        "swinlstm": SwinLSTMModel,
        "attention_residual": AttentionResidualModel,
    }
    if model_type in constructors:
        if model_type == "attention_residual":
            common["norm_groups"] = args.norm_groups
        return constructors[model_type](**common)
    if model_type == "marscdodnet":
        common.pop("hidden_dims")
        return MarsCDODTerrainFiLMModel(
            **common,
            encoder_hidden_dims=parse_hidden_dims(args.encoder_hidden_dims),
            decoder_hidden_dims=parse_hidden_dims(args.decoder_hidden_dims),
            norm_groups=args.norm_groups,
            film_scale=args.film_scale,
            static_feature_width=args.static_feature_width,
            static_pool_kernel=(args.static_pool_lat, args.static_pool_lon),
        )
    raise ValueError(f"Unknown model type: {model_type}")


def teacher_forcing_ratio(epoch: int, epochs: int, start: float, end: float) -> float:
    if epochs <= 1:
        return end
    return start + (end - start) * epoch / (epochs - 1)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: AdamW | None,
    teacher_forcing: float,
    grad_clip: float,
) -> tuple[float, float]:
    """Run one train or validation epoch and return mean loss and MAE."""
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    mae_sum = 0.0
    for x, future_aux, target in loader:
        x = x.to(device)
        future_aux = future_aux.to(device)
        target = target.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            prediction = model(
                x,
                future_aux=future_aux,
                target_seq=target if training else None,
                teacher_forcing_ratio=teacher_forcing if training else 0.0,
            )
            loss = criterion(prediction, target)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        loss_sum += float(loss.detach())
        mae_sum += float(torch.mean(torch.abs(prediction.detach() - target)))
    if not loader:
        raise ValueError("DataLoader is empty")
    return loss_sum / len(loader), mae_sum / len(loader)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[nn.Module, dict[str, list[float]]]:

    criterion = CompositeLoss(args.gdl_weight) if args.gdl_weight > 0 else nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr
    )
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    stale_epochs = 0
    history = {"train_loss": [], "val_loss": [], "train_mae": [], "val_mae": [], "teacher_forcing": []}

    for epoch in range(args.epochs):
        ratio = teacher_forcing_ratio(
            epoch, args.epochs, args.teacher_forcing_start, args.teacher_forcing_end
        )
        train_loss, train_mae = run_epoch(
            model, train_loader, criterion, device, optimizer, ratio, args.grad_clip
        )
        with torch.no_grad():
            val_loss, val_mae = run_epoch(
                model, val_loader, criterion, device, None, 0.0, args.grad_clip
            )
        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_mae"].append(train_mae)
        history["val_mae"].append(val_mae)
        history["teacher_forcing"].append(ratio)
        learning_rate = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1:03d}/{args.epochs:03d} "
            f"lr={learning_rate:.2e} tf={ratio:.3f} "
            f"train={train_loss:.6f} val={val_loss:.6f} val_mae={val_mae:.6f}"
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.early_stop_patience:
                print(f"Early stopping after {epoch + 1} epochs.")
                break
    model.load_state_dict(best_state)
    return model, history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MarsCDODNet and its recurrent baselines.")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-type", choices=MODEL_TYPES, default="marscdodnet")
    parser.add_argument("--static-path", type=Path)
    parser.add_argument("--input-steps", type=int, default=40)
    parser.add_argument("--input-channel-indices")
    parser.add_argument("--decoder-aux-indices", default="none")
    parser.add_argument("--cdod-channel-idx", type=int, default=0)
    parser.add_argument("--hidden-dims", default="64,64,64")
    parser.add_argument("--encoder-hidden-dims", default="128,128,256")
    parser.add_argument("--decoder-hidden-dims", default="128,64,64")
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--norm-groups", type=int, default=8)
    parser.add_argument("--film-scale", type=float, default=0.1)
    parser.add_argument("--static-feature-width", type=int, default=64)
    parser.add_argument("--static-pool-lat", type=int, default=20)
    parser.add_argument("--static-pool-lon", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--gdl-weight", type=float, default=0.0)
    parser.add_argument("--teacher-forcing-start", type=float, default=0.3)
    parser.add_argument("--teacher-forcing-end", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--detach-feedback", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device(args.device)
    print(f"Device: {device}")

    with np.load(args.data_path, allow_pickle=False) as data:
        required = ("X_train", "y_train", "X_val", "y_val")
        missing = [name for name in required if name not in data]
        if missing:
            raise KeyError(f"Dataset is missing: {', '.join(missing)}")
        x_train = np.asarray(data["X_train"], dtype=np.float32)
        y_train = np.asarray(data["y_train"], dtype=np.float32)
        x_val = np.asarray(data["X_val"], dtype=np.float32)
        y_val = np.asarray(data["y_val"], dtype=np.float32)

    if args.max_train_samples is not None:
        if args.max_train_samples <= 0:
            raise ValueError("--max-train-samples must be positive")
        x_train = x_train[: args.max_train_samples]
        y_train = y_train[: args.max_train_samples]
    input_indices = parse_indices(args.input_channel_indices, x_train.shape[-1])
    decoder_indices = parse_indices(args.decoder_aux_indices, x_train.shape[-1])
    train_dataset = MarsSequenceDataset(
        x_train, y_train, args.input_steps, input_indices, args.cdod_channel_idx, decoder_indices
    )
    val_dataset = MarsSequenceDataset(
        x_val, y_val, args.input_steps, input_indices, args.cdod_channel_idx, decoder_indices
    )
    print(f"Train samples: {len(train_dataset)}, validation samples: {len(val_dataset)}")
    print(f"Encoder channels: {input_indices}")
    print(f"Decoder auxiliary channels: {decoder_indices} ({train_dataset.future_aux_source})")

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    sample_x, sample_aux, sample_y = train_dataset[0]
    model = build_model(
        args.model_type,
        train_dataset,
        sample_x,
        sample_aux,
        sample_y,
        parse_hidden_dims(args.hidden_dims),
        args,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Model: {args.model_type}, trainable parameters: {parameter_count:,}")
    with torch.no_grad():
        check = model(
            sample_x.unsqueeze(0).to(device),
            future_aux=sample_aux.unsqueeze(0).to(device),
        )
    if tuple(check.shape) != (1, *sample_y.shape):
        raise RuntimeError(f"Forward output shape {tuple(check.shape)} does not match target {(1, *sample_y.shape)}")
    print("Forward shape check passed.")

    model, history = train(model, train_loader, val_loader, device, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_type": args.model_type,
        "model_state_dict": model.state_dict(),
        "input_channel_indices": input_indices,
        "decoder_aux_indices": decoder_indices,
        "cdod_channel_idx": args.cdod_channel_idx,
        "input_steps": args.input_steps,
    }
    if args.model_type == "marscdodnet":
        checkpoint["encoder_hidden_dims"] = parse_hidden_dims(args.encoder_hidden_dims)
        checkpoint["decoder_hidden_dims"] = parse_hidden_dims(args.decoder_hidden_dims)
        checkpoint["architecture_version"] = "marscdodnet_figure_v1"
    else:
        checkpoint["hidden_dims"] = parse_hidden_dims(args.hidden_dims)
    torch.save(checkpoint, args.output_dir / "best_model.pt")
    with (args.output_dir / "training_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    print(f"Saved checkpoint and history to {args.output_dir}")


if __name__ == "__main__":
    main()
