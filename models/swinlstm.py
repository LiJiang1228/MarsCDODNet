

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


def valid_head_count(channels, maximum):
    for heads in range(min(channels, maximum), 0, -1):
        if channels % heads == 0:
            return heads
    return 1


class PeriodicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = _pair(kernel_size)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
            bound = fan_in**-0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        pad_h, pad_w = self.padding
        x = F.pad(x, (pad_w, pad_w, 0, 0), mode="circular")
        x = F.pad(x, (0, 0, pad_h, pad_h))
        return F.conv2d(x, self.weight, self.bias)


class WindowAttention(nn.Module):


    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = valid_head_count(channels, num_heads)
        self.head_dim = channels // self.num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(channels, 3 * channels)
        self.projection = nn.Linear(channels, channels)

    def forward(self, x, window_size, attention_mask=None):
        batch, height, width, channels = x.shape
        pad_h = (window_size - height % window_size) % window_size
        pad_w = (window_size - width % window_size) % window_size
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        padded_h, padded_w = x.shape[1:3]
        windows = x.reshape(
            batch,
            padded_h // window_size,
            window_size,
            padded_w // window_size,
            window_size,
            channels,
        )
        windows = windows.permute(0, 1, 3, 2, 4, 5).reshape(
            -1, window_size * window_size, channels
        )
        qkv = self.qkv(windows).reshape(
            windows.shape[0], windows.shape[1], 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4)
        attention = (query @ key.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            if attention_mask.shape != (windows.shape[0] // batch, windows.shape[1], windows.shape[1]):
                raise ValueError("Invalid shifted-window attention mask shape")
            mask = attention_mask.repeat(batch, 1, 1).unsqueeze(1)
            attention = attention.masked_fill(~mask, -1e4)
        attention = torch.softmax(attention, dim=-1)
        output = (attention @ value).transpose(1, 2).reshape_as(windows)
        output = self.projection(output)
        output = output.reshape(
            batch,
            padded_h // window_size,
            padded_w // window_size,
            window_size,
            window_size,
            channels,
        )
        output = output.permute(0, 1, 3, 2, 4, 5).reshape(
            batch, padded_h, padded_w, channels
        )
        return output[:, :height, :width]


class SwinTransformerBlock(nn.Module):


    def __init__(self, channels, window_size=6, shift_size=0, num_heads=4, mlp_ratio=2.0):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(channels)
        self.attention = WindowAttention(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        mlp_channels = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mlp_channels),
            nn.GELU(),
            nn.Linear(mlp_channels, channels),
        )

    @staticmethod
    def _shifted_window_mask(height, width, window_size, shift_size, device):

        pad_h = (window_size - height % window_size) % window_size
        pad_w = (window_size - width % window_size) % window_size
        padded_h, padded_w = height + pad_h, width + pad_w
        valid = torch.ones((height, width), dtype=torch.bool, device=device)
        latitude_group = torch.zeros((height, width), dtype=torch.int64, device=device)
        if shift_size:
            valid = torch.roll(valid, shifts=(-shift_size, -shift_size), dims=(0, 1))
            source_latitude = torch.roll(
                torch.arange(height, device=device), shifts=-shift_size, dims=0
            )
            latitude_group = (
                (source_latitude[:, None] < shift_size).to(torch.int64).expand(-1, width)
            )
            latitude_group = torch.roll(latitude_group, shifts=(0, -shift_size), dims=(0, 1))
        if pad_h or pad_w:
            valid = F.pad(valid.to(torch.int8), (0, pad_w, 0, pad_h), value=0).bool()
            latitude_group = F.pad(latitude_group, (0, pad_w, 0, pad_h), value=-1)
        valid_windows = valid.reshape(
            padded_h // window_size, window_size, padded_w // window_size, window_size
        ).permute(0, 2, 1, 3).reshape(-1, window_size * window_size)
        group_windows = latitude_group.reshape(
            padded_h // window_size, window_size, padded_w // window_size, window_size
        ).permute(0, 2, 1, 3).reshape(-1, window_size * window_size)
        return (
            valid_windows[:, :, None]
            & valid_windows[:, None, :]
            & (group_windows[:, :, None] == group_windows[:, None, :])
        )

    def forward(self, x):
        _, _, height, width = x.shape
        window_size = max(1, min(self.window_size, height, width))
        shift_size = min(self.shift_size, window_size // 2) if window_size > 1 else 0
        residual = x.permute(0, 2, 3, 1)
        attention_input = self.norm1(residual)
        if shift_size:
            attention_input = torch.roll(
                attention_input, shifts=(-shift_size, -shift_size), dims=(1, 2)
            )
        attention_mask = self._shifted_window_mask(
            height, width, window_size, shift_size, x.device
        )
        attended = self.attention(attention_input, window_size, attention_mask)
        if shift_size:
            attended = torch.roll(attended, shifts=(shift_size, shift_size), dims=(1, 2))
        output = residual + attended
        output = output + self.mlp(self.norm2(output))
        return output.permute(0, 3, 1, 2).contiguous()


class SwinLSTMCell(nn.Module):


    def __init__(
        self,
        input_dim,
        hidden_dim,
        kernel_size=(3, 3),
        window_size=6,
        num_heads=4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_projection = PeriodicConv2d(
            input_dim + hidden_dim, hidden_dim, kernel_size
        )
        self.blocks = nn.ModuleList(
            (
                SwinTransformerBlock(hidden_dim, window_size, 0, num_heads),
                SwinTransformerBlock(
                    hidden_dim, window_size, window_size // 2, num_heads
                ),
            )
        )
        self.gates = nn.Conv2d(hidden_dim, 4 * hidden_dim, kernel_size=1)

    def forward(self, x, state):
        hidden, memory = state
        features = self.input_projection(torch.cat((x, hidden), dim=1))
        for block in self.blocks:
            features = block(features)
        update, forget, output, candidate = torch.chunk(self.gates(features), 4, dim=1)
        memory = (
            torch.sigmoid(forget + 1.0) * memory
            + torch.sigmoid(update) * torch.tanh(candidate)
        )
        hidden = torch.sigmoid(output) * torch.tanh(memory)
        return hidden, memory

    def init_state(self, batch_size, spatial_size, device):
        shape = (batch_size, self.hidden_dim, *spatial_size)
        return torch.zeros(shape, device=device), torch.zeros(shape, device=device)


def prepare_static_features(static_features, batch_size, height, width, device):
    static = torch.as_tensor(static_features, dtype=torch.float32, device=device)
    if static.ndim == 3:
        static = static.unsqueeze(0)
    if static.ndim != 4:
        raise ValueError(f"static_features must be [C,H,W] or [B,C,H,W], got {tuple(static.shape)}")
    static = F.adaptive_avg_pool2d(static, (height, width))
    if static.shape[0] == 1:
        static = static.expand(batch_size, -1, -1, -1)
    elif static.shape[0] != batch_size:
        raise ValueError(f"Static batch must be 1 or {batch_size}, got {static.shape[0]}")
    return static


class SwinLSTMModel(nn.Module):


    def __init__(
        self,
        input_timesteps=40,
        output_timesteps=12,
        input_height=36,
        input_width=60,
        input_channels=23,
        static_channels=8,
        cdod_input_channel_idx=0,
        decoder_aux_channels=0,
        hidden_dims=(64, 64, 64),
        kernel_size=(3, 3),
        dropout_prob=0.2,
        detach_feedback=False,
        static_features=None,
        window_size=6,
        num_heads=4,
    ):
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer")
        self.input_timesteps = input_timesteps
        self.output_timesteps = output_timesteps
        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels
        self.static_channels = static_channels
        self.cdod_input_channel_idx = cdod_input_channel_idx
        self.decoder_aux_channels = decoder_aux_channels
        self.hidden_dims = list(hidden_dims)
        self.detach_feedback = detach_feedback
        if static_features is not None:
            self.register_buffer(
                "default_static_features",
                torch.as_tensor(static_features, dtype=torch.float32),
                persistent=False,
            )
        else:
            self.default_static_features = None

        self.encoder_cells = nn.ModuleList()
        self.decoder_cells = nn.ModuleList()
        for index, hidden_dim in enumerate(self.hidden_dims):
            encoder_channels = (
                input_channels + static_channels if index == 0 else self.hidden_dims[index - 1]
            )
            decoder_channels = (
                1 + decoder_aux_channels + static_channels
                if index == 0
                else self.hidden_dims[index - 1]
            )
            self.encoder_cells.append(
                SwinLSTMCell(
                    encoder_channels,
                    hidden_dim,
                    kernel_size,
                    window_size,
                    num_heads,
                )
            )
            self.decoder_cells.append(
                SwinLSTMCell(
                    decoder_channels,
                    hidden_dim,
                    kernel_size,
                    window_size,
                    num_heads,
                )
            )
        self.dropout = nn.Dropout2d(dropout_prob)
        self.output_head = nn.Sequential(
            PeriodicConv2d(self.hidden_dims[-1], self.hidden_dims[-1], kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dims[-1], 1, kernel_size=1),
        )

    def _static(self, supplied, batch_size, device):
        source = self.default_static_features if supplied is None else supplied
        if source is None:
            raise ValueError("static_features must be supplied to the constructor or forward method")
        static = prepare_static_features(
            source, batch_size, self.input_height, self.input_width, device
        )
        if static.shape[1] != self.static_channels:
            raise ValueError(f"Expected {self.static_channels} static channels, got {static.shape[1]}")
        return static

    def forward(
        self,
        x_dynamic,
        static_features=None,
        future_aux=None,
        target_seq=None,
        teacher_forcing_ratio=0.0,
    ):
        batch, timesteps, _, height, width = x_dynamic.shape
        expected = (
            batch,
            self.input_timesteps,
            self.input_channels,
            self.input_height,
            self.input_width,
        )
        if tuple(x_dynamic.shape) != expected:
            raise ValueError(f"x_dynamic must have shape {expected}, got {tuple(x_dynamic.shape)}")
        static = self._static(static_features, batch, x_dynamic.device)
        static_sequence = static.unsqueeze(1).expand(-1, timesteps, -1, -1, -1)
        sequence = torch.cat((x_dynamic, static_sequence), dim=2)
        states = []
        for cell in self.encoder_cells:
            state = cell.init_state(batch, (height, width), x_dynamic.device)
            outputs = []
            for step in range(self.input_timesteps):
                state = cell(sequence[:, step], state)
                outputs.append(state[0])
            sequence = torch.stack(outputs, dim=1)
            states.append(state)

        previous = x_dynamic[:, -1, self.cdod_input_channel_idx : self.cdod_input_channel_idx + 1]
        predictions = []
        for step in range(self.output_timesteps):
            parts = [previous, static]
            if self.decoder_aux_channels:
                if future_aux is None:
                    raise ValueError("future_aux is required when decoder_aux_channels is positive")
                parts.insert(1, future_aux[:, step])
            layer_input = torch.cat(parts, dim=1)
            next_states = []
            for index, cell in enumerate(self.decoder_cells):
                state = cell(layer_input, states[index])
                next_states.append(state)
                layer_input = self.dropout(state[0]) if index < len(self.decoder_cells) - 1 else state[0]
            states = next_states
            prediction = self.output_head(self.dropout(layer_input))
            predictions.append(prediction)
            use_target = (
                self.training
                and target_seq is not None
                and teacher_forcing_ratio > 0
                and torch.rand((), device=x_dynamic.device) < teacher_forcing_ratio
            )
            previous = target_seq[:, step] if use_target else prediction
            if self.detach_feedback and not use_target:
                previous = previous.detach()
        return torch.stack(predictions, dim=1)


MarsCDODSwinLSTMModel = SwinLSTMModel


if __name__ == "__main__":
    from pathlib import Path

    import numpy as np

    reference_config = {
        "input_timesteps": 40,
        "output_timesteps": 12,
        "input_height": 36,
        "input_width": 60,
        "input_channels": 23,
        "static_channels": 8,
        "hidden_dims": (64, 64, 64),
        "kernel_size": (3, 3),
        "dropout_prob": 0.2,
        "decoder_aux_channels": 0,
        "window_size": 6,
        "num_heads": 4,
    }
    reference_model = SwinLSTMModel(**reference_config)
    total_parameters = sum(parameter.numel() for parameter in reference_model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in reference_model.parameters() if parameter.requires_grad
    )
    print("Model: SwinLSTMModel")
    print("Reference configuration:")
    for name, value in reference_config.items():
        print(f"  {name}: {value}")
    print(reference_model)
    print(f"Total parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")
    print("Reference dynamic input shape: [batch, 40, 23, 36, 60]")
    print("Reference static input shape: [8, 720, 1440]")
    print("Reference output shape: [batch, 12, 1, 36, 60]")

    example_dir = Path(__file__).resolve().parents[1] / "data_example"
    dynamic_input = torch.from_numpy(np.load(example_dir / "X_dynamic_example.npy")).permute(
        0, 1, 4, 2, 3
    )
    target_output = torch.from_numpy(np.load(example_dir / "y_cdod_example.npy")).permute(
        0, 1, 4, 2, 3
    )
    with np.load(example_dir / "static_terrain_example.npz") as static_file:
        static_input = torch.from_numpy(static_file["static_features"])
    reference_model.eval()
    with torch.inference_mode():
        model_output = reference_model(dynamic_input, static_input)
    print(f"Dynamic input shape: {tuple(dynamic_input.shape)}")
    print(f"Static input shape: {tuple(static_input.shape)}")
    print(f"Target output shape: {tuple(target_output.shape)}")
    print(f"Model output shape: {tuple(model_output.shape)}")
    print("Output layout: [batch, forecast_time, CDOD_channel, latitude, longitude]")
