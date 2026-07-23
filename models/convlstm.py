

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


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


class ConvLSTMCell(nn.Module):


    def __init__(self, input_dim, hidden_dim, kernel_size=(3, 3)):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gates = PeriodicConv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size)
        self.norm = nn.InstanceNorm2d(hidden_dim, affine=False)

    def forward(self, x, state):
        hidden, memory = state
        forget, update, output, candidate = torch.chunk(
            self.gates(torch.cat((x, hidden), dim=1)), 4, dim=1
        )
        memory = torch.sigmoid(forget) * memory + torch.sigmoid(update) * torch.tanh(candidate)
        hidden = self.norm(torch.sigmoid(output) * torch.tanh(memory))
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


class ConvLSTMModel(nn.Module):


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
            static_tensor = torch.as_tensor(static_features, dtype=torch.float32)
            self.register_buffer("default_static_features", static_tensor, persistent=False)
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
            self.encoder_cells.append(ConvLSTMCell(encoder_channels, hidden_dim, kernel_size))
            self.decoder_cells.append(ConvLSTMCell(decoder_channels, hidden_dim, kernel_size))
        self.dropout = nn.Dropout2d(dropout_prob)
        self.output_head = nn.Sequential(
            PeriodicConv2d(self.hidden_dims[-1], self.hidden_dims[-1], kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dims[-1], 1, kernel_size=1),
        )

    def _static(self, static_features, batch_size, device):
        source = self.default_static_features if static_features is None else static_features
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
        batch, timesteps, channels, height, width = x_dynamic.shape
        expected = (
            batch,
            self.input_timesteps,
            self.input_channels,
            self.input_height,
            self.input_width,
        )
        if tuple(x_dynamic.shape) != expected:
            raise ValueError(f"x_dynamic must have shape {expected}, got {tuple(x_dynamic.shape)}")
        if self.decoder_aux_channels:
            aux_shape = (batch, self.output_timesteps, self.decoder_aux_channels, height, width)
            if future_aux is None or tuple(future_aux.shape) != aux_shape:
                raise ValueError(f"future_aux must have shape {aux_shape}")

        static = self._static(static_features, batch, x_dynamic.device)
        static_sequence = static.unsqueeze(1).expand(-1, timesteps, -1, -1, -1)
        layer_sequence = torch.cat((x_dynamic, static_sequence), dim=2)
        states = []
        for cell in self.encoder_cells:
            state = cell.init_state(batch, (height, width), x_dynamic.device)
            outputs = []
            for step in range(self.input_timesteps):
                state = cell(layer_sequence[:, step], state)
                outputs.append(state[0])
            layer_sequence = torch.stack(outputs, dim=1)
            states.append(state)

        previous_cdod = x_dynamic[:, -1, self.cdod_input_channel_idx : self.cdod_input_channel_idx + 1]
        predictions = []
        for step in range(self.output_timesteps):
            decoder_parts = [previous_cdod, static]
            if self.decoder_aux_channels:
                decoder_parts.insert(1, future_aux[:, step])
            layer_input = torch.cat(decoder_parts, dim=1)
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
            previous_cdod = target_seq[:, step] if use_target else prediction
            if self.detach_feedback and not use_target:
                previous_cdod = previous_cdod.detach()
        return torch.stack(predictions, dim=1)


MarsCDODEncoderDecoderModel = ConvLSTMModel


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
    }
    reference_model = ConvLSTMModel(**reference_config)
    total_parameters = sum(parameter.numel() for parameter in reference_model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in reference_model.parameters() if parameter.requires_grad
    )
    print("Model: ConvLSTMModel")
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
