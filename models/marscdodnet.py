

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


def valid_group_count(channels, maximum=8):
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
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


class GroupNormConvLSTMCell(nn.Module):


    def __init__(self, input_dim, hidden_dim, kernel_size=(3, 3), norm_groups=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gates = PeriodicConv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size)
        self.norm = nn.GroupNorm(valid_group_count(hidden_dim, norm_groups), hidden_dim)

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


class AttentionResidualModel(nn.Module):


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
        hidden_dims=(32, 32),
        kernel_size=(3, 3),
        dropout_prob=0.1,
        detach_feedback=False,
        norm_groups=8,
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
                GroupNormConvLSTMCell(
                    encoder_channels, hidden_dim, kernel_size, norm_groups
                )
            )
            self.decoder_cells.append(
                GroupNormConvLSTMCell(
                    decoder_channels, hidden_dim, kernel_size, norm_groups
                )
            )
        top_dim = self.hidden_dims[-1]
        self.dropout = nn.Dropout2d(dropout_prob)
        self.attention_fusion = nn.Sequential(
            nn.Conv2d(2 * top_dim, top_dim, kernel_size=1),
            nn.GroupNorm(valid_group_count(top_dim, norm_groups), top_dim),
            nn.ReLU(inplace=True),
        )
        self.output_head = nn.Sequential(
            PeriodicConv2d(top_dim, top_dim, kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(top_dim, 1, kernel_size=1),
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

    @staticmethod
    def attend(decoder_hidden, encoder_memory):
        scale = decoder_hidden.shape[1] ** -0.5
        scores = (encoder_memory * decoder_hidden.unsqueeze(1)).sum(dim=2) * scale
        weights = torch.softmax(scores, dim=1)
        return (encoder_memory * weights.unsqueeze(2)).sum(dim=1)

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
        encoder_layers = []
        for cell in self.encoder_cells:
            state = cell.init_state(batch, (height, width), x_dynamic.device)
            outputs = []
            for step in range(self.input_timesteps):
                state = cell(sequence[:, step], state)
                outputs.append(state[0])
            sequence = torch.stack(outputs, dim=1)
            states.append(state)
            encoder_layers.append(sequence)

        encoder_memory = encoder_layers[-1]
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
            context = self.attend(layer_input, encoder_memory)
            fused = self.attention_fusion(torch.cat((layer_input, context), dim=1))
            prediction = previous + self.output_head(self.dropout(fused))
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


class StaticTerrainStream(nn.Module):


    def __init__(
        self,
        static_channels,
        decoder_hidden_dims,
        feature_width=64,
        norm_groups=8,
        pool_kernel=(20, 24),
    ):
        super().__init__()
        self.static_channels = static_channels
        self.pool_kernel = _pair(pool_kernel)
        self.feature_extractor = nn.Sequential(
            PeriodicConv2d(static_channels, 32, kernel_size=3),
            nn.GroupNorm(valid_group_count(32, norm_groups), 32),
            nn.ReLU(inplace=True),
            PeriodicConv2d(32, feature_width, kernel_size=3),
            nn.GroupNorm(valid_group_count(feature_width, norm_groups), feature_width),
            nn.ReLU(inplace=True),
            PeriodicConv2d(feature_width, feature_width, kernel_size=3),
            nn.GroupNorm(valid_group_count(feature_width, norm_groups), feature_width),
            nn.ReLU(inplace=True),
        )
        self.statistics_reducer = nn.Sequential(
            nn.Conv2d(3 * feature_width, feature_width, kernel_size=1),
            nn.GroupNorm(valid_group_count(feature_width, norm_groups), feature_width),
            nn.ReLU(inplace=True),
        )
        self.film_heads = nn.ModuleList(
            nn.Conv2d(feature_width, 2 * hidden_dim, kernel_size=1)
            for hidden_dim in decoder_hidden_dims
        )
        for head in self.film_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, static_features):
        if static_features.ndim == 3:
            static_features = static_features.unsqueeze(0)
        if static_features.ndim != 4 or static_features.shape[1] != self.static_channels:
            raise ValueError(
                f"static_features must be [B,{self.static_channels},H,W], "
                f"got {tuple(static_features.shape)}"
            )
        features = self.feature_extractor(static_features)
        mean = F.avg_pool2d(
            features, kernel_size=self.pool_kernel, stride=self.pool_kernel
        )
        maximum = F.max_pool2d(
            features, kernel_size=self.pool_kernel, stride=self.pool_kernel
        )
        square_mean = F.avg_pool2d(
            features.square(), kernel_size=self.pool_kernel, stride=self.pool_kernel
        )
        deviation = torch.sqrt(torch.clamp(square_mean - mean.square(), min=0) + 1e-6)
        coarse_features = self.statistics_reducer(
            torch.cat((mean, maximum, deviation), dim=1)
        )
        return [torch.chunk(head(coarse_features), 2, dim=1) for head in self.film_heads]


class MarsCDODTerrainFiLMModel(nn.Module):


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
        encoder_hidden_dims=(128, 128, 256),
        decoder_hidden_dims=(128, 64, 64),
        kernel_size=(3, 3),
        dropout_prob=0.2,
        detach_feedback=False,
        norm_groups=8,
        static_features=None,
        film_scale=0.1,
        static_feature_width=64,
        static_pool_kernel=(20, 24),
    ):
        super().__init__()
        if len(encoder_hidden_dims) != 3 or len(decoder_hidden_dims) != 3:
            raise ValueError(
                "MarsCDODNet requires three encoder and three decoder hidden dimensions"
            )
        if static_feature_width != 64:
            raise ValueError("The reference static branch requires static_feature_width=64")
        self.input_timesteps = input_timesteps
        self.output_timesteps = output_timesteps
        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels
        self.static_channels = static_channels
        self.cdod_input_channel_idx = cdod_input_channel_idx
        self.decoder_aux_channels = decoder_aux_channels
        self.encoder_hidden_dims = list(encoder_hidden_dims)
        self.decoder_hidden_dims = list(decoder_hidden_dims)
        self.detach_feedback = detach_feedback
        self.film_scale = float(film_scale)
        self.static_pool_kernel = _pair(static_pool_kernel)
        if static_features is not None:
            self.register_buffer(
                "default_static_features",
                torch.as_tensor(static_features, dtype=torch.float32),
                persistent=False,
            )
        else:
            self.default_static_features = None

        self.dynamic_encoder_cells = nn.ModuleList()
        self.dynamic_decoder_cells = nn.ModuleList()
        for index, hidden_dim in enumerate(self.encoder_hidden_dims):
            encoder_channels = (
                input_channels if index == 0 else self.encoder_hidden_dims[index - 1]
            )
            self.dynamic_encoder_cells.append(
                GroupNormConvLSTMCell(
                    encoder_channels, hidden_dim, kernel_size, norm_groups
                )
            )
        for index, hidden_dim in enumerate(self.decoder_hidden_dims):
            decoder_channels = (
                1 + decoder_aux_channels
                if index == 0
                else self.decoder_hidden_dims[index - 1]
            )
            self.dynamic_decoder_cells.append(
                GroupNormConvLSTMCell(
                    decoder_channels, hidden_dim, kernel_size, norm_groups
                )
            )
        encoder_top_dim = self.encoder_hidden_dims[-1]
        self.hidden_state_bridges = nn.ModuleList(
            nn.Conv2d(encoder_top_dim, hidden_dim, kernel_size=1)
            for hidden_dim in self.decoder_hidden_dims
        )
        self.memory_state_bridges = nn.ModuleList(
            nn.Conv2d(encoder_top_dim, hidden_dim, kernel_size=1)
            for hidden_dim in self.decoder_hidden_dims
        )
        self.static_stream = StaticTerrainStream(
            static_channels,
            self.decoder_hidden_dims,
            static_feature_width,
            norm_groups,
            self.static_pool_kernel,
        )
        top_dim = self.decoder_hidden_dims[-1]
        self.dropout = nn.Dropout2d(dropout_prob)
        self.output_head = nn.Sequential(
            PeriodicConv2d(top_dim, top_dim, kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(top_dim, 1, kernel_size=1),
        )

    def _static_maps(self, supplied, batch_size, device):
        source = self.default_static_features if supplied is None else supplied
        if source is None:
            raise ValueError("static_features must be supplied to the constructor or forward method")
        static = torch.as_tensor(source, dtype=torch.float32, device=device)
        if static.ndim == 3:
            static = static.unsqueeze(0)
        expected_size = (
            self.input_height * self.static_pool_kernel[0],
            self.input_width * self.static_pool_kernel[1],
        )
        if tuple(static.shape[-2:]) != expected_size:
            raise ValueError(
                f"Static grid must be {expected_size} for dynamic grid "
                f"{(self.input_height, self.input_width)}, got {tuple(static.shape[-2:])}"
            )
        layers = self.static_stream(static)
        scaled_maps = []
        for raw_gamma, raw_beta in layers:
            gamma = 1.0 + self.film_scale * raw_gamma
            beta = self.film_scale * raw_beta
            if gamma.shape[0] == 1 and batch_size > 1:
                gamma = gamma.expand(batch_size, -1, -1, -1)
                beta = beta.expand(batch_size, -1, -1, -1)
            scaled_maps.append((gamma, beta))
        if tuple(scaled_maps[0][0].shape[-2:]) != (
            self.input_height,
            self.input_width,
        ):
            raise ValueError(
                "Static pooling must produce the dynamic grid "
                f"{(self.input_height, self.input_width)}, "
                f"got {tuple(scaled_maps[0][0].shape[-2:])}"
            )
        return scaled_maps

    @staticmethod
    def apply_film(hidden, films, layer_index):
        if films is None:
            return hidden
        gamma, beta = films[layer_index]
        return gamma * hidden + beta

    def forward(
        self,
        x_dynamic,
        static_features=None,
        future_aux=None,
        target_seq=None,
        teacher_forcing_ratio=0.0,
    ):
        batch, _, _, height, width = x_dynamic.shape
        expected = (
            batch,
            self.input_timesteps,
            self.input_channels,
            self.input_height,
            self.input_width,
        )
        if tuple(x_dynamic.shape) != expected:
            raise ValueError(f"x_dynamic must have shape {expected}, got {tuple(x_dynamic.shape)}")
        if target_seq is not None:
            expected_target = (
                batch,
                self.output_timesteps,
                1,
                self.input_height,
                self.input_width,
            )
            if tuple(target_seq.shape) != expected_target:
                raise ValueError(
                    f"target_seq must have shape {expected_target}, got {tuple(target_seq.shape)}"
                )
        if self.decoder_aux_channels:
            expected_aux = (
                batch,
                self.output_timesteps,
                self.decoder_aux_channels,
                self.input_height,
                self.input_width,
            )
            if future_aux is None or tuple(future_aux.shape) != expected_aux:
                raise ValueError(
                    f"future_aux must have shape {expected_aux}, "
                    f"got {None if future_aux is None else tuple(future_aux.shape)}"
                )
        decoder_films = self._static_maps(static_features, batch, x_dynamic.device)

        sequence = x_dynamic
        top_encoder_state = None
        for cell in self.dynamic_encoder_cells:
            state = cell.init_state(batch, (height, width), x_dynamic.device)
            outputs = []
            for step in range(self.input_timesteps):
                state = cell(sequence[:, step], state)
                outputs.append(state[0])
            sequence = torch.stack(outputs, dim=1)
            top_encoder_state = state

        encoder_hidden, encoder_memory = top_encoder_state
        states = [
            (hidden_bridge(encoder_hidden), memory_bridge(encoder_memory))
            for hidden_bridge, memory_bridge in zip(
                self.hidden_state_bridges, self.memory_state_bridges
            )
        ]

        previous = x_dynamic[:, -1, self.cdod_input_channel_idx : self.cdod_input_channel_idx + 1]
        predictions = []
        for step in range(self.output_timesteps):
            if self.decoder_aux_channels:
                if future_aux is None:
                    raise ValueError("future_aux is required when decoder_aux_channels is positive")
                layer_input = torch.cat((previous, future_aux[:, step]), dim=1)
            else:
                layer_input = previous
            next_states = []
            for layer_index, cell in enumerate(self.dynamic_decoder_cells):
                hidden, memory = cell(layer_input, states[layer_index])
                hidden = self.apply_film(hidden, decoder_films, layer_index)
                next_states.append((hidden, memory))
                layer_input = (
                    self.dropout(hidden)
                    if layer_index < len(self.dynamic_decoder_cells) - 1
                    else hidden
                )
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


MarsCDODNet = MarsCDODTerrainFiLMModel
MarsCDODAttentionResidualModel = AttentionResidualModel


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
        "encoder_hidden_dims": (128, 128, 256),
        "decoder_hidden_dims": (128, 64, 64),
        "kernel_size": (3, 3),
        "dropout_prob": 0.2,
        "decoder_aux_channels": 0,
        "film_scale": 0.1,
        "static_feature_width": 64,
        "static_pool_kernel": (20, 24),
    }
    reference_model = MarsCDODNet(**reference_config)
    total_parameters = sum(parameter.numel() for parameter in reference_model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in reference_model.parameters() if parameter.requires_grad
    )
    print("Model: MarsCDODTerrainFiLMModel")
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
