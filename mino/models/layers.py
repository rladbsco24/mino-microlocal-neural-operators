from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _finite_tensor(tensor: Tensor, *, cap: float | None = None) -> Tensor:
    """Return a finite packet tensor, optionally capped symmetrically."""

    if cap is None:
        return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    cap = max(float(abs(cap)), 1e-6)
    return torch.nan_to_num(tensor, nan=0.0, posinf=cap, neginf=-cap).clamp(-cap, cap)


def _metadata_batch(metadata: Tensor, features: Tensor) -> Tensor:
    if metadata.dim() == 2:
        return metadata.unsqueeze(0).expand(features.shape[0], features.shape[1], metadata.shape[-1])
    if metadata.dim() == 3:
        if metadata.shape[0] != features.shape[0] or metadata.shape[1] != features.shape[1]:
            raise ValueError("Batched metadata must match feature batch and token dimensions.")
        return metadata
    raise ValueError("metadata must have shape (tokens, dim) or (batch, tokens, dim).")


def _helmholtz_shell_features(
    metadata: Tensor,
    features: Tensor | None = None,
    *,
    shell_radius: float,
    refractive_index: float,
    absorption: float,
    cap: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return finite Helmholtz symbol features on the packet phase window.

    The executable metadata frequencies are normalized packet coordinates, so
    ``shell_radius`` is a radius in that coordinate system.  A positive value
    fixes the shell.  A non-positive value uses a token-energy weighted shell
    estimate, which is safer when the executable packet frequencies are not
    calibrated to a physical wavenumber.  The returned signed real/imaginary
    weights are a finite real representation of ``(p+i eta)^{-1}``.
    """
    metadata = _finite_tensor(metadata)
    if features is not None:
        features = _finite_tensor(features, cap=1.0e6)
    cap = max(float(abs(cap)), 1e-6)
    freq = metadata[..., 2:4]
    freq_norm = torch.linalg.vector_norm(freq, dim=-1, keepdim=True).clamp_min(1e-8)
    unit_freq = freq / freq_norm
    log_freq = torch.log1p(freq_norm)
    if shell_radius > 0.0:
        radius = float(shell_radius) * math.sqrt(max(float(refractive_index), 1e-8))
        shell_center = torch.full_like(freq_norm, max(radius, 1e-6))
    else:
        if features is not None:
            token_energy = features.detach().square().mean(dim=-1, keepdim=True)
            token_energy = _finite_tensor(token_energy, cap=1.0e12)
            token_weight = (token_energy + 1e-8) * freq_norm.detach().square().clamp_min(1e-4)
            shell_center = (freq_norm * token_weight).sum(dim=1, keepdim=True) / token_weight.sum(dim=1, keepdim=True).clamp_min(1e-8)
        else:
            shell_center = freq_norm.mean(dim=1, keepdim=True) + freq_norm.std(dim=1, keepdim=True, unbiased=False)
        shell_center = shell_center.clamp_min(1e-6)
    shell_distance = (freq_norm - shell_center) / shell_center
    p_proxy = (freq_norm.square() - shell_center.square()) / shell_center.square().clamp_min(1e-8)
    eta = max(float(absorption), 1e-6)
    denominator = (p_proxy.square() + eta * eta).clamp_min(1e-8)
    resolvent_real = _finite_tensor(p_proxy / denominator, cap=cap)
    resolvent_imag = _finite_tensor(-eta / denominator, cap=cap)
    resolvent_envelope = torch.sqrt(resolvent_real.square() + resolvent_imag.square()).clamp(0.0, cap)
    position = metadata[..., :2] - 0.5
    position_norm = torch.linalg.vector_norm(position, dim=-1, keepdim=True).clamp_min(1e-8)
    outgoing_flux = (position * unit_freq).sum(dim=-1, keepdim=True) / position_norm
    outgoing_gate = 0.25 + 0.75 * torch.sigmoid(4.0 * outgoing_flux)
    return (
        _finite_tensor(log_freq),
        _finite_tensor(unit_freq),
        _finite_tensor(shell_distance, cap=cap),
        _finite_tensor(resolvent_envelope, cap=cap),
        _finite_tensor(outgoing_flux, cap=1.0),
        resolvent_real,
        resolvent_imag,
        _finite_tensor(outgoing_gate, cap=1.0),
        _finite_tensor(shell_center, cap=1.0e6),
    )


def _init_identity_or_zero(linear: nn.Linear, *, identity: bool) -> None:
    with torch.no_grad():
        linear.weight.zero_()
        if identity:
            rows, cols = linear.weight.shape
            diagonal = min(rows, cols)
            linear.weight[:diagonal, :diagonal] = torch.eye(
                diagonal,
                device=linear.weight.device,
                dtype=linear.weight.dtype,
            )


class ComplexPairSymbolAction(nn.Module):
    """Bias-free real implementation of latent complex multiplication.

    The layer lifts packet features to an explicit latent complex pair
    ``u+i v``, applies the exact real formula for multiplication by
    ``a+i b``, and projects the result back to the packet feature space.
    Biases are intentionally disabled so theorem-facing multiplier modes keep
    the zero-coefficient-to-zero-coefficient property.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.real_lift = nn.Linear(width, width, bias=False)
        self.imag_lift = nn.Linear(width, width, bias=False)
        self.real_project = nn.Linear(width, width, bias=False)
        self.imag_project = nn.Linear(width, width, bias=False)
        _init_identity_or_zero(self.real_lift, identity=True)
        _init_identity_or_zero(self.imag_lift, identity=False)
        _init_identity_or_zero(self.real_project, identity=True)
        _init_identity_or_zero(self.imag_project, identity=False)

    def forward(self, features: Tensor, real_multiplier: Tensor, imag_multiplier: Tensor) -> Tensor:
        real_features = self.real_lift(features)
        imag_features = self.imag_lift(features)
        out_real = real_multiplier * real_features - imag_multiplier * imag_features
        out_imag = real_multiplier * imag_features + imag_multiplier * real_features
        return self.real_project(out_real) + self.imag_project(out_imag)

    def latent_imaginary_energy(self, features: Tensor, eps: float = 1e-8) -> Tensor:
        real_features = self.real_lift(features)
        imag_features = self.imag_lift(features)
        return imag_features.square().mean() / real_features.square().mean().clamp_min(eps)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class EdgePacketKernel(nn.Module):
    """Edge-local packet symbol on retained canonical stencil entries.

    The input is an executable finite version of
    ``s(z_j, nu_ij; h)``: source packet features, transported source/target
    metadata, and the normalized local tube coordinate.  The multiplier is
    initialized at identity so enabling the edge kernel does not erase the
    existing packet transport path at initialization.
    """

    def __init__(self, width: int, metadata_dim: int, hidden_dim: int = 128, strength: float = 0.5) -> None:
        super().__init__()
        self.strength = float(strength)
        in_dim = width + 2 * metadata_dim + 4
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, width),
        )
        last = self.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(
        self,
        source_features: Tensor,
        source_metadata: Tensor,
        target_metadata: Tensor,
        local_tube_coordinate: Tensor,
    ) -> Tensor:
        source_features = _finite_tensor(source_features, cap=1.0e6)
        source_metadata = _finite_tensor(source_metadata, cap=10.0)
        target_metadata = _finite_tensor(target_metadata, cap=10.0)
        local_tube_coordinate = _finite_tensor(local_tube_coordinate, cap=10.0)
        raw = self.net(torch.cat([source_features, source_metadata, target_metadata, local_tube_coordinate], dim=-1))
        return 1.0 + self.strength * torch.tanh(_finite_tensor(raw, cap=30.0))


class CanonicalPropagationLayer(nn.Module):
    def __init__(
        self,
        width: int,
        metadata_dim: int = 5,
        hidden_dim: int = 128,
        transport_scale: float = 0.05,
        stencil_size: int | None = None,
        confidence_gating: bool = False,
        transport_parameterization: str = "mlp_displacement",
        sparse_topk: bool = False,
        wavefront_confidence_scale: float = 0.0,
        edge_symbol_parameterization: str = "none",
        edge_symbol_strength: float = 0.5,
    ) -> None:
        super().__init__()
        self.transport_scale = transport_scale
        self.stencil_size = stencil_size
        self.confidence_gating = confidence_gating
        self.transport_parameterization = transport_parameterization
        self.sparse_topk = sparse_topk
        self.wavefront_confidence_scale = float(wavefront_confidence_scale)
        self.edge_symbol_parameterization = edge_symbol_parameterization
        state_dim = width + metadata_dim
        self.transport_mlp = MLP(state_dim, hidden_dim, 4)
        if transport_parameterization == "hamiltonian_euler":
            self.hamiltonian_mlp = MLP(state_dim, hidden_dim, 1)
        elif transport_parameterization == "hamiltonian_verlet":
            # Separable Hamiltonian H(q,p; token)=V(q; token)+T(p; token).
            # For fixed token features, the Störmer-Verlet update is
            # symplectic in the packet metadata variables (q,p).
            self.potential_mlp = MLP(width + 2, hidden_dim, 1)
            self.kinetic_mlp = MLP(width + 2, hidden_dim, 1)
        elif transport_parameterization != "mlp_displacement":
            raise ValueError(f"Unsupported transport_parameterization: {transport_parameterization}")
        self.message_mlp = MLP(width + metadata_dim, hidden_dim, width)
        self.length_scale = nn.Parameter(torch.tensor([6.0, 6.0, 10.0, 10.0], dtype=torch.float32))
        self.output_gate = nn.Linear(width, width)
        self.temperature = math.sqrt(max(width, 1))
        if confidence_gating:
            self.confidence_head = nn.Linear(width + metadata_dim, width)
        else:
            self.confidence_head = None
        if edge_symbol_parameterization == "none":
            self.edge_symbol = None
        elif edge_symbol_parameterization == "local_packet_kernel":
            self.edge_symbol = EdgePacketKernel(
                width,
                metadata_dim=metadata_dim,
                hidden_dim=hidden_dim,
                strength=edge_symbol_strength,
            )
        else:
            raise ValueError(f"Unsupported edge_symbol_parameterization: {edge_symbol_parameterization}")
        self.last_wavefront_confidence_proxy = torch.tensor(0.0)
        self.last_local_tube_coordinate: Tensor | None = None
        self.last_edge_symbol_deviation_proxy = torch.tensor(0.0)

    def _transport_delta(self, features: Tensor, metadata_batch: Tensor) -> Tensor:
        state = torch.cat([features, metadata_batch], dim=-1)
        if self.transport_parameterization == "mlp_displacement":
            return torch.tanh(self.transport_mlp(state)) * self.transport_scale

        if self.transport_parameterization == "hamiltonian_euler":
            # Hamiltonian-Euler mode: learn H(x, xi, packet state) and use
            # (xdot, xidot) = (d_xi H, -d_x H). The metadata order is
            # (y, x, ky, kx, scale), so the canonical block uses the first four
            # coordinates. This is a restricted semiclassical bias, not a full
            # symplectic theorem by itself.
            with torch.enable_grad():
                metadata_var = metadata_batch.detach().clone().requires_grad_(True)
                h_state = torch.cat([features.detach(), metadata_var], dim=-1)
                hamiltonian = self.hamiltonian_mlp(h_state).sum()
                grad = torch.autograd.grad(
                    hamiltonian,
                    metadata_var,
                    create_graph=self.training,
                    retain_graph=self.training,
                )[0]
                delta = torch.zeros_like(metadata_var[..., :4])
                delta[..., 0:2] = grad[..., 2:4]
                delta[..., 2:4] = -grad[..., 0:2]
                return torch.tanh(delta) * self.transport_scale

        # Hamiltonian-Verlet mode: a separable Hamiltonian step in the first
        # four metadata coordinates q=(y,x), p=(ky,kx).  This is the
        # theorem-facing transport path: unlike explicit Euler, the forward
        # map is the Störmer-Verlet composition for fixed token features.
        with torch.enable_grad():
            features_const = features.detach()
            q0 = metadata_batch[..., 0:2]
            p0 = metadata_batch[..., 2:4]
            step = torch.as_tensor(self.transport_scale, device=features.device, dtype=features.dtype)

            q0_var = q0.detach().clone().requires_grad_(True)
            potential0 = self.potential_mlp(torch.cat([features_const, q0_var], dim=-1)).sum()
            grad_v0 = torch.autograd.grad(
                potential0,
                q0_var,
                create_graph=self.training,
                retain_graph=True,
            )[0]
            p_half = p0 - 0.5 * step * grad_v0

            p_half_var = p_half.detach().clone().requires_grad_(True)
            kinetic_half = self.kinetic_mlp(torch.cat([features_const, p_half_var], dim=-1)).sum()
            grad_t_half = torch.autograd.grad(
                kinetic_half,
                p_half_var,
                create_graph=self.training,
                retain_graph=True,
            )[0]
            q_new = q0 + step * grad_t_half

            q_new_var = q_new.detach().clone().requires_grad_(True)
            potential1 = self.potential_mlp(torch.cat([features_const, q_new_var], dim=-1)).sum()
            grad_v1 = torch.autograd.grad(
                potential1,
                q_new_var,
                create_graph=self.training,
                retain_graph=self.training,
            )[0]
            p_new = p_half - 0.5 * step * grad_v1
            return torch.cat([q_new - q0, p_new - p0], dim=-1)

    def _canonical_defect_proxy(self, delta: Tensor) -> Tensor:
        if self.transport_parameterization == "hamiltonian_verlet":
            return delta.new_tensor(0.0)
        return delta.square().mean()

    def _edge_symbol_multiplier(
        self,
        source_features: Tensor,
        source_metadata: Tensor,
        target_metadata: Tensor,
        local_tube_coordinate: Tensor,
    ) -> Tensor:
        if self.edge_symbol is None:
            self.last_edge_symbol_deviation_proxy = source_features.new_tensor(0.0).detach()
            return torch.ones_like(source_features)
        multiplier = self.edge_symbol(source_features, source_metadata, target_metadata, local_tube_coordinate)
        self.last_edge_symbol_deviation_proxy = (multiplier - 1.0).abs().mean().detach()
        return multiplier

    def forward(self, features: Tensor, metadata: Tensor) -> tuple[Tensor, Tensor]:
        batch, tokens, width = features.shape
        self.last_wavefront_confidence_proxy = features.new_tensor(0.0).detach()
        self.last_edge_symbol_deviation_proxy = features.new_tensor(0.0).detach()
        self.last_local_tube_coordinate = None
        metadata_batch = _metadata_batch(metadata, features)
        delta = self._transport_delta(features, metadata_batch)
        updated_metadata = metadata_batch.clone()
        updated_metadata[..., :4] = updated_metadata[..., :4] + delta

        scaled = updated_metadata[..., :4] * self.length_scale.view(1, 1, 4)
        use_topk = self.stencil_size is not None and 0 < self.stencil_size < tokens

        message_source = self.message_mlp(torch.cat([features, updated_metadata], dim=-1))
        if use_topk and self.sparse_topk:
            chunk_size = min(tokens, 256)
            chunks: list[Tensor] = []
            tube_chunks: list[Tensor] = []
            expanded_source = message_source.unsqueeze(1)
            expanded_metadata = updated_metadata[..., :4].unsqueeze(1)
            for start in range(0, tokens, chunk_size):
                stop = min(start + chunk_size, tokens)
                with torch.no_grad():
                    chunk_scaled = scaled[:, start:stop, :]
                    chunk_logits = -torch.cdist(chunk_scaled, scaled, p=2.0) / self.temperature
                    top_values, top_indices = torch.topk(chunk_logits, k=self.stencil_size, dim=-1)
                    weights = torch.softmax(top_values, dim=-1)
                gathered = torch.gather(
                    expanded_source.expand(-1, stop - start, -1, -1),
                    2,
                    top_indices.unsqueeze(-1).expand(-1, -1, -1, width),
                )
                gathered_metadata = torch.gather(
                    expanded_metadata.expand(-1, stop - start, -1, -1),
                    2,
                    top_indices.unsqueeze(-1).expand(-1, -1, -1, 4),
                )
                target_metadata = updated_metadata[:, start:stop, :4].unsqueeze(2)
                gathered_full_metadata = torch.gather(
                    updated_metadata.unsqueeze(1).expand(-1, stop - start, -1, -1),
                    2,
                    top_indices.unsqueeze(-1).expand(-1, -1, -1, updated_metadata.shape[-1]),
                )
                target_full_metadata = updated_metadata[:, start:stop, :].unsqueeze(2).expand_as(gathered_full_metadata)
                # Executable proxy for nu = h^{-1/2}(z_gamma-kappa(z_eta))
                # on the retained canonical stencil: a weighted local edge
                # coordinate seen by the symbol branch.
                edge_tube = target_metadata - gathered_metadata
                edge_multiplier = self._edge_symbol_multiplier(
                    gathered,
                    gathered_full_metadata,
                    target_full_metadata,
                    edge_tube,
                )
                tube_chunks.append((weights.unsqueeze(-1) * edge_tube).sum(dim=2))
                chunks.append((weights.unsqueeze(-1) * edge_multiplier * gathered).sum(dim=2))
            local_message = torch.cat(chunks, dim=1)
            local_tube_coordinate = torch.cat(tube_chunks, dim=1)
        else:
            distances = torch.cdist(scaled, scaled, p=2.0)
            logits = -distances / self.temperature
            if use_topk:
                top_values, top_indices = torch.topk(logits, k=self.stencil_size, dim=-1)
                masked_logits = torch.full_like(logits, float("-inf"))
                masked_logits.scatter_(-1, top_indices, top_values)
                logits = masked_logits
            weights = torch.softmax(logits, dim=-1)
            source_metadata = updated_metadata[:, None, :, :4]
            target_metadata = updated_metadata[:, :, None, :4]
            edge_tube = target_metadata - source_metadata
            source_features = message_source[:, None, :, :].expand(batch, tokens, tokens, width)
            source_full_metadata = updated_metadata[:, None, :, :].expand(batch, tokens, tokens, updated_metadata.shape[-1])
            target_full_metadata = updated_metadata[:, :, None, :].expand_as(source_full_metadata)
            edge_multiplier = self._edge_symbol_multiplier(
                source_features,
                source_full_metadata,
                target_full_metadata,
                edge_tube,
            )
            local_tube_coordinate = (weights.unsqueeze(-1) * edge_tube).sum(dim=2)
            local_message = (weights.unsqueeze(-1) * edge_multiplier * source_features).sum(dim=2)
        if self.confidence_head is not None:
            confidence_logits = self.confidence_head(torch.cat([features, updated_metadata], dim=-1))
            if self.wavefront_confidence_scale != 0.0:
                freq_norm = torch.linalg.vector_norm(updated_metadata[..., 2:4], dim=-1, keepdim=True)
                freq_baseline = freq_norm.mean(dim=1, keepdim=True).clamp_min(1e-6)
                # High-frequency packets should not be silently solved by the
                # physical-space refinement path; bias their packet transport
                # gate upward and record the bias as a mechanism diagnostic.
                wavefront_bias = self.wavefront_confidence_scale * (freq_norm / freq_baseline - 1.0)
                confidence_logits = confidence_logits + wavefront_bias
                self.last_wavefront_confidence_proxy = wavefront_bias.abs().mean().detach()
            else:
                self.last_wavefront_confidence_proxy = confidence_logits.new_tensor(0.0).detach()
            confidence = torch.sigmoid(confidence_logits)
            local_message = confidence * local_message
        gated = torch.sigmoid(self.output_gate(features))
        output = gated * local_message
        self.last_canonical_defect_proxy = self._canonical_defect_proxy(delta).detach()
        self.last_local_tube_coordinate = _finite_tensor(local_tube_coordinate, cap=10.0)
        return output, updated_metadata


class SymbolModulationLayer(nn.Module):
    def __init__(
        self,
        width: int,
        metadata_dim: int = 5,
        hidden_dim: int = 128,
        symbol_order: float = 0.0,
        symbol_parameterization: str = "mlp",
        helmholtz_shell_radius: float = 0.0,
        helmholtz_refractive_index: float = 1.0,
        helmholtz_absorption: float = 0.05,
        helmholtz_resolvent_cap: float = 6.0,
    ) -> None:
        super().__init__()
        self.symbol_order = float(symbol_order)
        self.symbol_parameterization = symbol_parameterization
        self.helmholtz_shell_radius = float(helmholtz_shell_radius)
        self.helmholtz_refractive_index = float(helmholtz_refractive_index)
        self.helmholtz_absorption = float(helmholtz_absorption)
        self.helmholtz_resolvent_cap = float(helmholtz_resolvent_cap)
        self.local_kernel_enabled = True
        self.helmholtz_symbol_mode = "full"
        if symbol_parameterization == "mlp":
            self.modulation = MLP(width + metadata_dim, hidden_dim, width * 2)
        elif symbol_parameterization == "spectral_order":
            # Metadata-only gates are closer to a packet symbol than a generic
            # feature-conditioned MLP: the learned multiplier varies with
            # phase-space coordinates and radial frequency features, while the
            # order weight below fixes the intended symbolic scaling.  In this
            # theorem-facing mode the action is multiplier-only; additive
            # packet sources belong to residual/landing terms, not to S^m.
            self.modulation = MLP(metadata_dim + 3, hidden_dim, width * 2)
        elif symbol_parameterization == "helmholtz_resolvent":
            # High-k Helmholtz needs a near-characteristic/radiation envelope,
            # not just a radial S^m scaling.  The finite metadata features now
            # expose an auto-calibrated shell |xi| ~= k sqrt(n), signed
            # real/imaginary resolvent weights, and an outgoing-flux gate.
            # The signed multiplier is applied on an explicit latent complex
            # pair, not by arbitrary channel-pair quadrature.
            self.modulation = MLP(metadata_dim + 9, hidden_dim, width * 2)
            self.complex_symbol = ComplexPairSymbolAction(width)
        else:
            raise ValueError(f"Unsupported symbol_parameterization: {symbol_parameterization}")
        self.local_kernel_proj = nn.Linear(4, width * 2, bias=False)
        nn.init.zeros_(self.local_kernel_proj.weight)

    def _order_weight(self, metadata: Tensor, features: Tensor) -> Tensor:
        if self.symbol_order == 0.0:
            return torch.ones(features.shape[0], features.shape[1], 1, device=features.device, dtype=features.dtype)
        metadata_batch = _metadata_batch(metadata, features)
        freq = metadata_batch[..., 2:4]
        freq_norm_sq = freq.square().sum(dim=-1, keepdim=True)
        weight = torch.pow(1.0 + freq_norm_sq, 0.5 * self.symbol_order)
        return _finite_tensor(weight, cap=10.0).clamp(0.0, 10.0)

    @staticmethod
    def _spectral_metadata(metadata: Tensor) -> Tensor:
        freq = metadata[..., 2:4]
        freq_norm = torch.linalg.vector_norm(freq, dim=-1, keepdim=True).clamp_min(1e-8)
        unit_freq = freq / freq_norm
        log_freq = torch.log1p(freq_norm)
        return torch.cat([metadata, log_freq, unit_freq], dim=-1)

    def _helmholtz_metadata(self, metadata: Tensor, features: Tensor | None = None) -> Tensor:
        log_freq, unit_freq, shell_distance, envelope, outgoing_flux, real, imag, outgoing_gate, _ = _helmholtz_shell_features(
            metadata,
            features,
            shell_radius=self.helmholtz_shell_radius,
            refractive_index=self.helmholtz_refractive_index,
            absorption=self.helmholtz_absorption,
            cap=self.helmholtz_resolvent_cap,
        )
        return torch.cat([metadata, log_freq, unit_freq, shell_distance, envelope, outgoing_flux, real, imag, outgoing_gate], dim=-1)

    def _helmholtz_resolvent_weights(self, metadata: Tensor, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        metadata_batch = _metadata_batch(metadata, features)
        _, _, _, _, _, real, imag, outgoing_gate, _ = _helmholtz_shell_features(
            metadata_batch,
            features,
            shell_radius=self.helmholtz_shell_radius,
            refractive_index=self.helmholtz_refractive_index,
            absorption=self.helmholtz_absorption,
            cap=self.helmholtz_resolvent_cap,
        )
        return real, imag, outgoing_gate

    def helmholtz_resolvent_diagnostics(self, metadata: Tensor, features: Tensor) -> dict[str, Tensor]:
        metadata_batch = _metadata_batch(metadata, features)
        _, _, shell_distance, envelope, outgoing_flux, real, imag, outgoing_gate, shell_center = _helmholtz_shell_features(
            metadata_batch,
            features,
            shell_radius=self.helmholtz_shell_radius,
            refractive_index=self.helmholtz_refractive_index,
            absorption=self.helmholtz_absorption,
            cap=self.helmholtz_resolvent_cap,
        )
        return {
            "helmholtz_shell_distance_proxy": shell_distance.abs().mean(),
            "helmholtz_resolvent_envelope_proxy": envelope.mean(),
            "helmholtz_outgoing_flux_proxy": outgoing_flux.mean(),
            "helmholtz_resolvent_real_proxy": real.abs().mean(),
            "helmholtz_resolvent_imag_proxy": imag.abs().mean(),
            "helmholtz_outgoing_gate_proxy": outgoing_gate.mean(),
            "helmholtz_shell_center_proxy": shell_center.mean(),
            "helmholtz_complex_latent_imag_energy_proxy": self.complex_symbol.latent_imaginary_energy(features),
        }

    def forward(self, features: Tensor, metadata: Tensor, local_tube_coordinate: Tensor | None = None) -> Tensor:
        metadata_batch = _metadata_batch(metadata, features)
        if self.symbol_parameterization == "spectral_order":
            symbol_input = self._spectral_metadata(metadata_batch)
            params = self.modulation(symbol_input)
        elif self.symbol_parameterization == "helmholtz_resolvent":
            symbol_input = self._helmholtz_metadata(metadata_batch, features)
            params = self.modulation(symbol_input)
        else:
            params = self.modulation(torch.cat([features, metadata_batch], dim=-1))
        params = _finite_tensor(params, cap=30.0)
        if local_tube_coordinate is not None and self.local_kernel_enabled:
            local_tube_coordinate = _finite_tensor(_metadata_batch(local_tube_coordinate, features), cap=10.0)
            params = params + self.local_kernel_proj(local_tube_coordinate)
        scale, shift = params.chunk(2, dim=-1)
        if self.symbol_parameterization == "spectral_order":
            return self._order_weight(metadata, features) * torch.tanh(scale) * features
        if self.symbol_parameterization == "helmholtz_resolvent":
            real_weight, imag_weight, outgoing_gate = self._helmholtz_resolvent_weights(metadata, features)
            real_multiplier = real_weight * torch.tanh(scale)
            imag_multiplier = imag_weight * torch.tanh(shift)
            mode = getattr(self, "helmholtz_symbol_mode", "full")
            if mode == "amplitude_only":
                imag_multiplier = torch.zeros_like(imag_multiplier)
            elif mode == "phase_only":
                real_multiplier = torch.ones_like(real_multiplier)
            elif mode != "full":
                raise ValueError(f"Unsupported helmholtz_symbol_mode: {mode}")
            symbol_action = self.complex_symbol(features, real_multiplier, imag_multiplier)
            return _finite_tensor(self._order_weight(metadata, features) * outgoing_gate * symbol_action, cap=1.0e6)
        return self._order_weight(metadata, features) * (torch.tanh(scale) * features + shift)


class IdentityPseudodifferentialBranch(nn.Module):
    """Packet-local symbol action for the identity canonical relation."""

    def __init__(
        self,
        width: int,
        metadata_dim: int = 5,
        hidden_dim: int = 128,
        symbol_order: float = -2.0,
        symbol_parameterization: str = "mlp",
        helmholtz_shell_radius: float = 0.0,
        helmholtz_refractive_index: float = 1.0,
        helmholtz_absorption: float = 0.05,
        helmholtz_resolvent_cap: float = 6.0,
    ) -> None:
        super().__init__()
        self.symbol_order = float(symbol_order)
        self.symbol_parameterization = symbol_parameterization
        self.helmholtz_shell_radius = float(helmholtz_shell_radius)
        self.helmholtz_refractive_index = float(helmholtz_refractive_index)
        self.helmholtz_absorption = float(helmholtz_absorption)
        self.helmholtz_resolvent_cap = float(helmholtz_resolvent_cap)
        self.helmholtz_symbol_mode = "full"
        if symbol_parameterization == "mlp":
            self.modulation = MLP(width + metadata_dim, hidden_dim, width * 2)
        elif symbol_parameterization == "spectral_order":
            self.modulation = MLP(metadata_dim + 3, hidden_dim, width * 2)
        elif symbol_parameterization == "helmholtz_resolvent":
            self.modulation = MLP(metadata_dim + 9, hidden_dim, width * 2)
            self.complex_symbol = ComplexPairSymbolAction(width)
        else:
            raise ValueError(f"Unsupported symbol_parameterization: {symbol_parameterization}")

    def _order_weight(self, metadata: Tensor, features: Tensor) -> Tensor:
        metadata_batch = _metadata_batch(metadata, features)
        freq = metadata_batch[..., 2:4]
        freq_norm_sq = freq.square().sum(dim=-1, keepdim=True)
        weight = torch.pow(1.0 + freq_norm_sq, 0.5 * self.symbol_order)
        return _finite_tensor(weight, cap=10.0).clamp(0.0, 10.0)

    def forward(self, features: Tensor, metadata: Tensor) -> Tensor:
        metadata_batch = _metadata_batch(metadata, features)
        if self.symbol_parameterization == "spectral_order":
            symbol_input = SymbolModulationLayer._spectral_metadata(metadata_batch)
            params = self.modulation(symbol_input)
        elif self.symbol_parameterization == "helmholtz_resolvent":
            log_freq, unit_freq, shell_distance, envelope, outgoing_flux, real, imag, outgoing_gate, _ = _helmholtz_shell_features(
                metadata_batch,
                features,
                shell_radius=self.helmholtz_shell_radius,
                refractive_index=self.helmholtz_refractive_index,
                absorption=self.helmholtz_absorption,
                cap=self.helmholtz_resolvent_cap,
            )
            symbol_input = torch.cat([metadata_batch, log_freq, unit_freq, shell_distance, envelope, outgoing_flux, real, imag, outgoing_gate], dim=-1)
            params = self.modulation(symbol_input)
        else:
            params = self.modulation(torch.cat([features, metadata_batch], dim=-1))
        params = _finite_tensor(params, cap=30.0)
        scale, shift = params.chunk(2, dim=-1)
        order_weight = self._order_weight(metadata, features)
        if self.symbol_parameterization == "spectral_order":
            return order_weight * torch.tanh(scale) * features
        if self.symbol_parameterization == "helmholtz_resolvent":
            _, _, _, _, _, real_weight, imag_weight, outgoing_gate, _ = _helmholtz_shell_features(
                metadata_batch,
                features,
                shell_radius=self.helmholtz_shell_radius,
                refractive_index=self.helmholtz_refractive_index,
                absorption=self.helmholtz_absorption,
                cap=self.helmholtz_resolvent_cap,
            )
            real_multiplier = real_weight * torch.tanh(scale)
            imag_multiplier = imag_weight * torch.tanh(shift)
            mode = getattr(self, "helmholtz_symbol_mode", "full")
            if mode == "amplitude_only":
                imag_multiplier = torch.zeros_like(imag_multiplier)
            elif mode == "phase_only":
                real_multiplier = torch.ones_like(real_multiplier)
            elif mode != "full":
                raise ValueError(f"Unsupported helmholtz_symbol_mode: {mode}")
            symbol_action = self.complex_symbol(features, real_multiplier, imag_multiplier)
            return _finite_tensor(order_weight * outgoing_gate * symbol_action, cap=1.0e6)
        return order_weight * (torch.tanh(scale) * features + shift)


class DissipativePseudodifferentialBranch(nn.Module):
    """Dissipative packet-local symbol action for parabolic regimes."""

    def __init__(
        self,
        width: int,
        metadata_dim: int = 5,
        hidden_dim: int = 128,
        time_step: float = 1.0,
    ) -> None:
        super().__init__()
        self.time_step = float(time_step)
        self.damping = MLP(width + metadata_dim, hidden_dim, width)

    def forward(self, features: Tensor, metadata: Tensor) -> tuple[Tensor, Tensor]:
        metadata_batch = _metadata_batch(metadata, features)
        raw = self.damping(torch.cat([features, metadata_batch], dim=-1))
        multiplier = torch.exp(-self.time_step * torch.nn.functional.softplus(raw))
        return (multiplier - 1.0) * features, multiplier


class SpectralResidualBlock(nn.Module):
    def __init__(self, channels: int, modes_height: int = 12, modes_width: int = 12) -> None:
        super().__init__()
        self.channels = channels
        self.modes_height = modes_height
        self.modes_width = modes_width
        weight = torch.zeros(channels, channels, modes_height, modes_width, dtype=torch.cfloat)
        self.weight = nn.Parameter(weight)

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        spectrum = torch.fft.rfft2(x, norm="ortho")
        out = torch.zeros_like(spectrum)
        mh = min(self.modes_height, spectrum.shape[-2])
        mw = min(self.modes_width, spectrum.shape[-1])
        out[:, :, :mh, :mw] = torch.einsum("bcij,ocij->boij", spectrum[:, :, :mh, :mw], self.weight[:, :, :mh, :mw])
        return torch.fft.irfft2(out, s=(height, width), norm="ortho")
