"""Partial differential equations modules."""

import torch
import torch.nn as nn

import derivative


class Equation(nn.Module):
    """Base class for a PDE. Stateless"""

    def __init__(self):
        super().__init__()

    def step(self, *args, **kwargs) -> torch.Tensor:
        """Performs one numerical step in time."""
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Performs one numerical step in time."""
        raise NotImplementedError


class HeatEquation1D(Equation):
    """Heat (diffusion) equation. """

    def __init__(self, alpha: float):
        super(HeatEquation1D, self).__init__()
        self.d_2_d_x = derivative.FiniteDifferences(
            derivative.FiniteDifferencesMethod.CENTRAL, 2, 2)
        self.alpha = alpha

    def forward(self, u: torch.Tensor, dx: float, dt: float):
        """∂u/∂t = ∇^2(u)"""
        # TODO(ngiladi): using implicitly periodic boundary conditions
        return u + dt * self.alpha * (
                self.d_2_d_x(u, derivative.Axis.X) / (dx ** 2))
