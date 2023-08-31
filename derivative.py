"""Spatial derivatives modules."""

import enum

import numpy as np
import torch
import torch.nn as nn

# Coefficients mapping [method][derivative-order][accuracy-order]
# https://en.wikipedia.org/wiki/Finite_difference_coefficient
FINITE_DIFFERENCES_COEFFICIENTS = {
    'FORWARD': {
        1: {
            1: (-1, 1),
            2: (-3 / 2, 2, -1 / 2),
        },
        2: {
            1: (1, -2, 1),
            2: (2, -5, 4, -1),
        }
    },
    'BACKWARD': {
        1: {
            1: (1, -1),
            2: (3 / 2, -2, 1 / 2),
        },
        2: {
            1: (1, -2, 1),
            2: (2, -5, 4, -1),
        }
    },
    'CENTRAL': {
        1: {
            2: (-1 / 2, 0, 1 / 2),
            4: (1 / 12, -2 / 3, 0, 2 / 3, -1 / 12),
        },
        2: {
            2: (1, -2, 1),
            4: (-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12),
        }
    }
}


class FiniteDifferencesMethod(enum.Enum):
    FORWARD = 1
    BACKWARD = 2
    CENTRAL = 3


class Axis(enum.Enum):
    X = 1
    Y = 0


class Derivative(nn.Module):
    """Base class for spatial derivative."""

    def __init__(self):
        super(Derivative, self).__init__()

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self.forward(*args, **kwargs)

    def forward(self, u: torch.Tensor, axis: Axis) -> torch.Tensor:
        """Applies derivative to tensor u."""
        raise NotImplementedError


class FiniteDifferences(Derivative):
    """Finite differences derivatives."""

    def __init__(self, method: FiniteDifferencesMethod,
                 derivative_order: int, accuracy_order: int):
        super(FiniteDifferences, self).__init__()
        self.method = method
        # TODO(ngiladi): assert values before referencing mapping
        self.coefficients = FINITE_DIFFERENCES_COEFFICIENTS[
            method.name.upper()][derivative_order][accuracy_order]

    def forward(self, u: torch.Tensor, axis: Axis) -> torch.Tensor:
        derivative = torch.zeros_like(u)
        if self.method is FiniteDifferencesMethod.FORWARD:
            for shift, coefficient in enumerate(self.coefficients):
                derivative += coefficient * torch.roll(u, -shift, axis.value)
        if self.method is FiniteDifferencesMethod.BACKWARD:
            for shift, coefficient in enumerate(self.coefficients):
                derivative += coefficient * torch.roll(u, shift, axis.value)
        if self.method is FiniteDifferencesMethod.CENTRAL:
            stencil = np.arange(-(len(self.coefficients) // 2),
                                (len(self.coefficients) // 2) + 1)
            for shift, coefficient in zip(stencil, self.coefficients):
                derivative += coefficient * torch.roll(u, shift, axis.value)
        return derivative


class FiniteVolumes(Derivative):
    """Finite differences derivatives."""
    # TODO(ngiladi): implement

    def __init__(self):
        super(FiniteVolumes, self).__init__()
        raise NotImplementedError('Not implemented yet')

    def forward(self, u: torch.Tensor, axis: Axis) -> torch.Tensor:
        raise NotImplementedError('Not implemented yet')
