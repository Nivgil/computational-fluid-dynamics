"""PDE solver modules."""

import equation
import torch
import numpy as np
import derivative

import matplotlib.pyplot as plt


def solve(u_0: torch.Tensor, dx: float, dt: float, eq: equation.Equation,
          time_to_simulate: float, save_state_freq: int) -> list[torch.Tensor]:
    # TODO(ngiladi): stopping metric should be used here
    # TODO(ngiladi): maybe return torch.stack(solution)
    current_t = 0
    step = 0
    solution = [u_0]
    while current_t < time_to_simulate:
        next_u = eq.step(u_0, dx, dt)
        step += 1
        current_t += dt
        if save_state_freq % step == 0:
            solution.append(next_u)
    return solution


if __name__ == '__main__':
    print('mini test for solve.py')
    eq = equation.HeatEquation1D(alpha=5)
    n_x = 128
    x_length = 1  # 10 meters
    d_x = x_length / n_x
    d_t = d_x / 2
    counts, bins = np.histogram(torch.normal(0., 1., (10000,)), bins=n_x)
    u_0 = torch.tensor(counts).reshape(1, -1) / 1_000
    print('n_x', n_x, 'x_length', x_length, 'd_x', d_x, 'd_t', d_t)
    print('u_0', u_0)
    plt.figure()
    plt.plot(u_0.flatten().numpy(), label='u_0')

    solution = solve(u_0, d_x, d_t, eq, 100_000, 10)
    print('solution', solution[-1])
    plt.plot(solution[-1].flatten().numpy(), label='u_final')
    plt.legend()
    plt.show()
