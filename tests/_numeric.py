"""Comparisons to frozen reference outputs, computed on another machine."""

import torch


def assert_f32_close(got, want, msg=None, padding=None) -> None:
    """Float32 results against frozen ones: equal to 1e-5 of the reference's largest
    value (`padding` placeholders left out of that). The CPU (its SIMD kernels, the
    thread count) moves the last bits of resampling and of matrix products, so
    these are not compared bit for bit."""
    values = want if padding is None else want[want != padding]
    atol = 1e-5 * float(values.abs().max()) if values.numel() else 0.0
    torch.testing.assert_close(got, want, rtol=1e-5, atol=atol, msg=msg)
