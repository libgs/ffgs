"""`ffgs.image`: the default against frozen reference outputs, and the "pad" /
"none" / custom-resize alternatives."""

import pytest
import torch
import torch.nn.functional as tnf
from _numeric import assert_f32_close
from torchvision.transforms.v2.functional import to_dtype

from ffgs import image


def _crop_case(reference, case: int):
    images = to_dtype(reference[f"crop.{case}.images"], torch.float32, scale=True)
    shape = tuple(reference.meta("crop_cases")[case]["shape_out"])
    return images, reference[f"crop.{case}.intrinsics"], shape


def _k(n: int = 2) -> torch.Tensor:
    k = torch.eye(3).repeat(n, 1, 1)
    k[:, 0, 0], k[:, 1, 1] = 0.9, 1.4
    k[:, 0, 2], k[:, 1, 2] = 0.47, 0.53
    return k


@pytest.mark.parametrize("resize_mode", ["bilinear", "lanczos"])
@pytest.mark.parametrize("case", [0, 1, 2, 3])
def test_default_fit_matches_the_reference(reference, resize_mode, case) -> None:
    images, intrinsics, shape = _crop_case(reference, case)
    want_images = reference[f"crop.{case}.{resize_mode}.images"]
    want_k = reference[f"crop.{case}.{resize_mode}.intrinsics"]
    for got_images, got_k in (
        image.rescale_and_crop(images, intrinsics, shape, resize_mode=resize_mode),
        image.fit_images(images, intrinsics, shape, resize_mode=resize_mode),
    ):
        assert_f32_close(got_images, want_images)
        assert_f32_close(got_k, want_k)
    k_only = image.fit_intrinsics(intrinsics, tuple(images.shape[-2:]), shape)
    assert_f32_close(k_only, want_k)


def test_resize_modes_match_the_reference(reference) -> None:
    assert list(image.RESIZE_MODES) == reference.meta("resize_modes")


def test_unknown_modes_are_rejected() -> None:
    with pytest.raises(ValueError, match="resize_mode"):
        image.rescale(torch.rand(3, 8, 8), (4, 4), resize_mode="nearest")
    with pytest.raises(ValueError, match="crop_mode"):
        image.fit_images(torch.rand(3, 8, 8), torch.eye(3), (4, 4), crop_mode="zoom")


def test_none_passes_inputs_through() -> None:
    images, k = torch.rand(2, 3, 8, 12), _k()
    got_images, got_k = image.fit_images(images, k, (8, 12), crop_mode="none")
    assert got_images is images and torch.equal(got_k, k)
    assert torch.equal(image.fit_intrinsics(k, (8, 12), (8, 12), "none"), k)
    with pytest.raises(ValueError, match="model shape"):
        image.fit_images(images, k, (8, 16), crop_mode="none")


@pytest.mark.parametrize(
    "shape_in,shape_out,scaled",
    [
        ((12, 24), (12, 12), (6, 12)),
        ((20, 10), (8, 12), (8, 4)),
        ((9, 15), (15, 25), (15, 25)),
    ],
)
def test_pad_keeps_content_and_moves_the_principal_point(
    shape_in, shape_out, scaled
) -> None:
    images, k = torch.rand(2, 3, *shape_in), _k()
    fill = 0.25
    padded, got_k = image.fit_images(
        images, k, shape_out, resize_mode="bilinear", crop_mode="pad", pad_value=fill
    )
    assert padded.shape == (2, 3, *shape_out)
    (h_out, w_out), (h_s, w_s) = shape_out, scaled
    top, left = (h_out - h_s) // 2, (w_out - w_s) // 2

    # Content: the whole image, resized, centred; the border is `pad_value`.
    content = padded[..., top : top + h_s, left : left + w_s]
    assert torch.equal(content, image.rescale(images, scaled, "bilinear"))
    border = torch.ones(shape_out, dtype=torch.bool)
    border[top : top + h_s, left : left + w_s] = False
    assert (padded[..., border] == fill).all()

    # K: a 3D point lands on the same content pixel before and after padding.
    points = torch.randn(50, 3)
    points[:, 2] = points[:, 2].abs() + 1

    def project(kn: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
        uv = (points / points[:, 2:]) @ kn[0].T  # normalised image coordinates
        return uv[:, :2] * torch.tensor([hw[1], hw[0]])  # pixels

    want = project(k, scaled) + torch.tensor([left, top])
    torch.testing.assert_close(project(got_k, shape_out), want)
    torch.testing.assert_close(
        image.fit_intrinsics(k, shape_in, shape_out, "pad"), got_k
    )


@pytest.mark.parametrize("crop_mode", ["crop", "pad"])
def test_custom_resize_replaces_the_kernel(crop_mode) -> None:
    calls = []

    def nearest(images: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
        calls.append(shape)
        return tnf.interpolate(images, size=shape, mode="nearest")

    images, k = torch.rand(2, 3, 20, 28), _k()
    got, got_k = image.fit_images(
        images, k, (12, 12), crop_mode=crop_mode, resize=nearest
    )
    scaled = image.scaled_shape((20, 28), (12, 12), crop_mode)
    assert calls == [scaled]
    resized = nearest(images, scaled)
    if crop_mode == "crop":
        want, want_k = image.center_crop(resized, k, (12, 12))
    else:
        want, want_k = resized.new_zeros(2, 3, 12, 12), None
        top, left = (12 - scaled[0]) // 2, (12 - scaled[1]) // 2
        want[..., top : top + scaled[0], left : left + scaled[1]] = resized
        want_k = image.fit_intrinsics(k, (20, 28), (12, 12), "pad")
    assert torch.equal(got, want)
    assert torch.equal(got_k, want_k)
