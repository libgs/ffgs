"""Processor building blocks against the default data path.

Frozen reference outputs of the default data path (scale to cover, centre crop,
first-view anchoring, scene_scale) are in `fixtures/`; `anchored_frame` +
`rescale_and_crop`, as a processor chains them, must reproduce them (to float32
precision: the CPU moves the last bits).
"""

import pytest
import torch
from _dummy import DummyProcessor
from _numeric import assert_f32_close

from ffgs import Views


def _views(reference, indices) -> Views:
    idx = torch.tensor(indices)
    return Views(
        reference["dataset.frames.images"][idx],
        reference["dataset.frames.intrinsics"][idx],
        reference["dataset.frames.c2w"][idx],
        normalized_intrinsics=True,
    )


@pytest.mark.parametrize("resize_mode", ["bilinear", "lanczos"])
def test_preprocess_matches_the_reference(reference, resize_mode) -> None:
    case = reference.meta("dataset_case")
    processor = DummyProcessor(
        image_shape=case["input_image_shape"],
        resize_mode=resize_mode,
        scene_scale=case["scene_scale"],
    )
    context = _views(reference, case["context"])
    prepared = processor.preprocess(context, torch.device("cpu"))
    want = f"dataset.{resize_mode}"

    fitted = processor.fit_views(context)
    assert_f32_close(fitted.images[0], reference[f"{want}.context.image"])
    assert_f32_close(fitted.intrinsics[0], reference[f"{want}.context.intrinsics"])
    assert_f32_close(
        prepared.model_input["images"][0], reference[f"{want}.context.image"]
    )
    assert_f32_close(
        prepared.model_input["c2w"][0], reference[f"{want}.context.extrinsics"]
    )

    # Target views: the same fit + the context's frame give the dataset's GT.
    target = processor.fit_views(_views(reference, case["target"]))
    assert_f32_close(target.images[0], reference[f"{want}.target.image"])
    assert_f32_close(target.intrinsics[0], reference[f"{want}.target.intrinsics"])
    model_c2w = prepared.frame.c2w_to_model(target.c2w)
    assert_f32_close(model_c2w[0], reference[f"{want}.target.extrinsics"])


def test_scene_scale_is_a_per_call_option(reference) -> None:
    processor = DummyProcessor(scene_scale=0.15)
    views = _views(reference, [0, 1, 2])
    cpu = torch.device("cpu")
    assert processor.preprocess(views, cpu).frame.scale == 0.15
    assert processor.preprocess(views, cpu, scene_scale=0.25).frame.scale == 0.25


def test_pixel_and_normalised_intrinsics_are_equivalent(reference) -> None:
    views = _views(reference, [0, 1])
    h, w = views.image_shape
    pixel = views.normalized_k.clone()
    pixel[..., 0, :] *= w
    pixel[..., 1, :] *= h
    as_pixel = Views(views.images, pixel, views.c2w)
    torch.testing.assert_close(as_pixel.normalized_k, views.normalized_k)
    torch.testing.assert_close(as_pixel.cameras.pixel_k, pixel)


@pytest.mark.parametrize("crop_mode", ["crop", "pad"])
def test_fit_cameras_matches_fit_views(reference, crop_mode) -> None:
    processor = DummyProcessor(image_shape=[16, 16], crop_mode=crop_mode)
    views = _views(reference, [0, 1, 2])
    fitted = processor.fit_views(views)
    cameras = processor.fit_cameras(views.cameras)
    assert cameras.image_shape == fitted.image_shape == (16, 16)
    assert torch.equal(cameras.intrinsics, fitted.intrinsics)


def test_processor_crop_modes_and_custom_resize(reference) -> None:
    views = _views(reference, [0, 1])
    as_is = DummyProcessor(image_shape=list(views.image_shape), crop_mode="none")
    assert torch.equal(as_is.fit_views(views).images, views.images)
    with pytest.raises(ValueError, match="crop_mode"):
        DummyProcessor(crop_mode="zoom")

    def flat(images: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
        return images.new_full((*images.shape[:-2], *shape), 0.5)

    custom = DummyProcessor(image_shape=[8, 16], resize=flat)
    assert custom.resize is flat
    assert (custom.fit_views(views).images == 0.5).all()
    # The callable is not part of the serialised config.
    assert "resize" not in custom.to_dict()
