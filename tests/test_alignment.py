import numpy as np

from rpa.transfer_learning.alignment import (
    center_covariances,
    recolor_covariances,
    align_mean_to_reference,
    dispersion,
    stretch_covariances,
    class_means,
    rotate_covariances,
)

from rpa.core.riemann_base import riemannian_mean
from rpa.core.spd_matrices import is_spd


def test_center_covariances_output_shape():

    covariances = np.array([
        [
            [2.0, 0.2],
            [0.2, 3.0],
        ],
        [
            [4.0, 0.5],
            [0.5, 5.0],
        ],
        [
            [3.0, 0.1],
            [0.1, 6.0],
        ],
    ])

    centered, mean = center_covariances(covariances)

    assert centered.shape == covariances.shape
    assert mean.shape == covariances.shape[1:]


def test_centered_covariances_are_spd():

    covariances = np.array([
        [
            [2.0, 0.2],
            [0.2, 3.0],
        ],
        [
            [4.0, 0.5],
            [0.5, 5.0],
        ],
        [
            [3.0, 0.1],
            [0.1, 6.0],
        ],
    ])

    centered, _ = center_covariances(covariances)

    for matrix in centered:
        assert is_spd(matrix)


def test_centered_mean_is_identity():

    covariances = np.array([
        [
            [2.0, 0.2],
            [0.2, 3.0],
        ],
        [
            [4.0, 0.5],
            [0.5, 5.0],
        ],
        [
            [3.0, 0.1],
            [0.1, 6.0],
        ],
    ])

    centered, _ = center_covariances(covariances)

    centered_mean = riemannian_mean(centered)

    assert np.allclose(centered_mean, np.eye(2), atol=1e-6)


def test_recolor_covariances_output_shape():

    centered_covariances = np.array([
        [
            [1.0, 0.1],
            [0.1, 1.5],
        ],
        [
            [1.2, 0.2],
            [0.2, 1.8],
        ],
    ])

    target_mean = np.array([
        [2.0, 0.3],
        [0.3, 3.0],
    ])

    recolored = recolor_covariances(
        centered_covariances,
        target_mean,
    )

    assert recolored.shape == centered_covariances.shape


def test_recolored_covariances_are_spd():

    centered_covariances = np.array([
        [
            [1.0, 0.1],
            [0.1, 1.5],
        ],
        [
            [1.2, 0.2],
            [0.2, 1.8],
        ],
    ])

    target_mean = np.array([
        [2.0, 0.3],
        [0.3, 3.0],
    ])

    recolored = recolor_covariances(
        centered_covariances,
        target_mean,
    )

    for matrix in recolored:
        assert is_spd(matrix)


def test_recolor_identity_gives_target_mean():

    centered_covariances = np.array([
        np.eye(2),
    ])

    target_mean = np.array([
        [2.0, 0.3],
        [0.3, 3.0],
    ])

    recolored = recolor_covariances(
        centered_covariances,
        target_mean,
    )

    assert np.allclose(
        recolored[0],
        target_mean,
        atol=1e-8,
    )


def test_center_then_recolor_mean_matches_target_mean():

    source_covariances = np.array([
        [
            [2.0, 0.2],
            [0.2, 3.0],
        ],
        [
            [4.0, 0.5],
            [0.5, 5.0],
        ],
        [
            [3.0, 0.1],
            [0.1, 6.0],
        ],
    ])

    target_covariances = np.array([
        [
            [6.0, 0.4],
            [0.4, 7.0],
        ],
        [
            [8.0, 0.6],
            [0.6, 9.0],
        ],
        [
            [7.0, 0.3],
            [0.3, 10.0],
        ],
    ])

    centered_source, _ = center_covariances(source_covariances)

    target_mean = riemannian_mean(target_covariances)

    recolored_source = recolor_covariances(
        centered_source,
        target_mean,
    )

    recolored_mean = riemannian_mean(recolored_source)

    assert np.allclose(
        recolored_mean,
        target_mean,
        atol=1e-6,
    )


def test_align_mean_to_reference_output_shape():

    covariances = np.array([
        [
            [2.0, 0.2],
            [0.2, 3.0],
        ],
        [
            [4.0, 0.5],
            [0.5, 5.0],
        ],
    ])

    reference_covariances = np.array([
        [
            [6.0, 0.4],
            [0.4, 7.0],
        ],
        [
            [8.0, 0.6],
            [0.6, 9.0],
        ],
    ])

    aligned, covariance_mean, reference_mean = align_mean_to_reference(
        covariances,
        reference_covariances,
    )

    assert aligned.shape == covariances.shape
    assert covariance_mean.shape == covariances.shape[1:]
    assert reference_mean.shape == reference_covariances.shape[1:]


def test_align_mean_to_reference_outputs_spd():

    covariances = np.array([
        [
            [2.0, 0.2],
            [0.2, 3.0],
        ],
        [
            [4.0, 0.5],
            [0.5, 5.0],
        ],
        [
            [3.0, 0.1],
            [0.1, 6.0],
        ],
    ])

    reference_covariances = np.array([
        [
            [6.0, 0.4],
            [0.4, 7.0],
        ],
        [
            [8.0, 0.6],
            [0.6, 9.0],
        ],
        [
            [7.0, 0.3],
            [0.3, 10.0],
        ],
    ])

    aligned, _, _ = align_mean_to_reference(
        covariances,
        reference_covariances,
    )

    for matrix in aligned:
        assert is_spd(matrix)


def test_align_mean_to_reference_mean_matches_reference_mean():

    covariances = np.array([
        [
            [2.0, 0.2],
            [0.2, 3.0],
        ],
        [
            [4.0, 0.5],
            [0.5, 5.0],
        ],
        [
            [3.0, 0.1],
            [0.1, 6.0],
        ],
    ])

    reference_covariances = np.array([
        [
            [6.0, 0.4],
            [0.4, 7.0],
        ],
        [
            [8.0, 0.6],
            [0.6, 9.0],
        ],
        [
            [7.0, 0.3],
            [0.3, 10.0],
        ],
    ])

    aligned, _, reference_mean = align_mean_to_reference(
        covariances,
        reference_covariances,
    )

    aligned_mean = riemannian_mean(aligned)

    assert np.allclose(
        aligned_mean,
        reference_mean,
        atol=1e-6,
    )


def test_dispersion_is_non_negative():

    covariances = np.array([
        [
            [2.0, 0.2],
            [0.2, 3.0],
        ],
        [
            [4.0, 0.5],
            [0.5, 5.0],
        ],
    ])

    d = dispersion(covariances)

    assert d >= 0.0


def test_dispersion_of_identical_matrices_is_zero():

    A = np.array([
        [2.0, 0.2],
        [0.2, 3.0],
    ])

    covariances = np.array([A, A, A])

    d = dispersion(covariances)

    assert np.isclose(d, 0.0, atol=1e-8)


def test_stretch_covariances_scale_one_returns_same():

    covariances = np.array([
        [
            [2.0, 0.2],
            [0.2, 3.0],
        ],
        [
            [4.0, 0.5],
            [0.5, 5.0],
        ],
    ])

    stretched = stretch_covariances(covariances, scale=1.0)

    assert np.allclose(stretched, covariances, atol=1e-8)


def test_stretch_covariances_scale_zero_returns_identity():

    covariances = np.array([
        [
            [2.0, 0.2],
            [0.2, 3.0],
        ],
        [
            [4.0, 0.5],
            [0.5, 5.0],
        ],
    ])

    stretched = stretch_covariances(covariances, scale=0.0)

    expected = np.array([
        np.eye(2),
        np.eye(2),
    ])

    assert np.allclose(stretched, expected, atol=1e-8)


def test_class_means_returns_one_mean_per_class():

    covariances = np.array([
        [
            [2.0, 0.2],
            [0.2, 3.0],
        ],
        [
            [3.0, 0.1],
            [0.1, 4.0],
        ],
        [
            [6.0, 0.3],
            [0.3, 7.0],
        ],
        [
            [7.0, 0.2],
            [0.2, 8.0],
        ],
    ])

    labels = np.array([0, 0, 1, 1])

    means = class_means(covariances, labels)

    assert set(means.keys()) == {0, 1}
    assert means[0].shape == (2, 2)
    assert means[1].shape == (2, 2)


def test_class_means_outputs_spd():

    covariances = np.array([
        [
            [2.0, 0.2],
            [0.2, 3.0],
        ],
        [
            [3.0, 0.1],
            [0.1, 4.0],
        ],
        [
            [6.0, 0.3],
            [0.3, 7.0],
        ],
        [
            [7.0, 0.2],
            [0.2, 8.0],
        ],
    ])

    labels = np.array([0, 0, 1, 1])

    means = class_means(covariances, labels)

    for mean in means.values():
        assert is_spd(mean)


def test_rotate_covariances_identity_rotation():

    covariances = np.array([
        [
            [2.0, 0.2],
            [0.2, 3.0],
        ],
        [
            [4.0, 0.5],
            [0.5, 5.0],
        ],
    ])

    U = np.eye(2)

    rotated = rotate_covariances(covariances, U)

    assert np.allclose(rotated, covariances, atol=1e-8)


def test_rotate_covariances_outputs_spd():

    covariances = np.array([
        [
            [2.0, 0.2],
            [0.2, 3.0],
        ],
        [
            [4.0, 0.5],
            [0.5, 5.0],
        ],
    ])

    U = np.array([
        [0.0, -1.0],
        [1.0, 0.0],
    ])

    rotated = rotate_covariances(covariances, U)

    for matrix in rotated:
        assert is_spd(matrix)