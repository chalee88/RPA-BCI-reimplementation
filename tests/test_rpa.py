import numpy as np

from rpa import RPA


def test_rpa_fit_stores_learned_parameters():

    source_covariances = np.array([
        [
            [2.0, 0.0],
            [0.0, 2.0],
        ],
        [
            [2.2, 0.0],
            [0.0, 2.2],
        ],
        [
            [8.0, 0.0],
            [0.0, 8.0],
        ],
        [
            [8.2, 0.0],
            [0.0, 8.2],
        ],
    ])

    source_labels = np.array([0, 0, 1, 1])

    target_labeled_covariances = np.array([
        [
            [3.0, 0.0],
            [0.0, 3.0],
        ],
        [
            [9.0, 0.0],
            [0.0, 9.0],
        ],
    ])

    target_labeled_labels = np.array([0, 1])

    model = RPA()

    model.fit(
        source_covariances,
        source_labels,
        target_labeled_covariances,
        target_labeled_labels,
    )

    assert model.source_mean_ is not None
    assert model.target_mean_ is not None
    assert model.source_dispersion_ is not None
    assert model.target_dispersion_ is not None
    assert model.scale_ is not None
    assert model.U_ is not None
    assert model.classifier_ is not None


def test_rpa_transform_target_output_shape():

    source_covariances = np.array([
        [
            [2.0, 0.0],
            [0.0, 2.0],
        ],
        [
            [2.2, 0.0],
            [0.0, 2.2],
        ],
        [
            [8.0, 0.0],
            [0.0, 8.0],
        ],
        [
            [8.2, 0.0],
            [0.0, 8.2],
        ],
    ])

    source_labels = np.array([0, 0, 1, 1])

    target_labeled_covariances = np.array([
        [
            [3.0, 0.0],
            [0.0, 3.0],
        ],
        [
            [9.0, 0.0],
            [0.0, 9.0],
        ],
    ])

    target_labeled_labels = np.array([0, 1])

    target_unlabeled_covariances = np.array([
        [
            [3.1, 0.0],
            [0.0, 3.1],
        ],
        [
            [9.1, 0.0],
            [0.0, 9.1],
        ],
    ])

    model = RPA()

    model.fit(
        source_covariances,
        source_labels,
        target_labeled_covariances,
        target_labeled_labels,
    )

    transformed = model.transform_target(target_unlabeled_covariances)

    assert transformed.shape == target_unlabeled_covariances.shape


def test_rpa_predict_output_shape():

    source_covariances = np.array([
        [
            [2.0, 0.0],
            [0.0, 2.0],
        ],
        [
            [2.2, 0.0],
            [0.0, 2.2],
        ],
        [
            [8.0, 0.0],
            [0.0, 8.0],
        ],
        [
            [8.2, 0.0],
            [0.0, 8.2],
        ],
    ])

    source_labels = np.array([0, 0, 1, 1])

    target_labeled_covariances = np.array([
        [
            [3.0, 0.0],
            [0.0, 3.0],
        ],
        [
            [9.0, 0.0],
            [0.0, 9.0],
        ],
    ])

    target_labeled_labels = np.array([0, 1])

    target_unlabeled_covariances = np.array([
        [
            [3.1, 0.0],
            [0.0, 3.1],
        ],
        [
            [9.1, 0.0],
            [0.0, 9.1],
        ],
    ])

    model = RPA()

    model.fit(
        source_covariances,
        source_labels,
        target_labeled_covariances,
        target_labeled_labels,
    )

    predictions = model.predict(target_unlabeled_covariances)

    assert predictions.shape == (2,)


def test_rpa_predict_before_fit_raises_error():

    target_covariances = np.array([
        [
            [3.1, 0.0],
            [0.0, 3.1],
        ],
    ])

    model = RPA()

    try:
        model.predict(target_covariances)
        assert False
    except RuntimeError:
        assert True