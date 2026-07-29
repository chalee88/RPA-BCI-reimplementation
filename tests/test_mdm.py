import numpy as np

from rpa.classification import MDM
from rpa.core.spd_matrices import is_spd


def test_mdm_fit_stores_classes():

    covariances = np.array([
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

    labels = np.array([0, 0, 1, 1])

    clf = MDM()
    clf.fit(covariances, labels)

    assert set(clf.classes_) == {0, 1}


def test_mdm_fit_stores_class_means(): # checks if the classifier remembers the class labels

    covariances = np.array([
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

    labels = np.array([0, 0, 1, 1])

    clf = MDM()
    clf.fit(covariances, labels)

    assert set(clf.class_means_.keys()) == {0, 1}

    for mean in clf.class_means_.values():
        assert mean.shape == (2, 2)
        assert is_spd(mean)


def test_mdm_predict_returns_correct_shape(): # check if you give two test samples, you get two predictions

    train_covariances = np.array([
        [
            [2.0, 0.0],
            [0.0, 2.0],
        ],
        [
            [8.0, 0.0],
            [0.0, 8.0],
        ],
    ])

    train_labels = np.array([0, 1])

    test_covariances = np.array([
        [
            [2.1, 0.0],
            [0.0, 2.1],
        ],
        [
            [8.1, 0.0],
            [0.0, 8.1],
        ],
    ])

    clf = MDM()
    clf.fit(train_covariances, train_labels)

    predictions = clf.predict(test_covariances)

    assert predictions.shape == (2,)


def test_mdm_predict_simple_case(): # 

    train_covariances = np.array([
        [
            [2.0, 0.0],
            [0.0, 2.0],
        ],
        [
            [8.0, 0.0],
            [0.0, 8.0],
        ],
    ])

    train_labels = np.array([0, 1])

    test_covariances = np.array([
        [
            [2.1, 0.0],
            [0.0, 2.1],
        ],
        [
            [8.1, 0.0],
            [0.0, 8.1],
        ],
    ])

    clf = MDM()
    clf.fit(train_covariances, train_labels)

    predictions = clf.predict(test_covariances)

    assert np.array_equal(predictions, np.array([0, 1]))


def test_mdm_predict_before_fit_raises_error():

    covariances = np.array([
        [
            [2.0, 0.0],
            [0.0, 2.0],
        ],
    ])

    clf = MDM()

    try:
        clf.predict(covariances)
        assert False
    except RuntimeError:
        assert True