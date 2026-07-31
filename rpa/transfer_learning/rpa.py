import numpy as np

from rpa.classification import MDM

from rpa.transfer_learning.alignment import (
    center_covariances,
    dispersion,
    stretch_covariances,
    rotate_covariances,
)

from rpa.transfer_learning.procrustes import estimate_rotation


class RPA:
    """
    Riemannian Procrustes Analysis for transfer learning with
    SPD covariance matrices.

    Current implementation:
        - recenter source and target data
        - stretch target data to match source dispersion
        - rotate target data using a placeholder identity rotation
        - classify using MDM

    The rotation matrix U is currently set to identity.
    A paper-faithful U estimation method will be added later.
    """

    def __init__(self):
        self.source_mean_ = None
        self.target_mean_ = None
        self.source_dispersion_ = None
        self.target_dispersion_ = None
        self.scale_ = None
        self.U_ = None
        self.classifier_ = None

    def fit(
        self,
        source_covariances,
        source_labels,
        target_labeled_covariances,
        target_labeled_labels,
    ):
        """
        Fit the RPA model.

        Parameters
        ----------
        source_covariances : ndarray
            Source SPD covariance matrices with shape
            (n_source, n_channels, n_channels).

        source_labels : ndarray
            Source labels with shape (n_source,).

        target_labeled_covariances : ndarray
            Small labeled target SPD covariance matrices with shape
            (n_target_labeled, n_channels, n_channels).

        target_labeled_labels : ndarray
            Target labeled labels with shape (n_target_labeled,).

        Returns
        -------
        self : RPA
            Fitted RPA model.
        """

        source_covariances = np.asarray(source_covariances, dtype=float)
        source_labels = np.asarray(source_labels)

        target_labeled_covariances = np.asarray(
            target_labeled_covariances,
            dtype=float,
        )
        target_labeled_labels = np.asarray(target_labeled_labels)

        self._validate_inputs(
            source_covariances,
            source_labels,
            target_labeled_covariances,
            target_labeled_labels,
        )

        # Step 1: center the source covariance distribution.
        source_centered, self.source_mean_ = center_covariances(
            source_covariances
        )

        # Step 2: center the labeled target covariance distribution.
        target_labeled_centered, self.target_mean_ = center_covariances(
            target_labeled_covariances
        )

        # Step 3: compute source and target dispersions around identity.
        n_channels = source_covariances.shape[1]
        identity = np.eye(n_channels)

        self.source_dispersion_ = dispersion(
            source_centered,
            mean=identity,
        )

        self.target_dispersion_ = dispersion(
            target_labeled_centered,
            mean=identity,
        )

        # Step 4: compute the stretching scale.
        if np.isclose(self.target_dispersion_, 0.0):
            self.scale_ = 1.0
        else:
            self.scale_ = np.sqrt(
                self.source_dispersion_ / self.target_dispersion_
            )

        # Step 5: stretch labeled target data.
        target_labeled_stretched = stretch_covariances(
            target_labeled_centered,
            self.scale_,
        )

        # Step 6: estimate rotation matrix U.
        self.U_ = estimate_rotation(
            source_centered,
            source_labels,
            target_labeled_stretched,
            target_labeled_labels,
        )

        # Step 7: rotate labeled target data.
        target_labeled_rotated = rotate_covariances(
            target_labeled_stretched,
            self.U_,
        )

        # Step 8: train MDM on source centered data plus transformed
        # labeled target data.
        train_covariances = np.concatenate(
            [
                source_centered,
                target_labeled_rotated,
            ],
            axis=0,
        )

        train_labels = np.concatenate(
            [
                source_labels,
                target_labeled_labels,
            ],
            axis=0,
        )

        self.classifier_ = MDM()
        self.classifier_.fit(train_covariances, train_labels)

        return self

    def transform_target(self, target_covariances):
        """
        Transform target covariance matrices using the learned RPA
        parameters.

        Parameters
        ----------
        target_covariances : ndarray
            Target SPD covariance matrices with shape
            (n_target, n_channels, n_channels).

        Returns
        -------
        target_rotated : ndarray
            Transformed target covariance matrices.
        """

        if self.target_mean_ is None:
            raise RuntimeError("RPA model has not been fitted yet.")

        target_covariances = np.asarray(target_covariances, dtype=float)

        # Step 1: center using the target mean learned during fit.
        target_centered, _ = center_covariances(
            target_covariances,
            mean=self.target_mean_,
        )

        # Step 2: apply the same stretching scale learned during fit.
        target_stretched = stretch_covariances(
            target_centered,
            self.scale_,
        )

        # Step 3: apply the same rotation learned during fit.
        target_rotated = rotate_covariances(
            target_stretched,
            self.U_,
        )

        return target_rotated

    def predict(self, target_covariances):
        """
        Predict labels for target covariance matrices.

        Parameters
        ----------
        target_covariances : ndarray
            Target SPD covariance matrices with shape
            (n_target, n_channels, n_channels).

        Returns
        -------
        predictions : ndarray
            Predicted labels.
        """

        if self.classifier_ is None:
            raise RuntimeError("RPA model has not been fitted yet.")

        target_transformed = self.transform_target(target_covariances)

        return self.classifier_.predict(target_transformed)

    @staticmethod
    def _validate_inputs(
        source_covariances,
        source_labels,
        target_labeled_covariances,
        target_labeled_labels,
    ):
        """
        Validate input shapes for RPA fitting.
        """

        if source_covariances.ndim != 3:
            raise ValueError(
                "source_covariances must have shape "
                "(n_source, n_channels, n_channels)."
            )

        if target_labeled_covariances.ndim != 3:
            raise ValueError(
                "target_labeled_covariances must have shape "
                "(n_target_labeled, n_channels, n_channels)."
            )

        if source_covariances.shape[1:] != target_labeled_covariances.shape[1:]:
            raise ValueError(
                "source_covariances and target_labeled_covariances "
                "must have the same matrix shape."
            )

        if source_labels.ndim != 1:
            raise ValueError("source_labels must be one-dimensional.")

        if target_labeled_labels.ndim != 1:
            raise ValueError(
                "target_labeled_labels must be one-dimensional."
            )

        if source_covariances.shape[0] != source_labels.shape[0]:
            raise ValueError(
                "source_covariances and source_labels must contain "
                "the same number of samples."
            )

        if target_labeled_covariances.shape[0] != target_labeled_labels.shape[0]:
            raise ValueError(
                "target_labeled_covariances and target_labeled_labels "
                "must contain the same number of samples."
            )