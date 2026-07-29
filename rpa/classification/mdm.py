import numpy as np 
from rpa.core.riemann_base import (
    riemannian_mean,
    riemannian_distance,
)

class MDM:
    """
    Minimum distance to mean classifier for SPD matrices
    """

    def __init__(self):
        self.classes_ = None
        self.class_means_ = None

    def fit(self, covariances, labels):
        """
        Fit the MDM classifier.

        Parameters
        ----------
        covariances : ndarray
            SPD covariance matrices with shape
            (n_matrices, n_channels, n_channels).

        labels : ndarray
            Class labels with shape (n_matrices,).

        Returns
        -------
        self : MDM
            Fitted classifier.
        """
        covariances = np.asarray(covariances, dtype=float)
        labels = np.asarray(labels)

        if covariances.ndim != 3:
            raise ValueError(
                "covariances must have shape"
                "(n_matrices, n_channels, n_channels)"
            )

        if labels.ndim != 1:
            raise ValueError("labels must be a one-dimensional array.")

        if covariances.shape[0] != labels.shape[0]:
            raise ValueError(
                "covariances and labels must contain the same number of samples."
            )

        self.classes_ = np.unique(labels) # stores all class labels seen during training 
        self.class_means_ = {} # stores the riemannian mean for each class

        for label in self.classes_:
            class_covariances = covariances[labels==label]
            self.class_means_[label] = riemannian_mean(class_covariances)

        return self

    def predict(self, covariances):
        """
        Predict labels for covariance matrices. 

        Parameters
        ----------
        covariances : ndarray
            SPD covariance matrices with shape
            (n_matrices, n_channels, n_channels).
        
        Returns 
        -------
        predictions : ndarray
            Predicted class labels for each covariance matrix.
        """

        if self.class_means_ is None:
            raise RuntimeError("MDM classifier has not been fitted yet.")
        
        covariances = np.asarray(covariances, dtype=float)

        if covariances.ndim != 3:
            raise ValueError(
                "covariances must have shape"
                "(n_matrices, n_channels, n_channels)"
            )

        predictions = []

        for covariance in covariances:
            distances = {}

            for label, mean in self.class_means_.items(): 
                distances[label] = riemannian_distance(covariance, mean)

            predicted_label = min(distances, key=distances.get)
            predictions.append(predicted_label)


        return np.array(predictions)

        
        
        
        
        