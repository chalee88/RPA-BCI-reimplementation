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
        covariances = np.asarray(covariances, dtype=float)
        labels = np.asarray(labels)

        self.classes_ = np.unique(labels)
        self.class_means_ = {}

        for label in self.classes_:
            self.class_means_[label] = riemannian_mean(
                covariances[labels == label]
            )

        return self

    def predict(self, covariances):
        if self.class_means_ is None:
            raise RuntimeError("MDM classifier has not been fitted yet")

        
        covariances = np.asarray(covariances, dtype=float)

        predictions = []

        for covariance in covariances:
            distances = {
                label: riemannian_distance(covariance, mean)
                for label, mean in self.class_means_.items()
            }

            predictions.append(min(distances, key=distances.get))

        
        return np.array(predictions)