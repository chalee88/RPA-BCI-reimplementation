import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from rpa import RPA
from rpa.classification import MDM

def make_spd_matrix(base, noise_level=0.05, random_state=None):
    """
    Create a noisy SPD matrix around a base SPD matrix.

    The noise is added symmetrically, and then the result is stabilized 
    by adding a small multiple of the identity matrix if needed.
    """

    rng = np.random.default_rng(random_state)

    n_channels = base.shape[0]

    noise = rng.normal(
        loc=0.0,
        scale=noise_level,
        size=(n_channels, n_channels),
    )

    noise = 0.5 * (noise + noise.T)

    matrix = base + noise

    # Stabilize matrix if noise made it close to non-SPD
    min_eigenvalue = np.min(np.linalg.eigvalsh(matrix))

    if min_eigenvalue <= 0:
        matrix = matrix + (abs(min_eigenvalue) + 1e-3) * np.eye(n_channels)

    matrix = 0.5 * (matrix + matrix.T)

    return matrix


def generate_domain(
    class_means,
    samples_per_class,
    noise_level,
    random_state,
):

    """
    Generate covariance matrices and labels from class-level SPD means.
    """

    rng = np.random.default_rng(random_state)

    covariances = []
    labels = []

    for label, class_mean in class_means.items():
        for _ in range(samples_per_class):
            covariance = make_spd_matrix(
                class_mean,
                noise_level=noise_level,
                random_state=rng.integers(0, 1_000_000),
            )

            covariances.append(covariance)
            labels.append(label)

    return np.array(covariances), np.array(labels)


def rotate_domain(covariances, rotation_matrix):
    """
    Apply a domain rotation to covariance matrices.
    This creates a synthetic target domain from the source domain.
    """

    rotated = np.array([
        rotation_matrix.T @ covariance @ rotation_matrix
        for covariance in covariances
    ])

    rotated = 0.5 * (rotated + np.transpose(rotated, axes=(0, 2, 1)))

    return rotated


def scale_domain(covariances, scale):
    """
    Apply simple scalar scaling to covariance matrices. 
    """

    return scale * covariances 


def main():
    random_state = 42
    rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------
    # 1. Define source class covariance means.
    # ------------------------------------------------------------
    #
    # These are anisotropic covariance matrices, meaning they have
    # orientation information. That makes rotation meaningful.
    source_class_means = {
        0: np.array([
            [2.0, 0.35],
            [0.35, 1.3],
        ]),
        1: np.array([
            [1.6, -0.25],
            [-0.25, 1.9],
        ]),
    }

    # ------------------------------------------------------------
    # 2. Generate source data.
    # ------------------------------------------------------------
    source_covariances, source_labels = generate_domain(
        class_means=source_class_means,
        samples_per_class=50,
        noise_level=0.18,
        random_state=random_state,
    ) 

    # ------------------------------------------------------------
    # 3. Generate target data by rotating and scaling source-like data.
    # ------------------------------------------------------------
    target_covariances, target_labels = generate_domain(
        class_means=source_class_means,
        samples_per_class=50,
        noise_level=0.18,
        random_state=random_state + 1,
    )

    angle = np.pi / 3.0

    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])

    target_covariances = rotate_domain(
        target_covariances,
        rotation_matrix,
    )

    target_covariances = scale_domain(
        target_covariances,
        scale=1.8,
    )

    # ------------------------------------------------------------
    # 4. Split target into a small labeled adaptation set and test set.
    # ------------------------------------------------------------
    target_labeled_covariances, target_test_covariances, target_labeled_labels, target_test_labels = train_test_split(
        target_covariances,
        target_labels,
        test_size=0.9,
        stratify=target_labels,
        random_state=random_state,
    )

    # ------------------------------------------------------------
    # 5. Baseline: train MDM only on source and test directly on target.
    # ------------------------------------------------------------
    baseline = MDM()
    baseline.fit(source_covariances, source_labels)

    baseline_predictions = baseline.predict(target_test_covariances)

    baseline_accuracy = accuracy_score(
        target_test_labels,
        baseline_predictions,
    )

    # ------------------------------------------------------------
    # 6. RPA: adapt using source + small labeled target set.
    # ------------------------------------------------------------
    rpa = RPA()

    rpa.fit(
        source_covariances,
        source_labels,
        target_labeled_covariances,
        target_labeled_labels,
    )

    rpa_predictions = rpa.predict(target_test_covariances)

    rpa_accuracy = accuracy_score(
        target_test_labels,
        rpa_predictions,
    )

    # ------------------------------------------------------------
    # 7. Print results.
    # ------------------------------------------------------------
    print("Synthetic RPA demo")
    print("==================")
    print(f"Source samples: {len(source_covariances)}")
    print(f"Labeled target adaptation samples: {len(target_labeled_covariances)}")
    print(f"Target test samples: {len(target_test_covariances)}")
    print()
    print(f"Baseline MDM accuracy: {baseline_accuracy:.3f}")
    print(f"RPA accuracy:          {rpa_accuracy:.3f}")
    print()
    print("Learned RPA parameters")
    print("----------------------")
    print(f"Scale factor: {rpa.scale_:.3f}")
    print("Rotation matrix U:")
    print(rpa.U_)


if __name__ == "__main__":
    main()