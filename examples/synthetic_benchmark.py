import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from rpa import RPA
from rpa.classification import MDM


def make_spd_matrix(base, noise_level, random_state=None):
    """
    Create a noisy SPD matrix around a base SPD matrix.

    The noise is symmetric. If the resulting matrix is not positive
    definite, it is shifted slightly toward the SPD cone by adding a
    multiple of the identity matrix.
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
    Generate a synthetic SPD covariance dataset.
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
    Rotate covariance matrices to simulate a target-domain shift.
    """

    rotated = np.array([
        rotation_matrix.T @ covariance @ rotation_matrix
        for covariance in covariances
    ])

    rotated = 0.5 * (
        rotated + np.transpose(rotated, axes=(0, 2, 1))
    )

    return rotated


def scale_domain(covariances, scale):
    """
    Scale covariance matrices to simulate a dispersion shift.
    """

    return scale * covariances


def get_synthetic_configs():
    """
    Return difficulty settings for the synthetic benchmark.

    Each setting changes the domain-shift difficulty by controlling
    noise, rotation angle, target scale, and number of labeled target
    adaptation samples.
    """

    return {
        "easy": {
            "noise_level": 0.18,
            "rotation_angle": np.pi / 3.0,
            "target_scale": 1.8,
            "target_test_size": 0.9,
        },
        "moderate": {
            "noise_level": 0.24,
            "rotation_angle": np.pi / 3.0,
            "target_scale": 1.8,
            "target_test_size": 0.9,
        },
        "hard": {
            "noise_level": 0.30,
            "rotation_angle": 5.0 * np.pi / 12.0,
            "target_scale": 2.0,
            "target_test_size": 0.85,
        },
    }


def make_random_spd_mean(n_channels, random_state, diagonal_shift=1.0):
    """
    Create a random SPD matrix to use as a class mean.

    A @ A.T is positive semi-definite.
    Adding diagonal_shoft * I makes it positive definite
    """
    rng = np.random.default_rng(random_state)

    A = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(n_channels, n_channels),
    )

    matrix = A @ A.T
    matrix = matrix + diagonal_shift * np.eye(n_channels)

    matrix = 0.5 * (matrix + matrix.T)

    return matrix


def make_source_class_means(n_channels, random_state):
    """
    Create two synthetic class-level SPD means.

    The two classes are generated from different random SPD matrices, 
    then normalized so the overall scale does not explode when the 
    number of channels increases.
    """

    class_0 = make_random_spd_mean(
        n_channels=n_channels, 
        random_state=random_state,
        diagonal_shift=1.0,
    )

    class_1 = make_random_spd_mean(
        n_channels=n_channels,
        random_state=random_state + 1_000, # use 1_000 adding part to make sure that the generated class 1 is different than class
        diagonal_shift=1.0,
    )

    # normalize class means so that the overall scale does not explode when the number of channels increases
    class_0 = class_0 / np.trace(class_0) * n_channels
    class_1 = class_1 / np.trace(class_1) * n_channels

    return {
        0: class_0,
        1: class_1,
    }


def make_random_orthogonal_matrix(n_channels, random_state):
    """
    Generate a random orthogonal matrix using QR decomposition to be used in rotation.
    """
    rng = np.random.default_rng(random_state)

    A = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(n_channels, n_channels),
    )

    Q, R = np.linalg.qr(A)

    # fix signs to make the result deterministic with respect to QR sign.
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0

    Q = Q @ np.diag(signs)

    return Q


def run_single_experiment(random_state, config, n_channels):
    """
    Run one synthetic source-to-target RPA experiment.
    """

    source_class_means = make_source_class_means(
        n_channels=n_channels,
        random_state=random_state + 20_000,
    )

    source_covariances, source_labels = generate_domain(
        class_means=source_class_means,
        samples_per_class=50,
        noise_level=config["noise_level"],
        random_state=random_state,
    )

    target_covariances, target_labels = generate_domain(
        class_means=source_class_means,
        samples_per_class=50,
        noise_level=config["noise_level"],
        random_state=random_state + 10_000,
    )

    rotation_matrix = make_random_orthogonal_matrix(
        n_channels=n_channels,
        random_state=random_state + 30_000,
    )

    target_covariances = rotate_domain(
        target_covariances,
        rotation_matrix,
    )

    target_covariances = scale_domain(
        target_covariances,
        scale=config["target_scale"],
    )

    (
        target_labeled_covariances,
        target_test_covariances,
        target_labeled_labels,
        target_test_labels,
    ) = train_test_split(
        target_covariances,
        target_labels,
        test_size=config["target_test_size"],
        stratify=target_labels,
        random_state=random_state,
    )

    baseline = MDM()
    baseline.fit(source_covariances, source_labels)

    baseline_predictions = baseline.predict(target_test_covariances)

    baseline_accuracy = accuracy_score(
        target_test_labels,
        baseline_predictions,
    )

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

    improvement = rpa_accuracy - baseline_accuracy

    result = {
        "seed": random_state,
        "n_channels": n_channels,
        "baseline_accuracy": baseline_accuracy,
        "rpa_accuracy": rpa_accuracy,
        "improvement": improvement,
        "scale": rpa.scale_,
        "n_target_labeled": len(target_labeled_covariances),
        "n_target_test": len(target_test_covariances),
    }

    return result


def summarize_setting(setting_name, results):
    """
    Print mean and standard deviation for one benchmark setting.
    """

    baseline_accuracies = np.array([
        result["baseline_accuracy"]
        for result in results
    ])

    rpa_accuracies = np.array([
        result["rpa_accuracy"]
        for result in results
    ])

    improvements = np.array([
        result["improvement"]
        for result in results
    ])

    scales = np.array([
        result["scale"]
        for result in results
    ])

    summary = {
        "setting": setting_name,
        "baseline_mean": baseline_accuracies.mean(),
        "baseline_std": baseline_accuracies.std(),
        "rpa_mean": rpa_accuracies.mean(),
        "rpa_std": rpa_accuracies.std(),
        "improvement_mean": improvements.mean(),
        "improvement_std": improvements.std(),
        "scale_mean": scales.mean(),
        "scale_std": scales.std(),
    }

    return summary


def print_summary_table(summaries):
    """
    Print a compact benchmark summary table.
    """

    print()
    print("Synthetic RPA benchmark by difficulty")
    print("=====================================")
    print()
    print(
        "Setting   | Baseline MDM      | RPA               | Improvement       | Scale"
    )
    print(
        "----------|-------------------|-------------------|-------------------|----------------"
    )

    for summary in summaries:
        print(
            f"{summary['setting']:<9} | "
            f"{summary['baseline_mean']:.3f} ± {summary['baseline_std']:.3f}     | "
            f"{summary['rpa_mean']:.3f} ± {summary['rpa_std']:.3f}     | "
            f"{summary['improvement_mean']:+.3f} ± {summary['improvement_std']:.3f}    | "
            f"{summary['scale_mean']:.3f} ± {summary['scale_std']:.3f}"
        )


def print_per_seed_results(setting_name, results):
    """
    Print detailed per-seed results for one setting.
    """

    print()
    print(f"Per-seed results: {setting_name}")
    print("-" * (18 + len(setting_name)))
    print("seed | baseline | rpa   | improvement | scale")
    print("-----|----------|-------|-------------|------")

    for result in results:
        print(
            f"{result['seed']:>4} | "
            f"{result['baseline_accuracy']:.3f}    | "
            f"{result['rpa_accuracy']:.3f} | "
            f"{result['improvement']:+.3f}      | "
            f"{result['scale']:.3f}"
        )


def main():
    n_runs = 5

    channel_counts = [2, 4, 8]

    configs = get_synthetic_configs()

    summaries = []
    all_results = {}

    for n_channels in channel_counts:
        for setting_name, config in configs.items():
            results = []

            for seed in range(n_runs):
                print(
                    f"Running {setting_name}, "
                    f"{n_channels} channels, "
                    f"seed {seed}"
                )
                result = run_single_experiment(
                    random_state=seed,
                    config=config,
                    n_channels=n_channels,
                )
                results.append(result)

            combined_name = f"{setting_name}_{n_channels}ch"

            all_results[combined_name] = results

            summary = summarize_setting(
                combined_name,
                results,
            )

            summary["n_channels"] = n_channels
            summary["difficulty"] = setting_name

            summaries.append(summary)

    print_summary_table(summaries)

    for setting_name, results in all_results.items():
        print_per_seed_results(setting_name, results)


if __name__ == "__main__":
    main()


# import numpy as np

# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split

# from rpa import RPA
# from rpa.classification import MDM


# def make_spd_matrix(base, noise_level=0.18, random_state=None):
#     """
#     Create a noisy SPD matrix around a base SPD matrix
    
#     The noise is symmetric. If the resulting matrix is not SPD,
#     it is shifted slightly toward the SPD code by adding a multiple of
#     the identity matrix.
#     """

#     rng = np.random.default_rng(random_state)

#     n_channels = base.shape[0]

#     noise = rng.normal(
#         loc=0.0,
#         scale=noise_level,
#         size=(n_channels, n_channels),
#     )

#     noise = 0.5 * (noise + noise.T) # makes the noise symmetric

#     matrix = base + noise

#     min_eigenvalue = np.min(np.linalg.eigvalsh(matrix))

#     if min_eigenvalue <= 0:
#         matrix = matrix + (abs(min_eigenvalue) + 1e-3) * np.eye(n_channels)

#     matrix = 0.5 * (matrix + matrix.T) # make sure they are SPD

#     return matrix


# def generate_domain(
#     class_means, 
#     samples_per_class,
#     noise_level,
#     random_state,
# ):

#     """
#     Generate a synthetic SPD covariance dataset. 

#     Parameters
#     ----------
#     class_means : dict
#         Mapping from class label to class level SPD mean.

#     samples_per_class : int
#         Number of samples generated for each class.

#     noise_level : float
#         Strength of symmetric noise added around each class mean.

#     random_state : int
#         Random seed.

#     Returns
#     -------
#     covariances : ndarray
#         SPD covariance matrices with shape 
#         (n_samples, n_channels, n_channels)

#     labels : ndarray
#         Class labels with shape (n_samples,).
#     """

#     rng = np.random.default_rng(random_state)

#     covariances = []
#     labels = []

#     for label, class_mean in class_means.items():
#         for _ in range(samples_per_class):
#             covariance = make_spd_matrix(
#                 class_mean,
#                 noise_level=noise_level,
#                 random_state=rng.integers(0, 1_000_000),
#             )

#             covariances.append(covariance)
#             labels.append(label)

#     return np.array(covariances), np.array(labels)


# def rotate_domain(covariances, rotation_matrix):
#     """
#     Rotate covariance matrices to simulate a target-domain shift.
#     """

#     rotated = np.array([
#         rotation_matrix.T @ covariance @ rotation_matrix
#         for covariance in covariances
#     ])

#     rotated = 0.5 * (
#         rotated + np.transpose(rotated, axes=(0, 2, 1))
#     )

#     return rotated


# def scale_domain(covariances, scale):
#     """
#     Scale covariance matrices to simulate a dispersion shift.
#     """

#     return scale * covariances


# def run_single_experiment(random_state):
#     """
#     Run one synthetic source-to-target RPA experiment.

#     Returns
#     -------
#     result : dict
#         Dictionary containing baseline accuracy, RPA accuracy,
#         improvement, and learned scale factor.
#     """

#     source_class_means = {
#         0: np.array([
#             [2.0, 0.35],
#             [0.35, 1.3],
#         ]),
#         1: np.array([
#             [1.6, -0.25],
#             [-0.25, 1.9],
#         ]),
#     }

#     source_covariances, source_labels = generate_domain(
#         class_means=source_class_means,
#         samples_per_class=50,
#         noise_level=0.18,
#         random_state=random_state,
#     )

#     target_covariances, target_labels = generate_domain(
#         class_means=source_class_means,
#         samples_per_class=50,
#         noise_level=0.18,
#         random_state=random_state + 10_000,
#     )

#     angle = np.pi / 3.0

#     rotation_matrix = np.array([
#         [np.cos(angle), -np.sin(angle)],
#         [np.sin(angle), np.cos(angle)],
#     ])

#     target_covariances = rotate_domain(
#         target_covariances,
#         rotation_matrix,
#     )

#     target_covariances = scale_domain(
#         target_covariances,
#         scale=1.8,
#     )

#     (
#         target_labeled_covariances,
#         target_test_covariances, 
#         target_labeled_labels, 
#         target_test_labels, 
#     ) = train_test_split(
#         target_covariances,
#         target_labels, 
#         test_size=0.9,
#         stratify=target_labels,
#         random_state=random_state,
#     )

#     baseline = MDM()
#     baseline.fit(source_covariances, source_labels)

#     baseline_predictions = baseline.predict(target_test_covariances)
#     baseline_accuracy = accuracy_score(target_test_labels, baseline_predictions)

#     rpa = RPA()
#     rpa.fit(
#         source_covariances, 
#         source_labels, 
#         target_labeled_covariances, 
#         target_labeled_labels,
#     )

#     rpa_predictions = rpa.predict(target_test_covariances)
#     rpa_accuracy = accuracy_score(target_test_labels, rpa_predictions)

#     improvement = rpa_accuracy - baseline_accuracy

#     result = {
#         "seed": random_state,
#         "baseline_accuracy": baseline_accuracy,
#         "rpa_accuracy": rpa_accuracy,
#         "improvement": improvement,
#         "scale": rpa.scale_,
#     }

#     return result


# def summarize_results(results):
#     """
#     Print mean and standard deviation of benchmark results.
#     """

#     baseline_accuracies = np.array([    
#         result["baseline_accuracy"]
#         for result in results
#     ])

#     rpa_accuracies = np.array([
#         result["rpa_accuracy"]
#         for result in results
#     ])

#     improvements = np.array([
#         result["improvement"]
#         for result in results
#     ])

#     scales = np.array([
#         result["scale"]
#         for result in results
#     ])


#     print()
#     print("Synthetic RPA benchmark")
#     print("=======================")
#     print(f"Number of runs: {len(results)}")
#     print()
#     print(
#         "Baseline MDM accuracy: "
#         f"{baseline_accuracies.mean():.3f} ± {baseline_accuracies.std():.3f}"
#     )
#     print(
#         "RPA accuracy:          "
#         f"{rpa_accuracies.mean():.3f} ± {rpa_accuracies.std():.3f}"
#     )
#     print(
#         "RPA improvement:       "
#         f"{improvements.mean():.3f} ± {improvements.std():.3f}"
#     )
#     print(
#         "RPA scale factor:      "
#         f"{scales.mean():.3f} ± {scales.std():.3f}"
#     )

#     print()
#     print("Per-seed results")
#     print("----------------")
#     print("seed | baseline | rpa   | improvement | scale")
#     print("-----|----------|-------|-------------|------")

#     for result in results:
#         print(
#             f"{result['seed']:>4} | "
#             f"{result['baseline_accuracy']:.3f}    | "
#             f"{result['rpa_accuracy']:.3f} | "
#             f"{result['improvement']:+.3f}      | "
#             f"{result['scale']:.3f}"
#         )


# def main():
#     n_runs = 30

#     results = []

#     for seed in range(n_runs):
#         result = run_single_experiment(seed)
#         results.append(result)

#     summarize_results(results)


# if __name__ == "__main__":
#     main()
    


