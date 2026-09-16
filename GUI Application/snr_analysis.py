import numpy as np


def calculate_noise_ptp(
    time_ms: np.ndarray,
    pcm: np.ndarray,
    noise_start: float = 0.0,
    noise_end: float = 1.0,
) -> float:

    noise_mask = (
        (time_ms >= noise_start)
        & (time_ms <= noise_end)
    )

    noise_pcm = pcm[noise_mask]

    if len(noise_pcm) == 0:
        raise ValueError("No samples found in the noise window.")

    noise_min = np.min(noise_pcm)
    noise_max = np.max(noise_pcm)

    return float(noise_max - noise_min)


def calculate_snr(
    cochlear_signal_ptp: float,
    vestibular_noise_ptp: float,
) -> float:

    if cochlear_signal_ptp <= 0:
        raise ValueError(
            "Cochlear Signal PtP must be greater than zero."
        )

    if vestibular_noise_ptp <= 0:
        raise ValueError(
            "Vestibular Noise PtP must be greater than zero."
        )

    return float(
        10.0 * np.log10(
            cochlear_signal_ptp / vestibular_noise_ptp
        )
    )