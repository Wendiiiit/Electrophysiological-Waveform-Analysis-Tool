import numpy as np



# A function to calcualte noise 
def calculate_noise_ptp(
    time_ms: np.ndarray,
    pcm: np.ndarray,
    noise_start: float = 0.0,
    noise_end: float = 1.0,
    
    # These parameters are converted into float regardless of their nature 
) -> float: # java equiv: float calculate_noise_ptp(np.ndarray time_ms, np.ndarray pcm, float noise_start = 0.0, float noise_end = 0.0)

    # noise_mask is an np.ndarray of a boolean, >= <= & 
    # what it is: an array as long as time_ms, where each element is True if the corresponding time_ms is inside the noise window
    # (time_ms >= noise_start): (ndarray[float] >= float): ndarray[bool]
    #      the above line gives us an array of bool which, for each corresponding time_ms, is it at least at the noise start?
    # (time_ms <= noise_end): this also gives us an array of booleans, telling if the corresponding time_ms is before or at the
    #       end of the noise window
    # therefore, if a time_ms member is both before the end of the noise window and also after its start, then it's noise!
    
    # Select waveform sample between 0 and 1 ms 
    noise_mask = (
        (time_ms >= noise_start)
        & (time_ms <= noise_end)
    )
    
    
    # pcm is an nd array of float, noise mask window is an ndarray of bool
    # pass in trues/falses to filter the pcm array
    # so we are only getting the pcm when the noise masks are true 
    # pcm here are all 100% noise
    noise_pcm = pcm[noise_mask]

    if len(noise_pcm) == 0:
        raise ValueError("No samples found in the noise window.")

    noise_min = np.min(noise_pcm)
    noise_max = np.max(noise_pcm)

    return float(noise_max - noise_min)





# A function to calculate the Signal to Noise ratio 
def calculate_snr(
    cochlear_signal_ptp: float | None,
    vestibular_signal_ptp: float | None,
    # should noise pcm be float or np.ndarray? 
    # noise_pcm: float,
) -> tuple[float | None, float | None]:
    
    
    # Noise is shared by both SNR calculations
    if noise_ptp <= 0: 
        raise ValueError (
            "Noise PtP must be greater than zero."
        )
    
    
    # Vestibular SNR 
    # Check cases where 
    # 1). there is no Vestibular PtP value at all 
    # 2). Value is something invalid 
    # 3). PtP value is zero or negative 
    if (
        vestibular_signal_ptp is None 
        or not np.isfinite(vestibular_signal_ptp)
        or vestibular_signal_ptp <= 0
    ):
        vestibular_snr = None
    
    else: 
        vestibular_snr = 10.0 * np.log10(
            vestibular_signal_ptp/noise_ptp
        )
    
    
    # Cochlear SNR 
    if (
        cochlear_signal_ptp is None 
        or not np.isfinite(cochlear_signal_ptp)
        or cochlear_signal_ptp <= 0
    ):
        cochlear_snr = None
    
    else: 
        cochlear_snr = 10.0 * np.log10(
            cochlear_signal_ptp/noise_ptp
        )
    
    return (
        float(vestibular_snr)
        if vestibular_snr is not None 
        else None,
        
        float(cochlear_snr)
        if cochlear_snr is not None
        else None, 
    )

        
