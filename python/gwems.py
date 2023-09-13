import os

import numpy as np

from scipy.io import wavfile


def gwems(audio_filename):
    """
    Creates Global Wideband Entire Modulation Spectrum (GWEMS).

    Creates Global Wideband Entire Modulation Spectrum (GWEMS) as described in
    "The Global Wideband Entire Modulation Spectrum: A Powerful, Compact,
    Fixed-Size Representation for Perceptually-Consistent Speech Evaluations"
    and "Filterbanks Used in GWEMS" both by Stephen Voran, Institute for
    Telecommunication Sciences in Boulder, Colorado.
        Usage: psi_m, psi_p = gwems(audio_filename)

    audio_filename is expected to be a .wav file with


    Parameters
    ----------
    audio_filename : str
        A .wav file with fs = 16, 32, 48, 22.05 or 44.1k.  Duration needs to
        be at least 60 ms. If file has more than one channel, then channel 1 is
        used.

    Returns
    -------
    psi_m : TODO: psi_m type
        Magnitude GWEMS.
    psi_p : TODO: psi_p type
        Phase GWEMS.

    Notes
    -----
    PsiM and PsiP are matrices with size Nmel by 8. They contain the
    magnitude and phase GWEMS, respectively, as given in Eqn. (6).
    Nmel depends on fs:
    fs      Nmel
    ------   ----
    16k       32
    32k       40
    48k       45
    22.05k    35
    44.1k     44

    For all 5 sample rates, the lower 32 mel bands match exactly. The
    bands above 32 cover the range from 8 kHz up to near the Nyquist
    frequency for the given sample rate.

    # TODO: Add python info here
    Written January 12, 2023 by S. Voran at Institute for Telecommunication
    Sciences in Boulder, Colorado, United States: svoran@ntia.gov
    Written and tested using MATLAB Version: 9.11.0.1809720 (R2021b) Update 1
    Note that Matlab uses one-based indexing, while the papers use
    zero-based indexing
    """
    # Check that .wav file has been specified
    _, ext = os.path.splitext(audio_filename)
    if ext != ".wav":
        raise ValueError("audio_filename is expected to name a .wav file.")

    # Read audio samples and sample rate
    fs, x = wavfile.read(audio_filename)
    # Extract channel 1
    if len(x.shape) > 1:
        x = x[:, 0]

    # Check for sufficient audio duration
    if len(x) / fs < 0.060:
        raise ValueError(
            "File contains less than 60 ms of audio which is not suitable for GWEMS"
        )

    Nw, Ns, Nmel, fu = get_constants(fs)
    # Sample rate fs determines
    #   - Nw (samples per window or frame)
    #   - Ns (stride in samples)
    #   - Nmel (number of mel spectrum samples)
    #   - fu (upper limit of analysis in Hz)

    # Number of audio samples available
    Na = len(x)
    # Calc. number of frames those samples allow
    Nf = np.floor((Na - Nw) / Ns) + 1
    # Set number of samples in DFT
    Nt = 2 * Nw

    # Make matrix Theta which implements filter bank that creates mel specrrum
    Theta = make_theta(fs, Nt, Nmel, fu)


def make_theta(fs, Nt, Nmel, fu):
    """
    Make matrix that implements a filter bank that creates mel spectrum.

    Parameters
    ----------
    fs : int
        Audio sample rate
    Nt : int
        Number of samples produced by DFT (must be even)
    Nmel : int
        Number of mel spectrum samples that will be produced
    fu : float
        Upper frequency limit of the analysis in Hz

    Returns
    -------
    Theta : np.ndarray
        NHertz by Nmel matrix representing filter bank that creates mel
        spectrum. NHertz = Nt/2 + 1.
    """

    # TODO: finish make_theta
    print("TODO")


def hz2mel(hz):
    """
    Convert Hz to mel scale

    Parameters
    ----------
    hz : float
        Frequency in hertz.

    Returns
    -------
    float
        Hertz value on the mel scale.
    """
    return 2595 * np.log10(1 + hz / 700)


def mel2hz(mel):
    """
    Convert mel scale to Hz.

    Parameters
    ----------
    mel : float
        Frequency in mel scale

    Returns
    -------
    float
        Frequency in Hz.
    """
    return 700 * (10 ** (mel / 2595) - 1)


def get_constants(fs):
    """
    Return sample rate dependent GWEMS constants.

    Parameters
    ----------
    fs : int
        Sample rate of audio sample.

    Returns
    -------
    Nw : int
        Samples per window or frame
    Ns : int
        Stride in samples
    Nmel : int
        Number of mel spectrum samples
    fu : float
        Upper limit of analysis in Hz.

    Notes
    -----
    Audio sample rate is used to look up Nw and Ns.
    Audio sample rate is used calculate Nmel and fu.
    """
    # Select paramters based on sample rate (from Table 2)
    if fs == 16000:
        Nw = 256
        Ns = 48
    elif fs == 32000:
        Nw = 512
        Ns = 96
    elif fs == 48000:
        Nw = 768
        Ns = 144
    elif fs == 22050:
        Nw = 384
        Ns = 66
    elif fs == 44100:
        Nw = 768
        Ns = 132
    else:
        raise ValueError(
            f"Unexepcted sample rate {fs}, (16k, 32, 48, 22.05, and 44.1k are expected)."
        )
    # Number of mel spectrum samples desired in DC to 8 kHz
    nWBmel = 32

    # Convert 8 kHz to mel
    mel8k = hz2mel(8000)

    # Width of mel interval so that nWBmel intervals cover DC to 8 kHz
    melDelta = mel8k / (nWBmel + 1)

    # Convert Nyquist frequency to mel
    melNyquist = hz2mel(fs / 2)

    # How many full melDelta intervals will fit into Nyquist band?
    Nmel = np.floor(melNyquist / melDelta) - 1

    # Find upper analysis limit in mel
    fuMel = melDelta * (Nmel + 1)
    fu = mel2hz(fuMel)

    return Nw, Ns, Nmel, fu
