import os
import numpy as np
from scipy.io import wavfile


def fms(audio_filename):
    """
    Creates fixed-size modulation spectrum (FMS).

    Creates fixed-size modulation spectrum (FMS) as described in
    XXX.
        Usage: psi_m, psi_p = fms(audio_filename)

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
        Magnitude FMS.
    psi_p : TODO: psi_p type
        Phase FMS.

    Notes
    -----
    PsiM and PsiP are matrices with size Nmel by 8. They contain the
    magnitude and phase FMS, respectively, as given in Eqn. (6).
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
    fs, x = load_audio(audio_filename)
    # Extract channel 1
    if len(x.shape) > 1:
        # x = x[:, 0]
        x = x[0]
    x = x[None, :]
    # Number of audio samples available
    Na = x.shape[1]
    # Check for sufficient audio duration
    if Na / fs < 0.060:
        raise ValueError(
            "File contains less than 60 ms of audio which is not suitable for FMS"
        )

    Nw, Ns, Nmel, fu = get_constants(fs)
    # Sample rate fs determines
    #   - Nw (samples per window or frame)
    #   - Ns (stride in samples)
    #   - Nmel (number of mel spectrum samples)
    #   - fu (upper limit of analysis in Hz)

    # Calc. number of frames those samples allow
    Nf = np.floor((Na - Nw) / Ns).astype(int) + 1
    # Set number of samples in DFT
    Nt = 2 * Nw

    # Make matrix Theta which implements filter bank that creates mel specrrum
    Theta = make_theta(fs, Nt, Nmel, fu)

    # Make matrix Phi which implements filterbank that creates modulation
    # spectrum
    Phi = makePhi(fs, Ns, Nf)

    # Generate normalized periodic Hamming Window with Nw samples - Eqn. (1)
    hammingWindow = (0.54 - 0.46 * np.cos(2 * np.pi * np.arange(0, Nw) / Nw)) / (
        0.54 * Nw
    )

    # Create matrix of windowed audio sample frames - Eqn. (1)
    xW = np.zeros((Nf, Nw))
    for f in range(Nf):
        sample_ix = np.arange(f * Ns, f * Ns + Nw)
        xW[f, :] = np.multiply(x[0, sample_ix], hammingWindow)

    # Zero pad matrix so frames have length Nt  - Eqn. (2)
    xTildeW = np.concatenate([xW, np.zeros((Nf, Nt - Nw))], axis=1)

    # DFT all frames - Eqn. (3)
    # X = fft(xTildeW, [], 2)
    X = np.fft.fft(xTildeW, axis=1)

    # Convert to mel spectrum - Eqn. (4)
    ss_ix = np.arange(Nt / 2 + 1).astype(int)
    P = np.matmul(np.abs(X[:, ss_ix]), Theta)
    # P = np.abs( X[:, 1:Nt/2 + 1] ) * Theta

    # Generate unnormalized symmetric Hamming Window with Nf samples - Eqn. (5)
    hammingWindow = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(0, Nf) / (Nf - 1))

    # Window and DFT all mel bins across frames - Eqn. (5)
    Gamma = np.fft.fft(
        np.multiply(P, hammingWindow[:, np.newaxis]), axis=0
    ).T.conjugate()

    # Number of Hz-scale spectral samples produced by DFT
    N = np.floor(Nf / 2) + 1
    # Filter DFT result to create modulation spectrum - Eqn. (6)
    N_ix = np.arange(N).astype(int)
    PsiM = np.matmul(np.abs(Gamma[:, N_ix]), Phi)
    PsiP = np.matmul(np.angle(Gamma[:, N_ix]), Phi)

    # (Straight from Matlab warning): Unwrapped phase might be a good option to
    # try in some cases
    # PsiP = unwrap( angle( Gamma(:,1:N) ) )*Phi
    return PsiM, PsiP


def load_audio(audio_filename):
    """
    Load floating point representation of audio file.

    Parameters
    ----------
    audio_filename : str
        Path to audio file.

    Returns
    -------
    fs : int
        Sample rate.
    x : np.ndarray
        Floating point representation of audio file.

    Raises
    ------
    RuntimeError
        Audio is not unsigned 8 bit int, signed 16 or 32 bit int, or floating
        point.
    """
    fs, x = wavfile.read(audio_filename)
    # Convert x to float representation (between -1 and 1)
    if np.issubdtype(x.dtype, np.floating):
        x = x
    elif x.dtype is np.dtype("uint8"):
        x = (x.astype("float") - 64) / 64
    elif x.dtype is np.dtype("int16"):
        x = x.astype("float") / (2**15)
    elif x.dtype is np.dtype("int32"):
        x = x.astype("float") / (2**31)
    else:
        raise RuntimeError(f"unknown audio type '{x.dtype}'")
    return fs, x


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
    # Convert upper analysis limit from Hz to mel, GWEMS filter paper Eqn. (1)
    fTildeU = 2595 * np.log10(1 + fu / 700)

    # %Find mel interval, GWEMS filter paper Eqn. (2)
    deltaTilda = fTildeU / (Nmel + 1)

    # Find band limits and centers in mel
    bTilda = deltaTilda * np.arange(0, Nmel + 2)

    # %Convert to Hz, GWEMS filter paper Eqn. (3)
    # b = 700 * (10.0 ^ (bTilda / 2595) - 1)
    b = 700 * (np.power(10.0, (bTilda / 2595)) - 1)

    # %Calculate DFT frequencies in Hz, GWEMS filter paper Eqn. (5)
    f = np.arange(0, Nt / 2 + 1) * fs / Nt

    # %Calculate filter normalizations, GWEMS filter paper Eqn. (6)
    # eta = 1./( b(3:Nmel+2) - b(1:Nmel) )
    eta = np.divide(1, b[2:] - b[:-2])

    # Number of DFT samples
    NHertz = Nt / 2 + 1

    # Convert to ints to make python happy
    NHertz = int(NHertz)
    Nmel = int(Nmel)
    # Calculate filterbank, GWEMS filter paper Eqn. (4)
    Theta = np.zeros((NHertz, Nmel))

    for i in range(Nmel):
        for k in range(NHertz):
            if b[i] <= f[k] and f[k] < b[i + 1]:
                # Lower slope
                Theta[k, i] = eta[i] * (f[k] - b[i]) / (b[i + 1] - b[i])
            elif b[i + 1] <= f[k] and f[k] < b[i + 2]:
                # Upper slope
                Theta[k, i] = eta[i] * (1 - (f[k] - b[i + 1]) / (b[i + 2] - b[i + 1]))
            else:
                Theta[k, i] = 0

    return Theta


def makePhi(fs, Ns, Nf):
    """
    Make matrix Phi which implements a filterbank that creates modulation
    spectrum

    Parameters
    ----------
    fs : int
        Audio sample rate
    Ns : int
        Stride of framing in samples (frame rate is fs/Ns frames per sec)
    Nf : int
        Number of frames to be processed

    Returns
    -------
    Phi : np.ndarray
        N by Nmod, where N = floor(Nf/2) + 1 and Nmod is 8
    """
    # number of spectral samples available
    N = np.floor(Nf / 2).astype(int) + 1
    # set number of modulation spectrum samples
    Nmod = 8

    # Calculate log-scale frequency interval, GWEMS filter paper Eqn. (8)
    DeltaBar = (np.log2(128) - np.log2(4)) / (Nmod - 1)

    # Calculate log-scale initial filter center frequencies,
    # GWEMS filter paper Eqn. (9)
    bBar = np.log2(4) + np.arange(0, Nmod) * DeltaBar

    # Calculate DFT bin spacing, GWEMS filter paper Eqn. (10)
    Deltah = fs / (Ns * Nf)

    # Ignore warning on log2(0)
    with np.errstate(divide="ignore"):
        # Calculate DFT bin log frequencies, GWEMS filter paper Eqn. (11)
        fBar = np.log2(np.arange(0, N) * Deltah)

    # Adjust log-scale initial filter center frequencies to fall on DFT bins,
    # GWEMS filter paper Eqn. (12)
    # bBarPrime = np.log2( np.round( ( 2.^bBar )/Deltah ) * Deltah );
    bBarPrime = np.log2(np.round(np.power(2, bBar) / Deltah) * Deltah)

    # Calculate filter half-widths, GWEMS filter paper Eqn. (13)
    deltaBar = DeltaBar / (2 - np.sqrt(2))

    # %Calculate filter weights to account for number of DFT samples spanned
    # %by each filter, GWEMS filter paper Eqn. (15)
    nu = []
    for m in range(Nmod):
        cond1 = -deltaBar <= (fBar - bBarPrime[m])
        cond2 = (fBar - bBarPrime[m]) < deltaBar
        sum = np.sum(cond1 & cond2)
        if sum == 0:
            val = np.inf
        else:
            val = 1 / sum

        nu.append(val)

    # Calculate filterbank, GWEMS filter paper Eqn. (14)
    Phi = np.zeros((N, Nmod))

    for m in range(Nmod):
        for k in range(N):
            if (bBarPrime[m] - deltaBar <= fBar[k]) and (fBar[k] < bBarPrime[m]):
                # Lower slope
                Phi[k, m] = nu[m] * (fBar[k] - (bBarPrime[m] - deltaBar)) / deltaBar
            elif (bBarPrime[m] <= fBar[k]) and (fBar[k] < bBarPrime[m] + deltaBar):
                # Upper slope
                Phi[k, m] = nu[m] * (1 - (fBar[k] - bBarPrime[m]) / deltaBar)
            else:
                Phi[k, m] = 0
    return Phi


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
    Return sample rate dependent FMS constants.

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
