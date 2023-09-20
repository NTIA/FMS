# FMS Python Code
Implements fixed-size modulation spectrum in FMS.py. Demo.ipynb applies FMS.py to 6 audio files and saves the results in .pkl and .csv files.

# How to run
## Installation
```
pip install .
```

## Usage
```
import fms
PsiM, PsiP = fms.fms(audio_filename)
```

### Inputs
 `audio_filename` is a path to a `.wav` audio file with a sampling rate of
16, 32, 48, 22.05, or 44.1 kHz. The minimum duration of the file is 60 ms. If the
file has more than one channel, the first channel is used.

### Outputs

The outputs are `PsiM` and `PsiP`. Each is a matrix with size Nmel by 8. They contain the magnitude and phase FMS, respectively. Nmel depends on fs:

|   fs     |  Nmel   |
|----------|---------|
|  16k     |  32     |
|  32k     |  40     |
|  48k     |  45     |
|  22.05k  |  35     |
|  44.1k   |  44     |


See `help(fms.fms)` for more details.

### Demo Code

The jupyter notebook demo.ipynb reads in each of six provided audio (.wav) files and calculates the FMS for each one.  This produces PsiM and PsiP.  These two results are stored in a single pickled file (.pkl) as a dictionary with keys 
`magnitude` and `phase`. In addition, each is stored individual in a CSV (.csv) file. 

The sample rates and durations of the six audio files are:

|   Audio File      |  Sample Rate |Duration|
|-------------------|--------------|--------|
| audioShort16k.wav |      16k     |  1.6 s |
| audioLong16k.wav  |      16k     |  3.5 s |
| audio32k.wav      |      32k     |  3.3 s |
| audio48k.wav      |      48k     |  2.9 s |
| audio22k.wav      |      22.05k  |  5.8 s |
| audio44k.wav      |      44.1k   |  3.0 s |

The corresponding output files for the six audio files are:

|   Audio File In   |  Pickle File Out  | CSV File Out          | CSV File Out         |
|-------------------|-------------------|-----------------------|----------------------|
| audioShort16k.wav | audioShort16k.pkl |audioShort16kPsiM.csv  |audioShort16kPsiP.csv |
| audioLong16k.wav  | audioLong16k.pkl  |audioLong16kPsiM.csv   |audioLong16kPsiP.csv  |
| audio32k.wav      | audio32k.pkl      |audio32kPsiM.csv       |audio32kPsiP.csv      |
| audio48k.wav      | audio48k.pkl      |audio48kPsiM.csv       |audio48kPsiP.csv      |
| audio22k.wav      | audio22k.pkl      |audio22kPsiM.csv       |audio22kPsiP.csv      |
| audio44k.wav      | audio44k.pkl      |audio44kPsiM.csv       |audio44kPsiP.csv      |

Six reference .pkl files are also provided: audioShort16kRef.pkl, audioLong16kRef.pkl, audio32kRef.pkl, audio48kRef.pkl, audio22kRef.pkl, and audio44kRef.pkl.  Each contains reference values of PsiM and PsiP to allow comparison with corresponding new results if desired.