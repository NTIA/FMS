# GWEMS MATLAB® Code
Implements Global Wideband Entire Modulation Spectrum in GWEMS.m.
Demo.m applies GWEMS.m to 6 audio files and saves the results in .mat and .csv files.

# How to run
* Clone to local repository  
* Open using MATLAB or Octave  
* Run demo.m or GWEMS.m.

Further details below (or type `help GWEMS` in the MATLAB command window).

# Usage

```
[PsiM, PsiP] = GWEMS(audioFilename)
```

## Inputs

The only input is audioFilename.  This is expected to name a .wav file with fs = 16, 32, 48, 22.05 or 44.1k. Duration needs to be at least 60 ms.  If file has more than one channel, then channel 1 is used. 

## Outputs

The outputs are PsiM and PsiP. Each is a matrix with size Nmel by 8. They contain the magnitude and phase GWEMS, respectively. Nmel depends on fs:

|   fs     |  Nmel   |
|----------|---------|
|  16k     |  32     |
|  32k     |  40     |
|  48k     |  45     |
|  22.05k  |  35     |
|  44.1k   |  44     |

## Demo Code

The script demo.m reads in each of six provided audio (.wav) files and calls gwems.m for each one.  This produces PsiM and PsiP.  These two results are stored in a single Matlab data file (.mat). In addition, each is stored individual in a CSV (.csv) file. 

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

|   Audio File In   |  Matlab File Out  | CSV File Out          | CSV File Out         |
|-------------------|-------------------|-----------------------|----------------------|
| audioShort16k.wav | audioShort16k.mat |audioShort16kPsiM.csv  |audioShort16kPsiP.csv |
| audioLong16k.wav  | audioLong16k.mat  |audioLong16kPsiM.csv   |audioLong16kPsiP.csv  |
| audio32k.wav      | audio32k.mat      |audio32kPsiM.csv       |audio32kPsiP.csv      |
| audio48k.wav      | audio48k.mat      |audio48kPsiM.csv       |audio48kPsiP.csv      |
| audio22k.wav      | audio22k.mat      |audio22kPsiM.csv       |audio22kPsiP.csv      |
| audio44k.wav      | audio44k.mat      |audio44kPsiM.csv       |audio44kPsiP.csv      |

Six reference .mat files are also provided: audioShort16kRef.mat, audioLong16kRef.mat, audio32kRef.mat, audio48kRef.mat, audio22kRef.mat, and audio44kRef.mat.  Each contains reference values of PsiM and PsiP to allow comparison with corresponding new results if desired.