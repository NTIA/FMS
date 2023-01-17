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

|   fs     |    Nmel |
|----------|---------|
| 16k      |    32   |
|  32k     |   40    |
|  48k     |  45     |
|  22.05k  | 35      |
|  44.1k   | 44      |