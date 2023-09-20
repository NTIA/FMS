function [PsiM, PsiP] = FMS(audioFilename)
% Creates fixed-size modulation spectrum (FMS) as described in
% Voran, S. and Pieper, J. "A Powerful, Fixed-Size Modulation 
% Spectrum Representation for Perceptually-Consistent Speech Evaluation."
%
% Usage: [PsiM, PsiP] = FMS(audioFilename)
%
% audioFilename is expected to be a .wav file with fs = 16, 32, 48, 22.05
% or 44.1k.  Duration needs to be at least 60 ms. If file has more than
% one channel, then channel 1 is used.
%
% PsiM and PsiP are matrices with size Nmel by 8. They contain the
% magnitude and phase FMS, respectively, as given in Eqn. (6).
% Nmel depends on fs:
%  fs      Nmel
%-------   ----
% 16k       32
% 32k       40
% 48k       45
% 22.05k    35 
% 44.1k     44
%
% For all 5 sample rates, the lower 32 mel bands match exactly. The 
% bands above 32 cover the range from 8 kHz up to near the Nyquist
% frequency for the given sample rate.
%
% Written January 12, 2023 by S. Voran at Institute for Telecommunication
% Sciences in Boulder, Colorado, United States: svoran@ntia.gov
%
% Written and tested using MATLAB Version: 9.11.0.1809720 (R2021b) Update 1
%
% Note that Matlab uses one-based indexing, while the papers use
% zero-based indexing

%Check that .wav file has been specified
[~, ~, ext] = fileparts(audioFilename);
if ~strcmp(ext,'.wav')
    error('audioFilename is expected to name a .wav file')
end


[x,fs] = audioread(audioFilename); %read audio samples and sample rate
x = x(:,1); %extract channel 1

%Check for sufficient audio duration 

if length(x)/fs < .060
    error(['File contains less than 60 ms of audio which is not ' ...
        'suitable for FMS'])
end

[Nw, Ns, Nmel, fu] = getConstants(fs);
%Sample rate fs determines
%  - Nw (samples per window or frame)
%  - Ns (stride in samples)
%  - Nmel (number of mel spectrum samples)
%  - fu (upper limit of analysis in Hz)

Na = length(x);             %Find number of audio samples available
Nf = floor( (Na-Nw)/Ns )+1;   %Calc. number of frames those samples allow
Nt = 2*Nw;                  %Set number of samples in DFT

%Make matrix Theta which implements filterbank that creates mel spectrum
Theta = makeTheta(fs, Nt, Nmel, fu);

%Make matrix Phi which implements filterbank that creates modulation
%spectrum
Phi = makePhi(fs, Ns, Nf);

%Generate normalized periodic Hamming Window with Nw samples - Eqn. (1)
hammingWindow = ( 0.54-0.46*cos( 2*pi*[0:Nw-1]/Nw ) )/(0.54*Nw);

%Create matrix of windowed audio sample frames - Eqn. (1)
xW = zeros(Nf,Nw);
for f = 0:Nf-1 %For each frame
    xW(f+1,:) = x(f*Ns + 1:f*Ns + Nw)' .* hammingWindow;
end

%Zero pad matrix so frames have length Nt  - Eqn. (2)
xTildeW = [xW, zeros(Nf,Nt-Nw)];

%DFT all frames - Eqn. (3)
X = fft(xTildeW, [], 2);

%Convert to mel spectrum - Eqn. (4)
P = abs( X(:, 1:Nt/2 + 1) ) * Theta;

%Generate unnormalized symmetric Hamming Window with Nf samples - Eqn. (5)
hammingWindow = ( 0.54  -0.46*cos( 2*pi*[0:Nf-1]/(Nf-1) ) );

%Window and DFT all mel bins across frames - Eqn. (5)
Gamma = fft(P.*hammingWindow',[],1)';

N = floor(Nf/2) + 1; %Number of Hz-scale spectral samples produced by DFT

%Filter DFT result to create modulation spectrum - Eqn. (6)
PsiM = abs(   Gamma(:,1:N) )*Phi;
PsiP = angle( Gamma(:,1:N) )*Phi;

%Unwrapped phase might be a good option to try in some cases
%PsiP = unwrap( angle( Gamma(:,1:N) ) )*Phi;

%--------------------------------------------------------------------------
function [Nw, Ns, Nmel, fu] = getConstants(fs)
%Uses audio sample rate fs to look up
%  - Nw (samples per window or frame)
%  - Ns (stride in samples)
%Uses audio sample rate fs to calculate
%  - Nmel (number of mel spectrum samples)
%  - fu (upper limit of analysis in Hz)

%Select paramters based on sample rate (from Table I)
if fs == 16000
    Nw = 256; %Number of samples per frame
    Ns = 48;  %Stride in samples (advance Ns samples each frame) 
elseif fs == 32000
    Nw = 512;
    Ns = 96;
elseif fs == 48000
    Nw = 768;
    Ns = 144;
elseif fs == 22050
    Nw = 384;
    Ns = 66;
elseif fs == 44100
    Nw = 768;
    Ns = 132;
else
    error(['Unexpected sample rate (16, 32, 48, 22.05, and ' ...
        '44.1k are expected).'])
end

%Number of mel spectrum samples desired in DC to 8 kHz 
nWBmel = 32;

%Convert 8 kHz to mel
mel8k = 2595*log10(1+8000/700); 

%Width of mel interval that so that nWBmel intervals cover DC to 8 kHz
melDelta = mel8k/(nWBmel+1); 

%Convert Nyquist frequency to mel
melNyquist = 2595*log10(1+(fs/2)/700); 

%How many full melDelta intervals will fit into Nyquist band?
Nmel = floor(melNyquist/melDelta)-1;

%Find upper analysis limit in mel
fuMel = melDelta*(Nmel+1); 
fu = 700*( 10^(fuMel/2595)-1 ); %Convert upper limit to Hz

%--------------------------------------------------------------------------
function Theta = makeTheta(fs, Nt, Nmel, fu)
% Make matrix Theta which implements a filterbank that creates mel spectrum
% fs is audio sample rate
% Nt is number of samples produced by DFT (must be even)
% Nmel is the number of mel spectrum samples that will be produced
% fu is the upper frequency limit of the analysis in Hz
%
% Theta is NHertz by Nmel, where NHertz =  Nt/2 + 1;

%Convert upper analysis limit from Hz to mel, FMS filter paper Eqn. (1)
fTildeU = 2595*log10(1+fu/700); 

%Find mel interval, FMS filter paper Eqn. (2)
deltaTilda = fTildeU/(Nmel+1);  %divid

%Find band limits and centers in mel
bTilda = deltaTilda*[0:Nmel+1];

%Convert to Hz, FMS filter paper Eqn. (3)
b = 700*(10.^(bTilda/2595)-1);

%Calculate DFT frequencies in Hz, FMS filter paper Eqn. (5)
f = [0:Nt/2]*fs/Nt;

%Calculate filter normalizations, FMS filter paper Eqn. (6)
eta = 1./( b(3:Nmel+2) - b(1:Nmel) );

NHertz = Nt/2 + 1; %Number of DFT samples

%Calculate filterbank, FMS filter paper Eqn. (4)
Theta = zeros(NHertz,Nmel);
for i = 0:Nmel-1 %Loop over bands
    for k = 0:NHertz-1 %Loop over DFT samples
        if ( b(i+1) <= f(k+1) ) && ( f(k+1) < b(i+2) ) %lower slope
          Theta(k+1,i+1) = eta(i+1)*(f(k+1)-b(i+1))/(b(i+2)-b(i+1));

        elseif ( b(i+2) <= f(k+1) ) && ( f(k+1) < b(i+3) ) %upper slope
          Theta(k+1,i+1) = eta(i+1)*(1 - (f(k+1)-b(i+2))/(b(i+3)-b(i+2)) );

        else %filter takes value zero
          Theta(k+1,i+1) = 0;
        end
    end
end

%--------------------------------------------------------------------------
function Phi = makePhi(fs, Ns, Nf)
% Make matrix Phi which implements a filterbank that creates modulation
% spectrum
% fs is audio sample rate
% Ns is stride of framing in samples (frame rate is fs/Ns frames per sec)
% Nf is the number of frames to be processed
%
% Phi is N by Nmod, where N =  floor(Nf/2) + 1 and Nmod is 8

N = floor(Nf/2) + 1; %number of spectral samples available
Nmod = 8; %set number of modulation spectrum samples

%Calculate log-scale frequency interval, FMS filter paper Eqn. (8)
DeltaBar = ( log2(128) - log2(4) )/(Nmod - 1);

%Calculate log-scale initial filter center frequencies, 
%FMS filter paper Eqn. (9)
bBar = log2(4) + [0:Nmod-1]*DeltaBar;

%Calculate DFT bin spacing, FMS filter paper Eqn. (10)
Deltah = fs/(Ns*Nf);

%Calculate DFT bin log frequencies, FMS filter paper Eqn. (11)
fBar = log2([0:N-1]*Deltah);

%Adjust log-scale initial filter center frequencies to fall on DFT bins,
%FMS filter paper Eqn. (12)
bBarPrime = log2( round( ( 2.^bBar )/Deltah ) * Deltah );

%Calculate filter half-widths, FMS filter paper Eqn. (13)
deltaBar = DeltaBar/(2-sqrt(2));

%Calculate filter weights to account for number of DFT samples spanned
%by each filter, FMS filter paper Eqn. (15)
nu = zeros(1,Nmod);
for m = 0:Nmod-1
    nu(m+1) = 1/sum( ( -deltaBar <= ( fBar - bBarPrime(m+1) ) ) & ...
        ( ( fBar - bBarPrime(m+1) ) < deltaBar ) );
end

%Calculate filterbank, FMS filter paper Eqn. (14)
Phi = zeros(N,Nmod);
for m = 0:Nmod - 1
    for k = 0:N - 1
        %lower slope
        if ( bBarPrime(m+1)-deltaBar <= fBar(k+1) ) && ...
                                ( fBar(k+1) < bBarPrime(m+1) )
            Phi(k+1,m+1) = nu(m+1)*( fBar(k+1)- ...
                                ( bBarPrime(m+1)-deltaBar ) )/deltaBar;

        %upper slope    
        elseif ( bBarPrime(m+1) <= fBar(k+1) ) && ...
                        ( fBar(k+1) < bBarPrime(m+1)+deltaBar )
            Phi(k+1,m+1) = nu(m+1)*...
                        ( 1 - ( fBar(k+1)-bBarPrime(m+1) )/deltaBar );

        %filter takes value 0    
        else
            Phi(k+1,m+1) = 0;
        end
    end
end
