% Calls FMS.m to create PsiM and PsiP for 6 different audio files.
% Results are saved in .mat and .csv files.
%
% Also contains example code to compare new results with reference results
%
% Written January 12, 2023 by S. Voran at Institute for Telecommunication
% Sciences in Boulder, Colorado, United States: svoran@ntia.gov
wav_path = '../wavs'; %Path containing wav files
fnames = {...
'audioShort16k.wav'     %fs = 16k,      length appx. 1.6 sec
'audioLong16k.wav'      %fs = 16k,      length appx. 3.5 sec
'audio22k.wav'          %fs = 22.05k,   length appx. 3.3 sec
'audio32k.wav'          %fs = 32k,      length appx. 2.9 sec
'audio44k.wav'          %fs = 44.1k,    length appx. 5.8 sec
'audio48k.wav'};        %fs = 48k,      length appx. 3.0 sec
reference_folder = 'reference_files';
output_folder = 'output';
if ~exist(output_folder, 'dir')
    mkdir(output_folder)
end
for i= 1:6 %Loop over audio files listed
    filepath = fullfile(wav_path, fnames{i});
    [PsiM, PsiP] = FMS(filepath);     %Apply FMS.m
    [~, name, ~] = fileparts(fnames{i}); %Extract base filename
    save(fullfile(output_folder, [name,'.mat']), 'PsiM', 'PsiP'); %Save both variables in .mat
    writematrix(PsiM,fullfile(output_folder, [name,'PsiM.csv']));    %Save PsiM in .csv
    writematrix(PsiP,fullfile(output_folder, [name,'PsiP.csv']));    %Save PsiP in .csv
end

%Example of how to compare new results with reference results
Ref  = load(fullfile(reference_folder, 'audio48kRef.mat')); %load reference results
New = load(fullfile(output_folder, 'audio48k.mat'));     %load new results

absError = abs(Ref.PsiM - New.PsiM); %calc. absolute difference for PsiM
%Display mean
disp(['Mean absolute PsiM error is: ', num2str( mean( absError(:) ) )])

absError = abs(Ref.PsiP - New.PsiP); %calc. absolute difference for PsiM
%Display mean
disp(['Mean absolute PsiP error is: ', num2str( mean( absError(:) ) )])