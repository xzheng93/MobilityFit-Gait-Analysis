function [ResultStruct] = my_sample_entropy(inputData,FS,LegLength,WindowLen,ApplyRealignment,ApplyRemoveSteps)

% DESCRIPTON: Trunk analysis of Iphone data without the need for step detection
% CL Nov 2019
% Adapted IH feb-april 2020

% koloms data of smartphone
% 1st column is time data;
% 2nd column is X, medio-lateral: + left, - right
% 3rd column is Y, vertical: + downwards, - upwards
% 4th column is Z, anterior- posterior : + forwards, - backwards

%% Input Trunk accelerations during locomotion in VT, ML, AP direction
% InputData: Acceleration signal with time and accelerations in VT,ML and
% AP direction.
% FS: sample frequency of the Accdata
% LegLength: length of the leg of the participant in m;


%% Output
% ResultStruct: structure coninting all outcome measured calculated
% Spectral parameters, spatiotemporal gait parameters, non-linear
% parameters
% fields and subfields: include the multiple measurements of a subject

%% Literature
% Richman & Moorman, 2000; [ sample entropy]
% Bisi & Stagni Gait & Posture 2016,  47 (6) 37-42
% Kavagnah et al., Eur J Appl Physiol 2005 94: 468?475; Human Movement Science 24(2005) 574?587 [ synchrony]
% Moe-Nilsen J Biomech 2004 37, 121-126 [ autorcorrelation step regularity and symmetry
% Kobsar et al. Gait & Posture 2014 39, 553?557 [ synchrony ]
% Rispen et al; Gait & Posture 2014, 40, 187 - 192 [realignment axes]
% Zijlstra & HofGait & Posture 2003  18,2, 1-10 [spatiotemporal gait variables]
% Lamoth et al, 2002  [index of harmonicity]
% Costa et al. 2003 Physica A 330 (2003) 5360 [ multiscale entropy]
% Cignetti F, Decker LM, Stergiou N. Ann Biomed Eng. 2012
% May;40(5):1122-30. doi: 10.1007/s10439-011-0474-3. Epub 2011 Nov 25. [
% Wofl vs. Rosenstein Lyapunov]


%% Settings
Gr = 9.81;                         % Gravity acceleration, multiplication factor for accelerations
StrideFreqEstimate = 1.00;         % Used to set search for stride frequency from 0.5*StrideFreqEstimate until 2*StrideFreqEstimate
StrideTimeRange = [0.4 4.0];       % Range to search for stride time (seconds)
IgnoreMinMaxStrides = 0.10;        % Number or percentage of highest&lowest values ignored for improved variability estimation
N_Harm = 12;                       % Number of harmonics used for harmonic ratio, index of harmonicity and phase fluctuation
LowFrequentPowerThresholds = ...
    [0.7 1.4];                     % Threshold frequencies for estimation of low-frequent power percentages
Lyap_m = 7;                        % Embedding dimension (used in Lyapunov estimations)
Lyap_FitWinLen = round(60/100*FS); % Fitting window length (used in Lyapunov estimations Rosenstein's method)
Sen_m = 5;                         % Dimension, the length of the subseries to be matched (used in sample entropy estimation)
Sen_r = 0.3;                       % Tolerance, the maximum distance between two samples to qualify as match, relative to std of DataIn (used in sample entropy estimation)
NStartEnd = [100];
M  = 5; % maximum template length
ResultStruct = struct();

%% Filter and Realign Accdata

% Apply Realignment & Filter data

if ApplyRealignment % apply realignment as described in Rispens S, Pijnappels M, van Schooten K, Beek PJ, Daffertshofer A, van Die?n JH (2014).
    data = inputData(:, [3,2,4]); % reorder data to  1 = V; 2= ML, 3 = AP%
    % Consistency of gait characteristics as determined from acceleration data collected at different trunk locations. Gait Posture 2014;40(1):187-92.
    [RealignedAcc, ~] = RealignSensorSignalHRAmp(data, FS);
    dataAcc = RealignedAcc;
    [B,A] = butter(2,20/(FS/2),'low');
    dataAcc_filt = filtfilt(B,A,dataAcc);
else % we asume tat data is already reorderd to 1 = V; 2= ML, 3 = AP in an earlier stage;
    [B,A] = butter(2,20/(FS/2),'low');
    dataAcc = inputData;
    dataAcc_filt = filtfilt(B,A,dataAcc);
end


%% Step dectection
% Determines the number of steps in the signal so that the first 30 and last 30 steps in the signal can be removed

if ApplyRemoveSteps
    
    % In order to run the step detection script we first need to run an autocorrelation function;
    [ResultStruct] = AutocorrStrides(dataAcc_filt,FS, StrideTimeRange,ResultStruct);
    
    % StrideTimeSamples is needed as an input for the stepcountFunc;
    StrideTimeSamples = ResultStruct.StrideTimeSamples;
    
    % Calculate the number of steps;
    [PksAndLocsCorrected] = StepcountFunc(dataAcc_filt,StrideTimeSamples,FS);
    % This function selects steps based on negative and positive values.
    % However to determine the steps correctly we only need one of these;
    LocsSteps = PksAndLocsCorrected(1:2:end,2);
    
    %% Cut data & remove currents results
    % Remove 20 steps in the beginning and end of data
    dataAccCut = dataAcc(LocsSteps(31):LocsSteps(end-30),:);
    dataAccCut_filt = dataAcc_filt(LocsSteps(31):LocsSteps(end-30),:);
    
    % Clear currently saved results from Autocorrelation Analysis
    
    clear ResultStruct;
    clear PksAndLocsCorrected;
    clear LocsSteps;
    
else;
    dataAccCut = dataAcc;
    dataAccCut_filt = dataAcc_filt;
end



%% Calculation non-linear parameters;

% cut into windows of size WindowLen
N_Windows = floor(size(dataAccCut,1)/WindowLen);
N_SkipBegin = ceil((size(dataAccCut,1)-N_Windows*WindowLen)/2);
SE= nan(N_Windows,3);

for WinNr = 1:N_Windows;
    AccWin = dataAccCut(N_SkipBegin+(WinNr-1)*WindowLen+(1:WindowLen),:);
    for j=1:3
        [SE(WinNr,j)] = funcSampleEntropy(AccWin(:,j), Sen_m, Sen_r);
        % no correction for FS; SE does increase with higher FS but effect is considered negligible as range is small (98-104HZ). Might consider updating r to account for larger ranges.
    end
end

SampleEntropy = nanmean(SE,1);

ResultStruct.SampleEntropy_V = SampleEntropy(1);
ResultStruct.SampleEntropy_ML = SampleEntropy(2);
ResultStruct.SampleEntropy_AP = SampleEntropy(3);


end