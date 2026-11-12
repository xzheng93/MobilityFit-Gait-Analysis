function [ResultStruct] = GaitOutcomesTrunkAccFuncIH_fast(inputData,FS,LegLength,WindowLen,ApplyRealignment,ApplyRemoveSteps)
% FAST copy of GaitOutcomesTrunkAccFuncIH for the convergence/length analysis only.
% IDENTICAL to the original EXCEPT: (1) the Rosenstein Lyapunov call
% (CalcMaxLyapConvGait) is removed because it is not used in the paper and it
% errors on very short segments; (2) the Wolf-Lyapunov and sample-entropy calls
% are wrapped in try/catch so one failing axis returns NaN instead of crashing.
% Sample entropy and LyapunovWolf computations are byte-for-byte the same as the
% original. Do NOT use for authoritative outcomes; use the original for those.

Gr = 9.81;
StrideFreqEstimate = 1.00;
StrideTimeRange = [0.4 4.0];
IgnoreMinMaxStrides = 0.10;
N_Harm = 12;
LowFrequentPowerThresholds = [0.7 1.4];
Lyap_m = 7;
Lyap_FitWinLen = round(60/100*FS);
Sen_m = 5;
Sen_r = 0.3;
NStartEnd = [100];
M  = 5;
ResultStruct = struct();

if ApplyRealignment
    data = inputData(:, [3,2,4]);
    [RealignedAcc, ~] = RealignSensorSignalHRAmp(data, FS);
    dataAcc = RealignedAcc;
    [B,A] = butter(2,20/(FS/2),'low');
    dataAcc_filt = filtfilt(B,A,dataAcc);
else
    [B,A] = butter(2,20/(FS/2),'low');
    dataAcc = inputData;
    dataAcc_filt = filtfilt(B,A,dataAcc);
end

if ApplyRemoveSteps
    [ResultStruct] = AutocorrStrides(dataAcc_filt,FS, StrideTimeRange,ResultStruct);
    StrideTimeSamples = ResultStruct.StrideTimeSamples;
    [PksAndLocsCorrected] = StepcountFunc(dataAcc_filt,StrideTimeSamples,FS);
    LocsSteps = PksAndLocsCorrected(1:2:end,2);
    dataAccCut = dataAcc(LocsSteps(31):LocsSteps(end-30),:);
    dataAccCut_filt = dataAcc_filt(LocsSteps(31):LocsSteps(end-30),:);
    clear ResultStruct; clear PksAndLocsCorrected; clear LocsSteps;
else
    dataAccCut = dataAcc;
    dataAccCut_filt = dataAcc_filt;
end

ResultStruct = struct;
[ResultStruct] = AutocorrStrides(dataAccCut_filt,FS, StrideTimeRange,ResultStruct);
StrideTimeSamples = ResultStruct.StrideTimeSamples;

ij = 1; dirSymm = [1,3];
for jk=1:length(dirSymm)
    [C, lags] = AutocorrRegSymmSteps(dataAccCut_filt(:,dirSymm(jk)));
    [Ad,p] = findpeaks(C,'MinPeakProminence',0.2, 'MinPeakHeight', 0.2);
    if size(Ad,1) > 1
        Ad1 = Ad(1); Ad2 = Ad(2);
        GaitSymm(:,ij) = abs((Ad1-Ad2)/mean([Ad1+Ad2]))*100;
    else
        GaitSymm(:,ij) = NaN;
    end
    ij = ij +1;
end
ResultStruct.GaitSymm_V = GaitSymm(1);
ResultStruct.GaitSymm_AP = GaitSymm(2);

[PksAndLocsCorrected] = StepcountFunc(dataAccCut_filt,StrideTimeSamples,FS);
LocsSteps = PksAndLocsCorrected(2:2:end,2);
if rem(size(LocsSteps,1),2) == 0
    LocsSteps2 = LocsSteps(1:2:end);
else
    LocsSteps2 = LocsSteps(3:2:end);
end
LocsSteps1 = LocsSteps(2:2:end);
DiffLocs2 = diff(LocsSteps2); DiffLocs1 = diff(LocsSteps1);
StepTime2 = DiffLocs2(1:end-1)/FS; StepTime1 = DiffLocs1(1:end-1)/FS;
SI = abs((2*(StepTime2-StepTime1))./(StepTime2+StepTime1))*100;
ResultStruct.GaitSymmIndex = mean(SI,'omitnan');

[ResultStruct] = SpatioTemporalGaitParameters(dataAccCut_filt,StrideTimeSamples,ApplyRealignment,LegLength,FS,IgnoreMinMaxStrides,ResultStruct);

AccVectorLen = sqrt(sum(dataAccCut_filt(:,1:3).^2,2));
[ResultStruct] = SpectralAnalysisGaitfunc(dataAccCut_filt,WindowLen,FS,N_Harm,LowFrequentPowerThresholds,AccVectorLen,ResultStruct);

N_Windows = floor(size(dataAccCut,1)/WindowLen);
N_SkipBegin = ceil((size(dataAccCut,1)-N_Windows*WindowLen)/2);
LyapunovWolf = nan(N_Windows,3);
LyapunovRosen = nan(N_Windows,3);
SE= nan(N_Windows,3);

for WinNr = 1:N_Windows
    AccWin = dataAccCut(N_SkipBegin+(WinNr-1)*WindowLen+(1:WindowLen),:);
    for j=1:3
        try
            [LyapunovWolf(WinNr,j),~] = CalcMaxLyapWolfFixedEvolv(AccWin(:,j),FS,struct('m',Lyap_m));
        catch
            LyapunovWolf(WinNr,j) = NaN;
        end
        % Rosenstein Lyapunov removed (not used in paper; errors on short segments)
        try
            [SE(WinNr,j)] = funcSampleEntropy(AccWin(:,j), Sen_m, Sen_r);
        catch
            SE(WinNr,j) = NaN;
        end
    end
end

LyapunovWolf = mean(LyapunovWolf,1,'omitnan');
LyapunovRosen = mean(LyapunovRosen,1,'omitnan');
SampleEntropy = mean(SE,1,'omitnan');

ResultStruct.LyapunovWolf_V = LyapunovWolf(1);
ResultStruct.LyapunovWolf_ML = LyapunovWolf(2);
ResultStruct.LyapunovWolf_AP = LyapunovWolf(3);
ResultStruct.LyapunovRosen_V = LyapunovRosen(1);
ResultStruct.LyapunovRosen_ML = LyapunovRosen(2);
ResultStruct.LyapunovRosen_AP = LyapunovRosen(3);
ResultStruct.SampleEntropy_V = SampleEntropy(1);
ResultStruct.SampleEntropy_ML = SampleEntropy(2);
ResultStruct.SampleEntropy_AP = SampleEntropy(3);

if isfield(ResultStruct,'StrideFrequency')
    LyapunovPerStrideWolf = LyapunovWolf/ResultStruct.StrideFrequency;
    LyapunovPerStrideRosen = LyapunovRosen/ResultStruct.StrideFrequency;
end
ResultStruct.LyapunovPerStrideWolf_V = LyapunovPerStrideWolf(1);
ResultStruct.LyapunovPerStrideWolf_ML = LyapunovPerStrideWolf(2);
ResultStruct.LyapunovPerStrideWolf_AP = LyapunovPerStrideWolf(3);
ResultStruct.LyapunovPerStrideRosen_V = LyapunovPerStrideRosen(1);
ResultStruct.LyapunovPerStrideRosen_ML = LyapunovPerStrideRosen(2);
ResultStruct.LyapunovPerStrideRosen_AP = LyapunovPerStrideRosen(3);

end
