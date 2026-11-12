function [ResultStruct] = GaitVariabilityAnalysisIH_WithoutTurns(inputData,FS,LegLength,ApplyRealignment,ApplyRemoveSteps);


% SCRIPT FOR ANAlysis straight parts
% NOG GOEDE BESCHRIJVING TOEVOEGEN.

%% Realign data
%data = inputData(:, [3,2,4]); % reorder data to  1 = V; 2= ML, 3 = AP

%Realign sensor data to VT-ML-AP frame
if ApplyRealignment % apply relignment as described in Rispens S, Pijnappels M, van Schooten K, Beek PJ, Daffertshofer A, van Die?n JH (2014).
    % Consistency of gait characteristics as determined from acceleration data collected at different trunk locations. Gait Posture 2014;40(1):187-92.
    [RealignedAcc, ~] = RealignSensorSignalHRAmp(data, FS);
    dataAcc = RealignedAcc;
else 
    dataAcc = inputData;
end

%% Filter data strongly & Determine location of steps

% Filter  data
[B,A] = butter(2,3/(FS/2),'low'); % Filters data very strongly which is needed to determine turns correctly
dataStepDetection = filtfilt(B,A,dataAcc);

% Determine steps;

%%%%%%% HIER MISSCHIEN ALTERNATIEF VOOR VAN RISPENS %%%%%%%%%%%%%

% Explanation of method: https://nl.mathworks.com/help/supportpkg/beagleboneblue/ref/counting-steps-using-beagleboneblue-hardware-example.html
% From website: To convert the XYZ acceleration vectors at each point in time into scalar values,
% calculate the magnitude of each vector. This way, you can detect large changes in overall acceleration,
% such as steps taken while walking, regardless of device orientation.

magfilt = sqrt(sum((dataStepDetection(:,1).^2) + (dataStepDetection(:,2).^2) + (dataStepDetection(:,3).^2), 2));
magNoGfilt = magfilt - mean(magfilt);
minPeakHeight2 = std(magNoGfilt);
[pks, locs] = findpeaks(magNoGfilt, 'MINPEAKHEIGHT', minPeakHeight2); % for step detection
numStepsOption2_filt = numel(pks); % counts number of steps;

%%%%%%%%%%%%%%%%%%%%%%%% TOT HIER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Determine locations of turns;

diffLocs = diff(locs); % calculates difference in step location
avg_diffLocs = mean(diffLocs); % average distance between steps
std_diffLocs = std(diffLocs); % standard deviation of distance between steps

figure;
findpeaks(diffLocs, 'MINPEAKHEIGHT', avg_diffLocs, 'MINPEAKDISTANCE',5); % these values have been chosen based on visual inspection of the signal
line([1 length(diffLocs)],[avg_diffLocs avg_diffLocs])
[pks_diffLocs, locs_diffLocs] = findpeaks(diffLocs, 'MINPEAKHEIGHT', avg_diffLocs,'MINPEAKDISTANCE',5);
locsTurns =  [locs(locs_diffLocs),  locs(locs_diffLocs+1)];

%% Visualizing turns

% Duplying signal + visualing
% to make second signal with the locations of the turns filled with NaN, so
% that both signals can be plotted above each other in a different colour

magNoGfilt_copy = magNoGfilt;
for k = 1: size(locsTurns,1);
    magNoGfilt_copy(locsTurns(k,1):locsTurns(k,2)) = NaN;
end


% visualising signal;
figure;
subplot(2,1,1)
hold on;
plot(magNoGfilt,'b')
plot(magNoGfilt_copy, 'r');
title('Inside Straight: Filtered data with turns highlighted in blue')
hold off;

%% Calculation
% VRAAG LAURENS zie blauwe blaadje

startPos = 1;
for i = 1: size(locsTurns,1);
    endPos = locsTurns(i,1)-1;
    
    inputData = dataAcc(startPos:endPos,:);
    WindowLen = size(inputData,1);
    ApplyRealignment = false;
    [ResultStruct] = GaitOutcomesTrunkAccFuncIH(inputData,FS,LegLength,WindowLen,ApplyRealignment,ApplyRemoveSteps); % Naam van deze moet nog aangepast.
    
    if i ==1 % only the firs time
        Parameters = fieldnames(ResultStruct);
        NrParameters = length(Parameters);
    end
    
    for j = 1:NrParameters % only works if for every bin we get the same outcomes (which is the case in this script)
        DataStraight.([char(Parameters(j))])(i) = ResultStruct.([char(Parameters(j))]);
    end
    startPos = locsTurns(i,2)+1;
    
end

clear ResultStruct;

% Calculate mean over the bins without turns to get 1 outcome value per parameter for inside
% straight;

for j = 1:NrParameters;
    ResultStruct.([char(Parameters(j))]) = nanmean(DataStraight.([char(Parameters(j))]))
end


end

