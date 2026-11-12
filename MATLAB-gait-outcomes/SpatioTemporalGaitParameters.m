function [ResultStruct] = SpatioTemporalGaitParameters(dataAccCut_filt,StrideTimeSamples,ApplyRealignment,LegLength,FS,IgnoreMinMaxStrides,ResultStruct);

Cutoff = 0.1;
MinDist = floor(0.7*0.5*StrideTimeSamples);  % Use StrideTimeSamples estimated above
DatalFilt = dataAccCut_filt; 

% From acceleration to velocity
Vel = cumsum(detrend(DatalFilt,'constant'))/FS;
[B,A] = butter(2,Cutoff/(FS/2),'high');
Pos = cumsum(filtfilt(B,A,Vel))/FS;
PosFilt = filtfilt(B,A,Pos);
PosFiltVT = PosFilt(:,1);

% Find minima and maxima in vertical position

% if ~ApplyRealignment % Signals were not realigned, so it has to be done here
%     MeanAcc = mean(AccLoco);
%     VT = MeanAcc'/norm(MeanAcc);
%     PosFiltVT = PosFilt*VT;
% end

% Find minima and maxima in vertical position
[PosPks,PosLocs] = findpeaks(PosFiltVT(:,1),'minpeakdistance',MinDist);
[NegPks,NegLocs] = findpeaks(-PosFiltVT(:,1),'minpeakdistance',MinDist);
NegPks = -NegPks;
if isempty(PosPks) && isempty(NegPks)
    PksAndLocs = zeros(0,3);
else
    PksAndLocs = sortrows([PosPks,PosLocs,ones(size(PosPks)) ; NegPks,NegLocs,-ones(size(NegPks))], 2);
end
% Correct events for two consecutive maxima or two consecutive minima
Events = PksAndLocs(:,2);
NewEvents = PksAndLocs(:,2);
Signs = PksAndLocs(:,3);
FalseEventsIX = find(diff(Signs)==0);
PksAndLocsToAdd = zeros(0,3);
PksAndLocsToAddNr = 0;
for i=1:numel(FalseEventsIX),
    FIX = FalseEventsIX(i);
    if FIX <= 2
        % remove the event
        NewEvents(FIX) = nan;
    elseif FIX >= numel(Events)-2
        % remove the next event
        NewEvents(FIX+1) = nan;
    else
        StrideTimesWhenAdding = [Events(FIX+1)-Events(FIX-2),Events(FIX+3)-Events(FIX)];
        StrideTimesWhenRemoving = Events(FIX+3)-Events(FIX-2);
        if max(abs(StrideTimesWhenAdding-StrideTimeSamples)) < abs(StrideTimesWhenRemoving-StrideTimeSamples)
            % add an event
            [M,IX] = min(Signs(FIX)*PosFiltVT((Events(FIX)+1):(Events(FIX+1)-1)));
            PksAndLocsToAddNr = PksAndLocsToAddNr+1;
            PksAndLocsToAdd(PksAndLocsToAddNr,:) = [M,Events(FIX)+IX,-Signs(FIX)];
        else
            % remove an event
            if FIX >= 5 && FIX <= numel(Events)-5
                ExpectedEvent = (Events(FIX-4)+Events(FIX+5))/2;
            else
                ExpectedEvent = (Events(FIX-2)+Events(FIX+3))/2;
            end
            if abs(Events(FIX)-ExpectedEvent) > abs(Events(FIX+1)-ExpectedEvent)
                NewEvents(FIX) = nan;
            else
                NewEvents(FIX+1) = nan;
            end
        end
    end
end
PksAndLocsCorrected = sortrows([PksAndLocs(~isnan(NewEvents),:);PksAndLocsToAdd],2);
% Find delta height and delta time
DH = abs(diff(PksAndLocsCorrected(:,1),1,1));
DT = diff(PksAndLocsCorrected(:,2),1,1);
% Correct outliers in delta h
MaxDH = min(median(DH)+3*mad(DH,1),LegLength/2);
DH(DH>MaxDH) = MaxDH;
% Estimate total length and speed
% (Use delta h and delta t to calculate walking speed: use formula from
% Z&H, but divide by 2 (skip factor 2)since we get the difference twice
% each step, and multiply by 1.25 which is the factor suggested by Z&H)
HalfStepLen = 1.25*sqrt(2*LegLength*DH-DH.^2);
Distance = sum(HalfStepLen);
WalkingSpeedMean = Distance/(sum(DT)/FS);
% Estimate variabilities between strides
StrideLengths = HalfStepLen(1:end-3) + HalfStepLen(2:end-2) + HalfStepLen(3:end-1) + HalfStepLen(4:end);
StrideTimes = PksAndLocsCorrected(5:end,2)-PksAndLocsCorrected(1:end-4,2);
StrideSpeeds = StrideLengths./(StrideTimes/FS);
WSS = nan(1,4);
STS = nan(1,4);
for i=1:4,
    STS(i) = std(StrideTimes(i:4:end))/FS;
    WSS(i) = std(StrideSpeeds(i:4:end));
end

StepLengthMean=mean(StrideLengths);

StrideTimeVariability = min(STS);
StrideSpeedVariability = min(WSS);
StrideLengthVariability = std(StrideLengths);
% Estimate Stride time variability and stride speed variability by removing highest and lowest part
if ~isinteger(IgnoreMinMaxStrides)
    IgnoreMinMaxStrides = ceil(IgnoreMinMaxStrides*size(StrideTimes,1));
end
StrideTimesSorted = sort(StrideTimes);
StrideTimeVariabilityOmitOutlier = std(StrideTimesSorted(1+IgnoreMinMaxStrides:end-IgnoreMinMaxStrides));
StrideSpeedSorted = sort(StrideSpeeds);
StrideSpeedVariabilityOmitOutlier = std(StrideSpeedSorted(1+IgnoreMinMaxStrides:end-IgnoreMinMaxStrides));
StrideLengthsSorted = sort(StrideLengths);
StrideLengthVariabilityOmitOutlier = std(StrideLengthsSorted(1+IgnoreMinMaxStrides:end-IgnoreMinMaxStrides));

ResultStruct.StepLengthMean = StepLengthMean;
ResultStruct.Distance = Distance; 
ResultStruct.WalkingSpeedMean = WalkingSpeedMean; 
ResultStruct.StrideTimeVariability = StrideTimeVariability;
ResultStruct.StrideSpeedVariability = StrideSpeedVariability;
ResultStruct.StrideLengthVariability = StrideLengthVariability;
ResultStruct.StrideTimeVariabilityOmitOutlier = StrideTimeVariabilityOmitOutlier;
ResultStruct.StrideSpeedVariabilityOmitOutlier = StrideSpeedVariabilityOmitOutlier;
ResultStruct.StrideLengthVariabilityOmitOutlier = StrideLengthVariabilityOmitOutlier;