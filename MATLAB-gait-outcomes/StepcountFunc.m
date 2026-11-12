function [PksAndLocsCorrected] = StepcountFunc(dataAcc_filt,StrideTimeSamples,FS);

% Step count funciton extracted from other function called: 

Cutoff = 0.1;
MinDist = floor(0.7*0.5*StrideTimeSamples);  % Use StrideTimeSamples estimated above
DataLFilt = dataAcc_filt; 

% From acceleration to 
Vel = cumsum(detrend(DataLFilt,'constant'))/FS;
[B,A] = butter(2,Cutoff/(FS/2),'high');
Pos = cumsum(filtfilt(B,A,Vel))/FS;
PosFilt = filtfilt(B,A,Pos);
PosFiltVT = PosFilt(:,1);

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

end