
function [ResultStruct] = HarmonicityFrequency(dataAccCut_filt,P,F, StrideFrequency,dF,LowFrequentPowerThresholds,N_Harm,FS,AccVectorLen,ResultStruct)

% Add sum of power spectra (as a rotation-invariant spectrum)
P = [P,sum(P,2)];
PS = sqrt(P);

% Calculate the measures for the power per separate dimension
for i=1:size(P,2);
    % Relative cumulative power and frequencies that correspond to these cumulative powers
    PCumRel = cumsum(P(:,i))/sum(P(:,i));
    PSCumRel = cumsum(PS(:,i))/sum(PS(:,i));
    FCumRel = F+0.5*dF;
    
    % Derive relative cumulative power for threshold frequencies
    Nfreqs = size(LowFrequentPowerThresholds,2);
    LowFrequentPercentage(i,1:Nfreqs) = interp1(FCumRel,PCumRel,LowFrequentPowerThresholds)*100;
    
    % Calculate relative power of first 10 harmonics, taking the power of each harmonic with a band of + and - 10% of the first
    % harmonic around it
    PHarm = zeros(N_Harm,1);
    PSHarm = zeros(N_Harm,1);
    for Harm = 1:N_Harm,
        FHarmRange = (Harm+[-0.1 0.1])*StrideFrequency;
        PHarm(Harm) = diff(interp1(FCumRel,PCumRel,FHarmRange));
        PSHarm(Harm) = diff(interp1(FCumRel,PSCumRel,FHarmRange));
    end
    
    % Derive index of harmonicity
    if i == 2 % for ML we expect odd instead of even harmonics
        IndexHarmonicity(i) = PHarm(1)/sum(PHarm(1:2:12));
    elseif i == 4
        IndexHarmonicity(i) = sum(PHarm(1:2))/sum(PHarm(1:12));
    else
        IndexHarmonicity(i) = PHarm(2)/sum(PHarm(2:2:12));
    end
    
    % Calculate the phase speed fluctuations
    StrideFreqFluctuation = nan(N_Harm,1);
    StrSamples = round(FS/StrideFrequency);
    for h=1:N_Harm,
        CutOffs = [StrideFrequency*(h-(1/3)) , StrideFrequency*(h+(1/3))]/(FS/2);
        if all(CutOffs<1) % for Stride frequencies above FS/20/2, the highest harmonics are not represented in the power spectrum
            [b,a] = butter(2,CutOffs);
            if i==4 % Take the vector length as a rotation-invariant signal
                AccFilt = filtfilt(b,a,AccVectorLen);
            else
                AccFilt = filtfilt(b,a,dataAccCut_filt(:,i));
            end
            Phase = unwrap(angle(hilbert(AccFilt)));
            SmoothPhase = sgolayfilt(Phase,1,2*(floor(FS/StrideFrequency/2))+1); % This is in fact identical to a boxcar filter with linear extrapolation at the edges
            StrideFreqFluctuation(h) = std(Phase(1+StrSamples:end,1)-Phase(1:end-StrSamples,1));
        end
    end
    FrequencyVariability(i) = nansum(StrideFreqFluctuation./(1:N_Harm)'.*PHarm)/nansum(PHarm);
    
    if i<4,
        % Derive harmonic ratio (two variants)
        if i == 2 % for ML we expect odd instead of even harmonics
            HarmonicRatio(i) = sum(PSHarm(1:2:end-1))/sum(PSHarm(2:2:end)); % relative to summed 3d spectrum
            HarmonicRatioP(i) = sum(PHarm(1:2:end-1))/sum(PHarm(2:2:end)); % relative to own spectrum
        else
            HarmonicRatio(i) = sum(PSHarm(2:2:end))/sum(PSHarm(1:2:end-1));
            HarmonicRatioP(i) = sum(PHarm(2:2:end))/sum(PHarm(1:2:end-1));
        end
    end
    
end

    ResultStruct.IndexHarmonicity_V = IndexHarmonicity(1); % higher smoother more regular patter
    ResultStruct.IndexHarmonicity_ML = IndexHarmonicity(2); % higher smoother more regular patter
    ResultStruct.IndexHarmonicity_AP = IndexHarmonicity(3); % higher smoother more regular patter
    ResultStruct.IndexHarmonicity_All = IndexHarmonicity(4);
    ResultStruct.HarmonicRatio_V = HarmonicRatio(1);
    ResultStruct.HarmonicRatio_ML = HarmonicRatio(2);
    ResultStruct.HarmonicRatio_AP = HarmonicRatio(3);
    ResultStruct.HarmonicRatioP_V = HarmonicRatioP(1);
    ResultStruct.HarmonicRatioP_ML = HarmonicRatioP(2);
    ResultStruct.HarmonicRatioP_AP = HarmonicRatioP(3);
    ResultStruct.FrequencyVariability_V = FrequencyVariability(1); 
    ResultStruct.FrequencyVariability_ML = FrequencyVariability(2);
    ResultStruct.FrequencyVariability_AP = FrequencyVariability(3); 
    ResultStruct.StrideFrequency = StrideFrequency;