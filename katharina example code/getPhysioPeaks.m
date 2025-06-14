function getPhysioPeaks

% get paths, filenames, participant IDs etc.
[paths,data] = getDataSpecs();

% initialize empty matrix
ppu_perTrial = zeros(10000,120);

%--- SAP
% ppuAfterOutcome + stimuluscode
% ppuAfterPE + outcomeface + stimuluscode
% ppuAfterPE

%--- AAA
% ppuAfterOutcome + stimuluscode


for t = 1:data.dataSet.nTasks
    currTask = data.dataSet.tasks{t};
    for n = 1:data.dataSet.nParticipants
        currPID =  data.dataSet.PIDs(n);
        disp(['reading ',num2str(currPID),' ',num2str(currTask)]);

        % get dataFile containing participant's responses and event timings
        load(paths.participant(n).task(t,1).dataFile);
        predField = [currTask,'Prediction'];
        PEtrials = find(dataFile.(predField).congruent==-1);
        participantActionTrials = find(dataFile.(predField).response(:,1)==1);

        % get optionsFile containing participant's task settings
        load(paths.participant(n).task(t,1).optsFile);
        stimulusActionTrials = find(options.task.inputs(:,2)==1);

        % load data file for this participant
        f = dir([paths.participant(n).periphDir,num2str(currPID),currTask,'_ppu.mat']);
        if ~isempty(f)
            load([paths.participant(n).periphDir,num2str(currPID),currTask,'_ppu.mat']);
            % smooth data using Savitzy-Golay Filter
            ppuData = smooth(ppu_data.data,'sgolay');

            ppuTime = ppu_data.time; % make time strings for stringcomp with task timings
            ppuTime.Format = 'hh:mm:ss'; ppuTime = string(ppuTime);
            % get peaks , i.e. maximum value within 50 points on x-axis
            [peaks,locations,~,~] = findpeaks(ppuData,'MinPeakDistance',50);

            % the first few datasets where saved in hh:mm:ss format, thus there
            % are several instances of the same time in the matrix. Use this to
            % add miliseconds to the times
            if strcmp(ppu_data.time.Format,'hh:mm:ss')
                % find out where a new timestamp starts
                diffvec = ppu_data.time(2:end)-ppu_data.time(1:end-1);
                newTime = logical(seconds(diffvec)); % make it a logical array
                ppu_data.time.Format = 'hh:mm:ss.SSS'; % convert into datetime format containing ms

                iBlock=1; % iBlock for blocks of the same timestamp recorded
                for i = 1:numel(newTime) % loop through time vector
                    if newTime(i)==1 % for the first block
                        if iBlock == 1
                            blockSize(iBlock) = i;
                            addMS = 1000/blockSize(iBlock);
                            for j = 1:blockSize(iBlock)
                                if j ==1
                                    ppu_data.time(j) = ppu_data.time(j)+milliseconds(addMS);
                                else
                                    ppu_data.time(j) = ppu_data.time(j-1)+milliseconds(addMS);
                                end
                            end
                            iBlock = iBlock+1;
                        else % for all the other blocks
                            blockSize(iBlock) = i-sum(blockSize);
                            iNewStart = sum(blockSize(1:iBlock-1))+1;
                            addMS = 1000/blockSize(iBlock);
                            for j = iNewStart:i
                                if j ==iNewStart
                                    ppu_data.time(j) = ppu_data.time(j)+milliseconds(addMS);
                                else
                                    ppu_data.time(j) = ppu_data.time(j-1)+milliseconds(addMS);
                                end
                            end
                            iBlock = iBlock+1;
                        end
                    end
                end % END time vector loop
                clear iBlock;
                clear blockSize;
            end % END datetime format exception loop

            % Find ppu vector indices for ppu time windows on each trial
            for iTime = 1:numel(dataFile.events.outcome_startTime)
                % find ppu time index at the same time the outcome was presented
                ids = find(strcmp(dataFile.events.outcome_startTime(iTime),ppuTime));
                ppu_outcomeIds(iTime) = ids(1);
                ids = find(strcmp(dataFile.events.iti_startTime(iTime),ppuTime));
                ppu_itiIds(iTime) = ids(1);

                if iTime < numel(dataFile.events.outcome_startTime)
                    ids = find(strcmp(dataFile.events.stimulus_startTime(iTime+1),ppuTime));
                    ppu_newStimIds(iTime) = ids(1);
                else
                    ppu_newStimIds(iTime) = ppu_newStimIds(iTime-1)+1000;
                end
            end

            for iTrial = 1:numel(dataFile.events.outcome_startTime)
                startIdx = ppu_outcomeIds(iTrial);
                stopIdx = ppu_newStimIds(iTrial);
                currData = ppuData(startIdx:stopIdx);
                ppu_perTrial(:,iTrial) = [currData;zeros(10000-numel(currData),1)];
                maxPPUs(iTrial)    = max(currData);
                peakIds = locations(locations>startIdx); peakIds = peakIds(peakIds<stopIdx);
                trialDurs(iTrial)  = ppu_data.time(stopIdx )-ppu_data.time(startIdx);
                nPeaks(iTrial)     = numel(peakIds);
                timeWindow(iTrial) = ppu_data.time(stopIdx)-ppu_data.time(startIdx);
            end

            durations = seconds(trialDurs);
            HR_all = (nPeaks./durations)*60;
            HR_allTrials_mean(n) = mean(HR_all);
            HR_allTrials_min(n)  = min(HR_all);
            HR_allTrials_max(n)  = max(HR_all);
            HR_PEtrials_mean(n)  = mean(HR_all(PEtrials));
            HR_PEtrials_min(n)   = min(HR_all(PEtrials));
            HR_PEtrials_max(n)   = max(HR_all(PEtrials));
            amp_allTrials_mean(n)= mean(maxPPUs);
            amp_allTrials_min(n) = min(maxPPUs);
            amp_allTrials_max(n) = max(maxPPUs);

            if strcmp(currTask,'SAP')
                HR_StimSmiletrials_mean(n) = mean(HR_all(stimulusActionTrials));
                HR_StimSmiletrials_min(n)  = min(HR_all(stimulusActionTrials));
                HR_StimSmiletrials_max(n)  = max(HR_all(stimulusActionTrials));
                amp_StimSmiletrials_mean(n) = mean(maxPPUs(stimulusActionTrials));
                amp_StimSmiletrials_min(n)  = min(maxPPUs(stimulusActionTrials));
                amp_StimSmiletrials_max(n)  = max(maxPPUs(stimulusActionTrials));

                HR_partSmiletrials_mean(n) = mean(HR_all(participantActionTrials));
                HR_partSmiletrials_min(n)  = min(HR_all(participantActionTrials));
                HR_partSmiletrials_max(n)  = max(HR_all(participantActionTrials));
                amp_partSmiletrials_mean(n) = mean(maxPPUs(participantActionTrials));
                amp_partSmiletrials_min(n)  = min(maxPPUs(participantActionTrials));
                amp_partSmiletrials_max(n)  = max(maxPPUs(participantActionTrials));

            elseif strcmp(currTask,'SAPC')
                HR_StimGoodtrials_mean(n) = mean(HR_all(stimulusActionTrials));
                HR_StimGoodtrials_min(n)  = min(HR_all(stimulusActionTrials));
                HR_StimGoodtrials_max(n)  = max(HR_all(stimulusActionTrials));
                amp_StimGoodtrials_mean(n) = mean(maxPPUs(stimulusActionTrials));
                amp_StimGoodtrials_min(n)  = min(maxPPUs(stimulusActionTrials));
                amp_StimGoodtrials_max(n)  = max(maxPPUs(stimulusActionTrials));

                HR_partCollecttrials_mean(n) = mean(HR_all(participantActionTrials));
                HR_partCollecttrials_min(n)  = min(HR_all(participantActionTrials));
                HR_partCollectrials_max(n)   = max(HR_all(participantActionTrials));
                amp_partCollecttrials_mean(n) = mean(maxPPUs(participantActionTrials));
                amp_partCollecttrials_min(n)  = min(maxPPUs(participantActionTrials));
                amp_partCollectrials_max(n)   = max(maxPPUs(participantActionTrials));

            elseif strcmp(currTask,'AAA')
                HR_partApproachtrials_mean(n) = mean(HR_all(participantActionTrials));
                HR_partApproachtrials_min(n)  = min(HR_all(participantActionTrials));
                HR_partApproachtrials_max(n)  = max(HR_all(participantActionTrials));
                amp_partApproachtrials_mean(n) = mean(maxPPUs(participantActionTrials));
                amp_partApproachtrials_min(n)  = min(maxPPUs(participantActionTrials));
                amp_partApproachtrials_max(n)  = max(maxPPUs(participantActionTrials));
            end
        else
            if strcmp(currTask,'SAP')
                HR_StimSmiletrials_mean(n) = NaN;
                HR_StimSmiletrials_min(n)  = NaN;
                HR_StimSmiletrials_max(n)  = NaN;
                amp_StimSmiletrials_mean(n) = NaN;
                amp_StimSmiletrials_min(n)  = NaN;
                amp_StimSmiletrials_max(n)  = NaN;

                HR_partSmiletrials_mean(n) = NaN;
                HR_partSmiletrials_min(n)  = NaN;
                HR_partSmiletrials_max(n)  = NaN;
                amp_partSmiletrials_mean(n) = NaN;
                amp_partSmiletrials_min(n)  = NaN;
                amp_partSmiletrials_max(n)  = NaN;

            elseif strcmp(currTask,'SAPC')
                HR_StimGoodtrials_mean(n) = NaN;
                HR_StimGoodtrials_min(n)  = NaN;
                HR_StimGoodtrials_max(n)  = NaN;
                amp_StimGoodtrials_mean(n) = NaN;
                amp_StimGoodtrials_min(n)  = NaN;
                amp_StimGoodtrials_max(n)  = NaN;

                HR_partCollecttrials_mean(n) = NaN;
                HR_partCollecttrials_min(n)  = NaN;
                HR_partCollectrials_max(n)   = NaN;
                amp_partCollecttrials_mean(n) = NaN;
                amp_partCollecttrials_min(n)  = NaN;
                amp_partCollectrials_max(n)   = NaN;

            elseif strcmp(currTask,'AAA')
                HR_partApproachtrials_mean(n) = NaN;
                HR_partApproachtrials_min(n)  = NaN;
                HR_partApproachtrials_max(n)  = NaN;
                amp_partApproachtrials_mean(n) = NaN;
                amp_partApproachtrials_min(n)  = NaN;
                amp_partApproachtrials_max(n)  = NaN;
            end
        end
    end

if strcmp(currTask,'SAP')
SAP_HRTable = table(data.dataSet.PIDs,HR_allTrials_mean,HR_allTrials_min,HR_allTrials_max,...
    HR_PEtrials_mean, HR_PEtrials_min, HR_PEtrials_max,HR_StimSmiletrials_mean,R_StimSmiletrials_min,...
    HR_StimSmiletrials_max,HR_partSmiletrials_mean,HR_partSmiletrials_min,HR_partSmiletrials_max,'VariableNames',...
        {'ID','meanHR_all','minHR_all','maxHR_all','meanHR_PE','minHR_PE','maxHR_PE','meanHR_stimSmile',...
        'minHR_stimSmile','maxHR_stimSmile','meanHR_participantSmile',...
        'minHR_participantSmile','maxHR_participantSmile'});

save([paths.group.questData ,'SAP_HRTable.mat'],'SAP_HRTable');
writetable(SAP_HRTable,[paths.group.questData,'SAP_HRTable.csv']);

SAP_AmplitudeTable = table(data.dataSet.PIDs,amp_allTrials_mean,amp_allTrials_min,amp_allTrials_max,...
    amp_PEtrials_mean, HR_PEtrials_min, HR_PEtrials_max,HR_StimSmiletrials_mean,R_StimSmiletrials_min,...
    amp_StimSmiletrials_max,amp_partSmiletrials_mean,amp_partSmiletrials_min,amp_partSmiletrials_max,'VariableNames',...
        {'ID','meanHR_all','minHR_all','maxHR_all','meanHR_PE','minHR_PE','maxHR_PE','meanHR_stimSmile',...
        'minAmp_stimSmile','maxAmp_stimSmile','meanAmp_participantSmile',...
        'minAmp_participantSmile','maxAmp_participantSmile'});

save([paths.group.questData ,'SAP_AmplitudeTable.mat'],'SAP_AmplitudeTable');
writetable(SAP_AmplitudeTable,[paths.group.questData,'SAP_AmplitudeTable.csv']);
end


end