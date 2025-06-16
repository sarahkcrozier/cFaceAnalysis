function segmentsTableIncongruent = cFaceIncongruentTrials(participantID)

%% extract cFace incongruent trial timings
%% see segmentsTableIncongruent
%% needs to be merged with cFacePupilData.m

options = specifyOptions;

%% save text of Matlab session
diary(fullfile(options.paths.analysis,'output_precressingpupils_cface.txt'))

% Format participant ID
IDstring = sprintf('%03d', participantID);

% Find matching folder
participantFolders = dir(options.paths.data);
folderNames = {participantFolders.name};
folderNames = folderNames(~ismember(folderNames, {'.','..'}));  % skip system entries

matchIdx = contains(folderNames, IDstring);
if ~any(matchIdx)
    error('No matching folder found for participant ID %s', IDstring);
end
folderName = folderNames{find(matchIdx, 1)};
folderPath = fullfile(options.paths.data, folderName);
    
    % Ignore '.' and '..'
    if folderName(1) == '.' || ~participantFolders(participantID).isdir==1
        %continue;
        return;
    else
        f = fullfile(folderPath, 'beh','cface*MHE*','*out.csv');
        summaryFile = dir(f);
           disp(summaryFile)
        if isempty(summaryFile)
            %continue;
        return;
        else
            summaryFilePath = fullfile(summaryFile.folder, summaryFile.name);
            splitPath = split(folderPath,'/');
            splitPath = split(splitPath{end},'_');
            subjectNumber =  splitPath(1); % note modified from splitpath(2) to splitpath(1)
         
            
            %% get info about trial segments
           
            % Detect the import options of the file
            opts = detectImportOptions(strcat(summaryFile.folder,'/',summaryFile.name));
            % Specify the variable names to import - timestamps,
            % congruentTrials (incongruent==0), participantEmotion
            opts.SelectedVariableNames = {'ParticipantID', 'stimMove_onset', 'fixation_onset', ... 
                'cong', 'ptemot', 'trigger_onset'};
            % Read the table
            segmentsTable = readtable(strcat(summaryFile.folder,'/',summaryFile.name), opts);
            segmentsTable.trialNo = (1:height(segmentsTable))';

            % Initialise 'time to next trigger' column with NaNs
            segmentsTable.timeToNextTrigger = NaN(height(segmentsTable),1);
            
            % Calculate time difference between stimMove_onset of trial i and trigger_onset of trial i+1
            for i = 1:height(segmentsTable)-1
                segmentsTable.timeToNextTrigger(i) = ...
                    segmentsTable.trigger_onset(i+1) - segmentsTable.stimMove_onset(i);
            end
            
            % Assign a default value (e.g., 4s) for the final trial
            segmentsTable.timeToNextTrigger(end) = 4;

            segmentsTableIncongruent = segmentsTable(segmentsTable.cong == 0, :)
            
            

%{    
           
            % downsampling to 100 Hz
            downsampled_time = downsample(dataSmoothInterp.data.t, 10);
            downsampled_data = downsample(dataSmoothInterp.data.pupilValues, 10);
            
            baselinemat = [];
            valueTable = [];
            number = [];
            segmentTimesteps = 400;%4 s

            for segmentIndx = 1:height(segmentsTable)

                if segmentsTable.trigger_onset(segmentIndx) > 0

                    condSegmentsStart = segmentsTable.trigger_onset(segmentIndx);
                    baselineSegmentsStart = condSegmentsStart - 0.1;
                    % Get segment indexes of smooth signal:
                    curSignalSection = find(...
                        downsampled_time >= condSegmentsStart,1);
                    baselineSection = ...
                        (downsampled_time >= baselineSegmentsStart)...
                        &(downsampled_time < condSegmentsStart);
                    baseline = median(downsampled_data(baselineSection));
                    newColumn = downsampled_data(curSignalSection:curSignalSection+segmentTimesteps)/baseline;
                    valueTable = [valueTable, newColumn(1:segmentTimesteps)];
                    number = [number, segmentsTable.trialNo(segmentIndx)];
                    baselinemat = [baselinemat, baseline];
                end
            end

            % Calculate the mean and standard deviation
            baseline_mean = mean(baselinemat);
            baseline_std = std(baselinemat);
            
            % Standardize the vector
            baseline_z  = (baselinemat - baseline_mean) / baseline_std;

            if width(valueTable)<80
                missingTrails = 80 - width(valueTable);
                fprintf('%i trial(s) excluded',missingTrails)
            end

            % Write to CSV
            csvPath = fullfile(options.paths.data,strcat(subjectNumber,...
                '_cface_BaselineRatioCorrPupil.csv'));
            writematrix([ [NaN, number]; ...
                ((0:segmentTimesteps-1)./100)', valueTable], csvPath{1});

            % Assuming valueTable is your table, converted from matrix to table
            baselineTable = array2table([number',baselinemat', baseline_z']);
            % Create the column names
            colNames = {'trial_number', 'baseline_pupil', 'z_score'};
            
            % Assign the column names to the table
            baselineTable.Properties.VariableNames = colNames;

            csvPath = fullfile(options.paths.data,strcat(subjectNumber,...
                '_cface_MedianBaseline.csv'));
            writetable(baselineTable, csvPath{1});
%}
    
            %figure;
            %for k = 1:width(valueTable)
            %    plot((0:segmentTimesteps-1)./100,valueTable(:,k)) 
            %    hold on
            %end
            
            % Specify the axis labels
            %xlabel('Time after trigger onset, s')
            %ylabel('Baseline-corrected pupil diameter, a.u.')
            
            % Specify the size of the figure
            %set(gcf, 'Units', 'Inches', 'Position', [0, 0, 9, 6])
            %figurePath = fullfile(options.paths.data,strcat(subjectNumber,'_PupilTraces.jpg'));
            % Save the figure
            %saveas(gcf, figurePath{1});
            
            % Close the current figure
            %close(gcf)
            
            %figure;
            %histogram(baselinemat)
            
            % Specify the axis labels
            %xlabel('Baseline pupil diameter')
            %ylabel('Counts')
            
            % Specify the size of the figure
            %set(gcf, 'Units', 'Inches', 'Position', [0, 0, 9, 6])
            
            % Save the figure
            %histPath = fullfile(directory,strcat(subjectNumber,'_BaselineHistogram.jpg'));
            %saveas(gcf, histPath{1});
            
            % Close the current figure
            %close(gcf)
        end
    end
%end

diary off










