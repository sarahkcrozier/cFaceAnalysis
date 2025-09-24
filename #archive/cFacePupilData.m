%% extract participant data from RedCap and Heart Beat Discrimination Analysis file
%% extract cFace incongruent trial pupil response data (average for all incongruent trials, peak amplitude in x second window after stimulus response)

options.study.acronym = 'PCNS';
options.study.tasks = {'cFace','FF1','HBD'};

options.paths.workingDir = pwd;
options.paths.analysis   = ['/Users/yamaan/Projects/',options.study.acronym,filesep,options.study.tasks{1},'Analysis',filesep];
options.paths.data       = ['/Users/yamaan/Projects/',options.study.acronym,filesep,'Data/TestData',filesep]; %temp local data path for testing

%% get files
participantFolders = dir(options.paths.data);

%% save text of Matlab session
diary(fullfile(options.paths.analysis,'output_precressingpupils_cface.txt'))

% Loop over Participant folders
for i = 1:length(participantFolders)
    % Get folder name and path
    folderName = participantFolders(i).name;
    folderPath = fullfile(options.paths.data, folderName);
    
    % Ignore '.' and '..'
    if folderName(1) == '.' || ~participantFolders(i).isdir==1
        continue;
    else
        f = fullfile(folderPath, 'beh','cface*','*.edf');
        edfFiles = dir(f);
        if isempty(edfFiles)
            continue
        else
            edfFilePath = fullfile(edfFiles.folder, edfFiles.name);
            splitPath = split(folderPath,'/');
            splitPath = split(splitPath{end},'_');
            subjectNumber =  splitPath(1); % note modified from splitpath(2) to splitpath(1)
            %% extract EDF into .mat  with Edf2Mat
    
            disp(strcat('Converting: ',edfFilePath))
            edf = edfread(edfFilePath);
    
            % check if pupil diameter was used
            if edf.PUPIL.DIAMETER == 1 
                disp('Values are given as pupil diameter in a.u.')
            elseif edf.PUPIL.AREA == 1
                disp('Values are given as pupil area in a.u.')
            end
    
            % check used eye
            L_raw           = double(edf.Samples.pa(:,1));
            R_raw           = double(edf.Samples.pa(:,2));
            if strcmp(char(edf.Events.Start.info),'BINOCULAR')
                error('Binocular data')
            elseif strcmp(char(edf.Events.Start.info),'RIGHT')
                pupilValues = R_raw;
                disp('Right eye was used')
            elseif strcmp(char(edf.Events.Start.info),'LEFT')
                pupilValues = L_raw;
                disp('Left eye was used')
            end
            curEye = char(edf.Events.Start.info);
    
            % set missing pupil values to NaN
            pupilValues(pupilValues==0) = NaN;
            
            % get sync time
            sync_ind=find(strcmp(edf.Events.Messages.info,'SyncPulseReceived'));
            sync_time=edf.Events.Messages.time(sync_ind); 
            %get sampling times in s
            t = (edf.Samples.time-sync_time)/1000;
            %get sampling rate
            SamplingRate = 1/(t(2)-t(1));
    
            %% get info about trial segments
    
            summary_FileName = dir(strcat(edfFiles.folder,'/*_out.csv'));
            % Detect the import options of the file
            opts = detectImportOptions(strcat(edfFiles.folder,'/',summary_FileName.name));
            % Specify the variable names to import
            opts.SelectedVariableNames = {'instruct_onset', 'instruct_duration', ... 
                'postinstruct_onset', 'postinstruct_duration', ... 
                'trigger_onset', 'trigger_duration', ... 
                'postTrigger_onset', 'postTrigger_duration', ... 
                'stimMove_onset', 'stimMove_duration', ... 
                'fixation_onset', 'fixation_duration', ... 
                'stim_id', 'type'};
            % Read the table
            segmentsTable = readtable(strcat(edfFiles.folder,'/',summary_FileName.name), opts);
            segmentsTable.trialNo = (1:height(segmentsTable))';
            
       
            unit = 'px';
            pupil     = struct('t',t,'pupilValues',pupilValues);
            pupilStruc = struct('data',pupil,...
                'diameterUnit',unit,...
                'segmentsTable',segmentsTable, ...
                'SamplingRate',SamplingRate);
            %% calculate the blinking time
            BlinkDuration = sum(edf.Events.Eblink.duration)/1000;
            TotalDuration = t(end)-t(1);
            BlinkRatio = BlinkDuration*100/TotalDuration;
            
            fprintf('Blink duration of participant %s: %.3f%%\n', ...
                subjectNumber{1}, BlinkRatio);
    
            %% filtering
            standardRawSettings = rawDataFilter();
    
            [valOut,~,~] ...
            = filterRawData(pupilStruc.data.t,pupilStruc.data.pupilValues,...
            standardRawSettings);
            
            dataFilt = pupilStruc;
            dataFilt.data.t = dataFilt.data.t(valOut);
            dataFilt.data.pupilValues = dataFilt.data.pupilValues(valOut);
            pupilStruc.dataFilt = dataFilt;
            lostData = (sum(~isnan(pupilStruc.data.pupilValues))-sum(valOut))/sum(~isnan(pupilStruc.data.pupilValues));
    
            fprintf('Data loss from filtering: %.2f%%\n',lostData*100)
            dataSmoothInterp = UpsamplingAndSmooth(dataFilt);
    
            %% cubic spline interpolation for missing data
            validIndices = ~isnan(dataSmoothInterp.data.pupilValues);
            dataSmoothInterp.data.pupilValues = interp1(...
                dataSmoothInterp.data.t(validIndices), ...
                dataSmoothInterp.data.pupilValues(validIndices),...
                dataSmoothInterp.data.t, 'spline');
    
            pupilStruc.dataSmoothInterp = dataSmoothInterp;
            
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
end

diary off









               
%% helper functions
            
function s = UpsamplingAndSmooth(s)
            % Gets Interpolates and smoothes the samples.
            %
            %--------------------------------------------------------------
            % The upsampling frequency [Hz] used to generate the smooth
            % signal:
            interp_upsamplingFreq     = 1000;
            
            % Calculate the low pass filter specs using the cutoff
            % frequency [Hz], filter order, and the upsample frequency
            % specified above:
            LpFilt_cutoffFreq         = 4;
            LpFilt_order              = 4;
            [LpFilt_B,LpFilt_A] ...
                = butter(LpFilt_order,2*LpFilt_cutoffFreq...
                /interp_upsamplingFreq);
            
            % Maximum gap [s] in the used raw samples to interpolate over
            % (section that were interpolated over larger distances will be
            % set to missing; i.e. NaN):
            interp_maxGap_s             = 0.025;

            % Interpolate the valid samples to form the signal:
            validAreas            = s.data.pupilValues;
            valid_t_s             = s.data.t;
            
            % Generate the upsampled time vector (seconds):
             if s.SamplingRate ~= 1000
                t_upsampled = ...
                    (valid_t_s(1):(1/interp_upsamplingFreq):valid_t_s(end))';
                diaInterp = ...
                    interp1(valid_t_s,validAreas,t_upsampled,'linear');
             else
                t_upsampled = valid_t_s;
                diaInterp = validAreas;
             end
            
            % Filter:
            diaInterp = filtfilt(LpFilt_B, LpFilt_A, diaInterp);
            
            % Calculate the gaps (the edges dont really matter, samples
            % close to the edges wil be given a gap of 0 subsequently).

            gaps_s = discretize(t_upsampled,valid_t_s,diff(valid_t_s));

            % Set samples that are really close to the raw samples as
            % having no gap (scale the closeness with the sampling freq.).
            % In this case it is set to half the sampling interval:
            notTouchingYouTolerance_s = 0.5/interp_upsamplingFreq;
            
            almostTouching = ismembertol(t_upsampled,valid_t_s...
             ,notTouchingYouTolerance_s,'DataScale',1);

            
            % Now actually set the upsampled timestamps that are really
            % close to the timestamps of the valid measured datapoints to
            % zero:
            gaps_s(logical(almostTouching)) = 0;
            
            % Remove gaps:
            diaInterp(gaps_s > interp_maxGap_s) = NaN;
            
            % Save data to model:
            s.data.t             = t_upsampled;
            s.data.pupilValues = diaInterp;
            
end



