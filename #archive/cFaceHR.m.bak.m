function HR = cFaceHR(participantID)

%% extract cFace baseline timings
% Extracts the first instruct_onset timing from cFace baseline for a given participant

options = specifyOptions;
IDstring = sprintf('%03d', participantID);


%% save text of Matlab session
diary(fullfile(options.paths.analysis,'output_HR_cface.txt'))

%% Locate participant folder
participantFolders = dir(options.paths.data);
folderNames = {participantFolders.name};
folderNames = folderNames(~ismember(folderNames, {'.','..'}));  % skip system entries

% Find participant folder name and path

matchIdx = contains(folderNames, IDstring);  % logical index

if ~any(matchIdx)
    HR.baseline = NaN;  
    HR.incongruentAverage = NaN;  
    % error('No matching folder found for participant ID %s', IDstring);
    return
end

if sum(matchIdx) > 1
    warning('Multiple folders matched for ID %s. Using the first.', IDstring);
end

folderName = folderNames{find(matchIdx, 1)};
folderPath = fullfile(options.paths.data, folderName);
disp(['Using folder: ', folderName]);
 
%% ---- STEP 1: Locate instruct_onset ----
filePattern = fullfile(folderPath, 'beh', 'cface*MH*', '*out.csv');
summaryFile = dir(filePattern);


if isempty(summaryFile)
    HR.baseline = NaN;  
    HR.incongruentAverage = NaN;  
    % error('No matching file found for participant %s in folder %s', IDstring, folderPath);
    return
end

summaryFilePath = fullfile(summaryFile.folder, summaryFile.name);

% Read instruct_onset from CSV
opts = detectImportOptions(summaryFilePath);
opts.SelectedVariableNames = 'instruct_onset';
segmentsTable = readtable(summaryFilePath, opts);

baselineTiming = segmentsTable.instruct_onset(1);  % return first onset NOTE THIS IS RETURNING 0 for 003. Check are the same
baselineTiming = max(baselineTiming, 10); % if the baselineTiming is 0, then use 10secs as baseline timing

%% ---- STEP 2: Locate PPG file, smooth ----


ppgFilePattern = fullfile(folderPath, 'beh', 'cface*MH*', '*ppg.csv');
ppgFile = dir(ppgFilePattern);
if isempty(ppgFile)
    warning('No PPG file found for participant %s. Skipping PPG  calculation.', IDstring);
    meanPPG = NaN;
else
    ppgFilePath = fullfile(ppgFile.folder, ppgFile.name);
    
    % Read only 'time' and 'PPG' columns
    ppgOpts = detectImportOptions(ppgFilePath);
    ppgOpts.SelectedVariableNames = {'time', 'PPG'};
    ppgTable = readtable(ppgFilePath, ppgOpts);

    % smooth data using Savitzy-Golay Filter
    ppgData = smoothdata(ppgTable.PPG,'sgolay');
    
    % Find all rows where time <= baselineTiming
    validRows = ppgTable.time <= baselineTiming;
    
    if any(validRows)
        % extract only the ppg from the given window
        ppgWindow = ppgData;
        ppgWindow(~validRows,:) = [];
        % get peaks , i.e. maximum value within 60 points on x-axis
        [peaks,locations,~,~] = findpeaks(ppgWindow,'MinPeakDistance',56);
        
        fig = figure;  % create a new figure window
        findpeaks(ppgWindow, 'MinPeakDistance', 56);  % plot with peaks
        % Define the filename and path
        filename = sprintf('ppgBaselinePeaksPlot_%s.fig', IDstring);
        savePath = fullfile(options.paths.plots, filename);
        saveas(fig, savePath);
        close all;

        peakCount = numel(peaks);
        HR.baseline = (peakCount/baselineTiming)*60;
        

        %% getting mean of all valid ppg rows (orgignial hack mean)
        % meanPPG = mean(ppgTable.PPG(validRows), 'omitnan');



    else
        warning('No PPG data before baseline timing for participant %s.', IDstring);
        % meanPPG = NaN;
    end

%% ---- STEP 3: Compute mean PPG for the window following stimMove_onset on all incongruent trials ----
segmentsTableIncongruent = cFaceIncongruentTrials(participantID);

% Calculate mean PPG in windows after each incongruent trial onset
onsets = segmentsTableIncongruent.stimMove_onset;
windows = segmentsTableIncongruent.timeToNextTrigger;
trialNo = segmentsTableIncongruent.trialNo;
minWindow = 4;
% allIncongruentHR = NaN(height(segmentsTableIncongruent), 1); % not sure
% what this was needed for

for i = 1:length(onsets)
    onsetTime = onsets(i);
    %if window is < 4secs, then use 4 secs
    window = max(windows(i), minWindow);
    timeWindowIdx = (ppgTable.time > onsetTime) & (ppgTable.time <= onsetTime + window);
    if any(timeWindowIdx)
        %ppgMeans(i) = mean(ppgData(timeWindowIdx), 'omitnan');
        [peaks,locations,~,~] = findpeaks(ppgData(timeWindowIdx),'MinPeakDistance',56);
        
        %% save plot of each response window
       % fig = figure;  % create a new figure window
       % findpeaks(ppgData(timeWindowIdx),'MinPeakDistance',56);  % plot with peaks
       % % Define the filename and path
       % filename = sprintf('ppgIncongruentPeaksPlot_%s__%d.fig', IDstring, trialNo(i));
       % savePath = fullfile(options.paths.plots, filename);
       % saveas(fig, savePath);
       % close all;

        ppgMeans(i) = (numel(peaks)/window)*60;
        
        %% print the HRs for each incongruent window
        % rate = (numel(peaks)/window)*60;
        % outputStr = sprintf('Trial %d: Rate = %.2f', trialNo(i), rate);
        % disp(outputStr);

    else
        warning('No PPG data found for ', window, 'second window after incongruent trial %d (onset = %.3f)', i, onsetTime);
    end
end

% Final result: mean PPG across all incongruent windows
HR.incongruentAverage = mean(ppgMeans, 'omitnan');


end

% Report result
fprintf('Baseline timing for %s: %.3f seconds\n', IDstring, baselineTiming);
fprintf('Mean PPG before baseline: %.3f\n', HR.baseline);
fprintf('Mean PPG in post-incongruent windows: %.3f\n', HR.incongruentAverage);

diary off

end










