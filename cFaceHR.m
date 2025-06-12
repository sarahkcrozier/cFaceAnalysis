function cFaceHR(participantID)

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
    error('No matching folder found for participant ID %s', IDstring);
end
if sum(matchIdx) > 1
    warning('Multiple folders matched for ID %s. Using the first.', IDstring);
end

folderName = folderNames{find(matchIdx, 1)};
folderPath = fullfile(options.paths.data, folderName);
disp(['Using folder: ', folderName]);
 
%% ---- STEP 1: Locate instruct_onset ----
filePattern = fullfile(folderPath, 'beh', 'cface*MHE*', '*out.csv');
summaryFile = dir(filePattern);


if isempty(summaryFile)
    error('No matching file found for participant %s in folder %s', IDstring, folderPath);
end

summaryFilePath = fullfile(summaryFile.folder, summaryFile.name);

% Read instruct_onset from CSV
opts = detectImportOptions(summaryFilePath);
opts.SelectedVariableNames = 'instruct_onset';
segmentsTable = readtable(summaryFilePath, opts);

baselineTiming = segmentsTable.instruct_onset(2);  % return first onset NOTE THIS IS RETURNING 0 for 003. Check are the same

%% ---- STEP 2: Locate PPG file and compute mean PPG up to baseline ----
% note I know this is not what we're supposed to do with PPG but it is an
% interim step

ppgFilePattern = fullfile(folderPath, 'beh', 'cface*MHE*', '*ppg.csv');
ppgFile = dir(ppgFilePattern);
if isempty(ppgFile)
    warning('No PPG file found for participant %s. Skipping PPG mean calculation.', IDstring);
    meanPPG = NaN;
else
    ppgFilePath = fullfile(ppgFile.folder, ppgFile.name);
    
    % Read only 'time' and 'PPG' columns
    ppgOpts = detectImportOptions(ppgFilePath);
    ppgOpts.SelectedVariableNames = {'time', 'PPG'};
    ppgTable = readtable(ppgFilePath, ppgOpts);
    
    % Find all rows where time <= baselineTiming
    validRows = ppgTable.time <= baselineTiming;
    
    if any(validRows)
        meanPPG = mean(ppgTable.PPG(validRows), 'omitnan');
    else
        warning('No PPG data before baseline timing for participant %s.', IDstring);
        meanPPG = NaN;
    end
end

% Report result
fprintf('Baseline timing for %s: %.3f seconds\n', IDstring, baselineTiming);
fprintf('Mean PPG before baseline: %.3f\n', meanPPG);


%% ---- STEP 3: Compute mean PPG for the five seconds following stimMove_onset on all incongruent trials ----
segmentsTableIncongruent = cFaceIncongruentTrials(participantID);

% Calculate mean PPG in 5-second windows after each incongruent trial onset
onsets = segmentsTableIncongruent.stimMove_onset;
ppgMeans = NaN(height(segmentsTableIncongruent), 1);

for i = 1:length(onsets)
    onsetTime = onsets(i);
    timeWindowIdx = (ppgTable.time > onsetTime) & (ppgTable.time <= onsetTime + 5);
    
    if any(timeWindowIdx)
        ppgMeans(i) = mean(ppgTable.PPG(timeWindowIdx), 'omitnan');
    else
        warning('No PPG data found for 5s window after incongruent trial %d (onset = %.3f)', i, onsetTime);
    end
end

% Final result: mean PPG across all incongruent windows
meanPPG_IncongruentWindow = mean(ppgMeans, 'omitnan');

fprintf('Mean PPG in 5s post-incongruent windows: %.3f\n', meanPPG_IncongruentWindow);

diary off
end










