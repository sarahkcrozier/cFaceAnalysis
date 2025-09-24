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
    
% Locate summary file
summaryFile = dir(fullfile(folderPath, 'beh', 'cface*MH*', '*out.csv'));
if isempty(summaryFile)
    error('No summary file found for participant %s.', IDstring);
end
summaryFilePath = fullfile(summaryFile.folder, summaryFile.name);

% Import relevant columns
opts = detectImportOptions(summaryFilePath);
opts.SelectedVariableNames = {'ParticipantID', 'stimMove_onset', 'fixation_onset', ...
    'cong', 'ptemot', 'trigger_onset'};
segmentsTable = readtable(summaryFilePath, opts);
segmentsTable.trialNo = (1:height(segmentsTable))';

% Add timeToNextTrigger
segmentsTable.timeToNextTrigger = NaN(height(segmentsTable), 1);
for i = 1:height(segmentsTable)-1
    segmentsTable.timeToNextTrigger(i) = ...
        segmentsTable.trigger_onset(i+1) - segmentsTable.stimMove_onset(i);
end
segmentsTable.timeToNextTrigger(end) = 4;

% Return only incongruent trials
segmentsTableIncongruent = segmentsTable(segmentsTable.cong == 0, :);

diary off










