function segmentsTableIncongruent = cFaceIncongruentTrials(participantID,diaryName)

%% extract cFace incongruent trial timings
%% see segmentsTableIncongruent
%% needs to be merged with cFacePupilData.m

options = specifyOptions;

%% save text of Matlab session
diary(fullfile(options.paths.workingDir,diaryName))

% Format participant ID
IDstring = sprintf('%03d', participantID);

% Find matching folder
participantFolders = dir(options.paths.rawData);
folderNames = {participantFolders.name};
folderNames = folderNames(~ismember(folderNames, {'.','..'}));  % skip system entries

matchIdx = contains(folderNames, IDstring);
if ~any(matchIdx)
    error('No matching folder found for participant ID %s', IDstring);
end
folderName = folderNames{find(matchIdx, 1)};
folderPath = fullfile(options.paths.rawData, folderName);
    
% Locate summary file
summaryFile = dir([folderPath,filesep,'beh',filesep,'cface*MH*out.csv']);
if isempty(summaryFile)
    error('No summary file found for participant %s.', IDstring);
end
summaryFilePath = fullfile(summaryFile.folder,summaryFile.name);

% Import relevant columns
opts = detectImportOptions(summaryFilePath);
opts.SelectedVariableNames = {'ParticipantID', 'stimMove_onset', 'fixation_onset', 'fixation_duration', ...
    'cong', 'ptemot'};
segmentsTable = readtable(summaryFilePath, opts);
segmentsTable.trialNo = (1:height(segmentsTable))';

% Add responseWindow
segmentsTable.responseWindow = ...
    (segmentsTable.fixation_onset + segmentsTable.fixation_duration) - segmentsTable.stimMove_onset;

segmentsTableIncongruent = segmentsTable(segmentsTable.cong==0, ...
    {'trialNo','stimMove_onset','fixation_onset','fixation_duration','responseWindow','cong','ptemot'});

diary off










