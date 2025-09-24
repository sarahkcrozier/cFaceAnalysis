% function cFacePupilArea(participantID)
% 
% options = specifyOptions;
% ascDir  = options.paths.EDFtoASC;
% pupilDir  = options.paths.eyeData;
% 
% segmentsTable = cFaceTrialSegments(participantID);
% instruct_onset = segmentsTable.instruct_onset(1);
% 
% pupilData     = struct('participantID',participantID,'eye',eye,'syncpulse',NaN,'baseline_s',(instruct_onset - 10),'baseline_e',instruct_onset);
% 
% Now can you help write the following:
% 
% 
%  for all participants in ascDir (where asc files are labelled by the participant numbers, eg participant 58 is 058.asc
% identify participantID and save in the predefined struct pupilData.participantID
% scan through the asc file and: 
% identify pupilData.eye by searching for "SAMPLES","GAZE", then "RIGHT" (save stuct variable as L for left and R for right)
%  identify pupilData.syncpulse (the number directly preceding "SyncPulseReceived"). If two or more instances of syncpulse appear in the file, take the last one.
% remove header (everything before data columns after "RECORD CR 1000 2 1 ",pupilData.eye (ie, L or R)
% create a table with the column headers eyeLinkTime, X, Y, pupilArea, convertedTime_s (calculated as (eyeLinkTime - SyncPulseReceived) / 1000), trial, participantID, message
% The fist 4 columns eyeLinkTime, X, Y, pupilArea, are to be taken directly from the ASC file columns tha start directly after "RECORD CR 1000 2 1 ",pupilData.eye (ie, L or R)
% Trial number can be identified by the note "MSG" followed by the trial eyeLink time unit start time, then the string "TRIALID" and then the trial number (from 0 to 79).
% These messages should not be included in the table, but they should be used to identify the start time for each trial, and the trial number can be listed in the table for all consecutive times until the start
% of the following trial. Trial variable data prior to the start of trials can be listed as NaN (this will be used for baseline data).
% Other messages including ESACC, SFIX, SBLINK should be recorded in the message column, they do not go in the table. Each message has a slightly different recording method: 
% SBLINK pupil area (ie, when someone blinks) will be 0.0. SBLINK message will be printed at the start and end of the blink. Message should be recorded as SBLINK for all the blink rows inbetween those two messages.
% SFIX is a fixation: where the x variable is identical for more than one row. Record SFIX in the message for these rows. 
% The start of a saccade is listed as, for example,: "SSACC L  2443833", where SSACC means start of saccade, followed by the eye that is being tracked (here L), and the eyelink time (here 2443833)
%     The end of a saccade looks like this, as an example: 
% ESACC L  2443833	2444021	189	  998.1	  501.9	  967.9	  496.4	   0.42	    783
% DO not record the saccade information in message, just strip it out. 
% The end of the data oin the asc file is identified by the message "MSG	[eyelink time] End Study" where [eyelink time] is the final time stamp. 
% do not record any data from this message onwards. 
% 
% save the table as a csv file in pupilDir, something like: 
% csvpath = fullfile(options.paths.eyeData,strcat(num2str(participantID),...
%     '_cface_eyeData.csv'));  


function cFacePupilArea(participantID)
% cFacePupilArea
% Parse EyeLink ASC files to produce per-sample pupil area tables, aligned to
% sync-pulse time, with trial labels and message annotations, and save as CSV.
%
% - Scans *.asc files in options.paths.EDFtoASC (ascDir).
% - Extracts participantID from filename (e.g., 058.asc -> 58).
% - Determines tracked eye (L/R) from SAMPLES/GAZE block (RIGHT=>R, LEFT=>L).
% - Finds last "SyncPulseReceived" time and converts EyeLink time to seconds.
% - Removes header up to "RECORD CR 1000 2 1 <Eye>".
% - Parses sample rows into: eyeLinkTime, X, Y, pupilArea.
% - Assigns 'trial' using messages "MSG <time> TRIALID <n>" (0..79).
% - Before first trial, trial = NaN (baseline).
% - Annotates message column:
%       * SBLINK: tag all rows between SBLINK start and end (pupil area zero).
%       * SFIX: tag rows where X is unchanged for >= 2 consecutive samples.
%   (SSACC/ESACC lines are stripped and not recorded.)
% - Stops parsing at "MSG <time> End Study".
% - Saves table as "<NNN>_cface_eyeData.csv" in options.paths.eyeData.
%
% Input:
%   participantID (numeric or char) to process only one participant (matching file).
%
% Dependencies:
%   - specifyOptions (returns paths)
%   - cFaceTrialSegments(participantID) (to compute baseline window)
%
% Notes:
%   - If segments table is unavailable for a participant, baseline window is left NaN.
%   - The parser assumes standard EyeLink ASCII layout with SAMPLES block.

% -----------------------------
% Setup: resolve paths and file list
% -----------------------------

options  = specifyOptions;
ascDir   = options.paths.EDFtoASC;
pupilDir = options.paths.eyeData;

% -----------------------------
% Resolve filename from participantID
% -----------------------------
if isnumeric(participantID)
    participantID_str = sprintf('%03d', participantID);  % zero-padded (e.g., 58 -> '058')
else
    participantID_str = participantID;
end

ascFile = dir(fullfile(ascDir, [participantID_str, '.asc']));
assert(~isempty(ascFile), 'No ASC file found for participant %s.', participantID_str);

ascPath = fullfile(ascFile.folder, ascFile.name);
fprintf('Processing ASC: %s\n', ascPath);

% -----------------------------
% Read raw ASC text
% -----------------------------
rawText = fileread(ascPath);

% -----------------------------
% Determine tracked eye (R/L)
% -----------------------------
if ~isempty(regexpi(rawText, '\bSAMPLES\s+GAZE\s+RIGHT\b'))
    trackedEye = 'R';
elseif ~isempty(regexpi(rawText, '\bSAMPLES\s+GAZE\s+LEFT\b'))
    trackedEye = 'L';
else
    recEye = regexp(rawText, 'RECORD\s+CR\s+1000\s+2\s+1\s+([LR])', 'tokens', 'once');
    trackedEye = ~isempty(recEye) * upper(recEye{1});
    if isempty(trackedEye), trackedEye = 'R'; end
end

% -----------------------------
% Find last SyncPulseReceived
% -----------------------------
syncMatches = regexp(rawText, 'MSG\s+(\d+)\s+SyncPulseReceived', 'tokens');
if isempty(syncMatches)
    syncPulse = 0;
else
    syncPulse = str2double(syncMatches{end}{1});
end

% -----------------------------
% Trim header after RECORD line
% -----------------------------
recPattern = sprintf('RECORD\\s+CR\\s+1000\\s+2\\s+1\\s+%s', trackedEye);
recIdx = regexp(rawText, recPattern, 'start', 'once');
assert(~isempty(recIdx), 'RECORD line not found.');
dataText = rawText(recIdx:end);

% Cut at "End Study" if present
endStudyIdx = regexp(dataText, 'MSG\s+\d+\s+End\s+Study', 'start', 'once');
if ~isempty(endStudyIdx)
    dataText = dataText(1:endStudyIdx-1);
end

% -----------------------------
% Extract TRIAL start times
% -----------------------------
trialMsgs = regexp(dataText, 'MSG\s+(\d+)\s+TRIALID\s+(\d+)', 'tokens');
trialStartTimes = cellfun(@(t) str2double(t{1}), trialMsgs);
trialNumbers    = cellfun(@(t) str2double(t{2}), trialMsgs);

% -----------------------------
% Remove saccade message lines
% -----------------------------
dataText = regexprep(dataText, 'SSACC[^\n\r]*[\r\n]+', '');
dataText = regexprep(dataText, 'ESACC[^\n\r]*[\r\n]+', '');

% -----------------------------
% Parse numeric sample rows
% -----------------------------
lines = regexp(dataText, '\r?\n', 'split');
sampleRows = {};
for i = 1:numel(lines)
    L = strtrim(lines{i});
    if isempty(L) || strncmp(L, 'MSG', 3), continue; end
    if ~isstrprop(L(1), 'digit'), continue; end
    nums = regexp(L, '([+-]?\d+(?:\.\d+)?)', 'match');
    if numel(nums) < 4, continue; end
    sampleRows(end+1,:) = nums(1:4); %#ok<AGROW>
end

timeVec  = str2double(sampleRows(:,1));
xVec     = str2double(sampleRows(:,2));
yVec     = str2double(sampleRows(:,3));
pupilVec = str2double(sampleRows(:,4));
nSamples = numel(timeVec);

% -----------------------------
% Assign trial numbers
% -----------------------------
trialVec = NaN(nSamples,1);
if ~isempty(trialStartTimes)
    edges = [-Inf; trialStartTimes(:); Inf];
    binIdx = discretize(timeVec, edges);
    for b = 2:(numel(edges)-1)
        trialVec(binIdx==b) = trialNumbers(b-1);
    end
end

% -----------------------------
% Messages (SBLINK, SFIX)
% -----------------------------
msgVec = strings(nSamples,1);
% Tag blinks by pupil=0
msgVec(pupilVec==0) = "SBLINK";
% Tag fixations by repeated X
sameX = [false; diff(xVec)==0];
sfixMask = sameX | [sameX(2:end); false];
msgVec(sfixMask & msgVec=="") = "SFIX";

% -----------------------------
% Build output table
% -----------------------------
convertedTime_s = (timeVec - syncPulse) / 1000;
eyeCol = repmat(string(trackedEye), nSamples, 1);
participantCol = repmat(str2double(participantID_str), nSamples, 1);

eyeDataTbl = table(timeVec, xVec, yVec, pupilVec, convertedTime_s, trialVec, ...
    participantCol, msgVec, eyeCol, ...
    'VariableNames', {'eyeLinkTime','X','Y','pupilArea','convertedTime_s','trial','participantID','message','eye'});

% -----------------------------
% Save CSV
% -----------------------------
csvFileName = sprintf('%s_cface_eyeData.csv', participantID_str);
csvPath = fullfile(pupilDir, csvFileName);
writetable(eyeDataTbl, csvPath);
fprintf('Saved: %s (%d samples)\n', csvPath, nSamples);

end