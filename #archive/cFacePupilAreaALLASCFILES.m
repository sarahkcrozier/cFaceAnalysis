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
% - Scans all *.asc files in options.paths.EDFtoASC (ascDir).
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
% Optional input:
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
options = specifyOptions;
ascDir  = options.paths.EDFtoASC;
pupilDir = options.paths.eyeData;

% Find .asc files (optionally filter by participantID argument)
allAsc = dir(fullfile(ascDir, '*.asc'));
if nargin >= 1 && ~isempty(participantID)
    if isnumeric(participantID), participantID = sprintf('%03d', participantID); end
    mask = contains({allAsc.name}, participantID, 'IgnoreCase', true);
    ascFiles = allAsc(mask);
else
    ascFiles = allAsc;
end
assert(~isempty(ascFiles), 'No .asc files found in %s.', ascDir);

% -----------------------------
% Process each ASC file
% -----------------------------
for f = 1:numel(ascFiles)
    ascPath = fullfile(ascFiles(f).folder, ascFiles(f).name);
    fprintf('Processing ASC: %s\n', ascPath);

    % -----------------------------
    % Derive participantID from the filename (e.g., "058.asc" -> 58)
    % -----------------------------
    tokens = regexp(ascFiles(f).name, '(?<id>\d+)\.asc$', 'names', 'once');
    if isempty(tokens)
        warning('Skipping file with nonstandard name: %s', ascFiles(f).name);
        continue;
    end
    participantID_str = tokens.id;                 % zero-padded string
    participantID_num = str2double(participantID_str);

    % -----------------------------
    % Load entire file text for regex extraction and line parsing
    % -----------------------------
    rawText = fileread(ascPath);

    % -----------------------------
    % Detect tracked eye from SAMPLES/GAZE block
    %   Look for "... SAMPLES GAZE RIGHT ..." or "... LEFT ..."
    %   Save as 'R' or 'L' in pupilData.eye
    % -----------------------------
    if ~isempty(regexpi(rawText, '\bSAMPLES\s+GAZE\s+RIGHT\b'))
        trackedEye = 'R';
    elseif ~isempty(regexpi(rawText, '\bSAMPLES\s+GAZE\s+LEFT\b'))
        trackedEye = 'L';
    else
        % Fallback: infer from RECORD line suffix if present
        recEye = regexp(rawText, 'RECORD\s+CR\s+1000\s+2\s+1\s+([LR])', 'tokens', 'once');
        if ~isempty(recEye)
            trackedEye = upper(recEye{1});
        else
            warning('Could not determine tracked eye for %s. Assuming RIGHT.', ascPath);
            trackedEye = 'R';
        end
    end

    % -----------------------------
    % Find the last SyncPulseReceived time (preceding number)
    %   Usually appears like: "MSG <time> SyncPulseReceived"
    % -----------------------------
    syncMatches = regexp(rawText, 'MSG\s+(\d+)\s+SyncPulseReceived', 'tokens');
    if isempty(syncMatches)
        warning('No SyncPulseReceived found in %s. Using 0.', ascPath);
        syncPulse = 0;
    else
        lastTok   = syncMatches{end};
        syncPulse = str2double(lastTok{1});
    end

    % -----------------------------
    % Trim header up to the RECORD line for the tracked eye
    %   We start parsing data AFTER this marker.
    % -----------------------------
    recPattern = sprintf('RECORD\\s+CR\\s+1000\\s+2\\s+1\\s+%s', trackedEye);
    recIdx = regexp(rawText, recPattern, 'start', 'once');
    if isempty(recIdx)
        % Fall back to the first occurrence of "RECORD CR 1000 2 1"
        recIdx = regexp(rawText, 'RECORD\s+CR\s+1000\s+2\s+1', 'start', 'once');
    end
    assert(~isempty(recIdx), 'RECORD line not found in %s.', ascPath);

    dataText = rawText(recIdx:end);

    % -----------------------------
    % Stop at "End Study" message if present
    % -----------------------------
    endStudyIdx = regexp(dataText, 'MSG\s+\d+\s+End\s+Study', 'start', 'once');
    if ~isempty(endStudyIdx)
        dataText = dataText(1:endStudyIdx-1);
    end

    % -----------------------------
    % Extract TRIAL start times and trial numbers:
    %   "MSG <time> TRIALID <n>"
    % -----------------------------
    trialMsgs = regexp(dataText, 'MSG\s+(\d+)\s+TRIALID\s+(\d+)', 'tokens');
    trialStartTimes = [];
    trialNumbers    = [];
    if ~isempty(trialMsgs)
        trialStartTimes = cellfun(@(t) str2double(t{1}), trialMsgs);
        trialNumbers    = cellfun(@(t) str2double(t{2}), trialMsgs);
        % Ensure sorted by time
        [trialStartTimes, sortIdx] = sort(trialStartTimes);
        trialNumbers = trialNumbers(sortIdx);
    end

    % -----------------------------
    % Extract SBLINK start/end pairs (blink spans)
    %   Patterns usually appear as:
    %     "SBLINK <Eye> <startTime>" and later "EBLINK <Eye> <startTime> <endTime> ..."
    %   For robustness, also tag rows with pupilArea == 0 as SBLINK.
    % -----------------------------
    % Start of blink (SBLINK) line: "SBLINK <Eye> <start>"
    sblinkStarts = regexp(dataText, sprintf('SBLINK\\s+%s\\s+(\\d+)', trackedEye), 'tokens');
    % End of blink (EBLINK) line: "EBLINK <Eye> <start> <end> ..."
    eblinkPairs  = regexp(dataText, sprintf('EBLINK\\s+%s\\s+(\\d+)\\s+(\\d+)', trackedEye), 'tokens');

    blinkWindows = []; % Nx2 of [start end]
    if ~isempty(eblinkPairs)
        for k = 1:numel(eblinkPairs)
            startT = str2double(eblinkPairs{k}{1});
            endT   = str2double(eblinkPairs{k}{2});
            blinkWindows = [blinkWindows; startT, endT]; %#ok<AGROW>
        end
    elseif ~isempty(sblinkStarts)
        % If only SBLINK starts are found (rare), keep starts; ends will be inferred via pupil==0 later
        starts = cellfun(@(t) str2double(t{1}), sblinkStarts);
        blinkWindows = [starts(:), starts(:)]; % provisional; will expand via pupil==0 heuristic
    end

    % -----------------------------
    % Remove saccade message lines entirely (SSACC / ESACC) so they aren't parsed as samples
    % -----------------------------
    dataText = regexprep(dataText, 'SSACC[^\n\r]*[\r\n]+', '');
    dataText = regexprep(dataText, 'ESACC[^\n\r]*[\r\n]+', '');

    % -----------------------------
    % Identify the start of numeric sample lines:
    % Many ASC exports list samples as lines starting with a timestamp (integer),
    % followed by X, Y, pupil (floats). We will read all lines that start with digits.
    % -----------------------------
    % Split lines for selective parsing
    lines = regexp(dataText, '\r?\n', 'split');

    % Preallocate storage (grow dynamically if needed)
    timeVec   = [];
    xVec      = [];
    yVec      = [];
    pupilVec  = [];
    msgVec    = strings(0,1); % message annotations per sample row

    % Helper to parse a numeric sample line: "t x y pupil ..." (we take first 4 fields)
    isSample = false(numel(lines),1);
    sampleRows = cell(0,4);

    for i = 1:numel(lines)
        L = strtrim(lines{i});
        if isempty(L)
            continue;
        end

        % Skip pure MSG lines (we will use them to populate trials separately)
        if strncmp(L, 'MSG', 3)
            continue;
        end
        % Skip other non-numeric headers
        if ~isstrprop(L(1), 'digit')
            continue;
        end

        % Attempt to read first four numeric tokens
        nums = regexp(L, '([+-]?\d+(?:\.\d+)?)', 'match');
        if numel(nums) < 4
            continue;
        end

        isSample(i) = true;
        sampleRows(end+1, :) = nums(1:4); %#ok<AGROW>
    end

    if isempty(sampleRows)
        warning('No sample rows parsed from %s.', ascPath);
        continue;
    end

    % Convert sample rows to numeric vectors
    timeVec  = str2double(sampleRows(:,1));          % EyeLink time (ms)
    xVec     = str2double(sampleRows(:,2));          % X position
    yVec     = str2double(sampleRows(:,3));          % Y position
    pupilVec = str2double(sampleRows(:,4));          % Pupil area

    nSamples = numel(timeVec);
    msgVec   = strings(nSamples,1);                  % annotation holder

    % -----------------------------
    % Assign trial numbers by time:
    % Each "MSG <t> TRIALID <n>" marks the start of trial n; apply until next start.
    % Prior to first trial -> NaN
    % -----------------------------
    trialVec = NaN(nSamples,1);
    if ~isempty(trialStartTimes)
        % For each sample time, find the last trial start <= sample time
        % Use discretize with bin edges at trialStartTimes
        edges = [-Inf; trialStartTimes(:); Inf];
        binIdx = discretize(timeVec, edges);
        % binIdx == 1 -> pre first trial (NaN); 2..numTrials+1 map to trials 1..numTrials
        for b = 2:(numel(edges)-1)
            mask = (binIdx == b);
            trialVec(mask) = trialNumbers(b-1);
        end
    end

    % -----------------------------
    % Tag SBLINK across rows that fall within blink windows
    % Also (robustness) tag rows with pupil==0 as SBLINK
    % -----------------------------
    if ~isempty(blinkWindows)
        for k = 1:size(blinkWindows,1)
            t0 = blinkWindows(k,1);
            t1 = blinkWindows(k,2);
            inBlink = (timeVec >= t0) & (timeVec <= t1);
            msgVec(inBlink) = "SBLINK";
        end
    end
    zeroPupil = (pupilVec == 0);
    msgVec(zeroPupil) = "SBLINK";

    % -----------------------------
    % Tag SFIX:
    % Define as runs where X remains exactly the same for >=2 consecutive samples.
    % (Conservative; you can relax with a small tolerance if desired.)
    % -----------------------------
    sameX = [false; diff(xVec)==0];
    % Extend tagging to cover both members of each equal pair/run
    sfixMask = sameX | [sameX(2:end); false];
    msgVec(sfixMask & msgVec=="") = "SFIX";

    % -----------------------------
    % Build output table
    % -----------------------------
    convertedTime_s = (timeVec - syncPulse) / 1000;  % align to sync pulse, seconds
    participantCol  = repmat(participantID_num, nSamples, 1);
    eyeCol          = repmat(string(trackedEye), nSamples, 1);

    eyeDataTbl = table( ...
        timeVec, xVec, yVec, pupilVec, convertedTime_s, trialVec, participantCol, msgVec, eyeCol, ...
        'VariableNames', {'eyeLinkTime','X','Y','pupilArea','convertedTime_s','trial','participantID','message','eye'} ...
    );

    % -----------------------------
    % Add baseline window from segments table (if available)
    % -----------------------------
    baseline_s = NaN; baseline_e = NaN;
    try
        segmentsTable = cFaceTrialSegments(participantID_num);
        instruct_onset = segmentsTable.instruct_onset(1);
        baseline_s = instruct_onset - 10;
        baseline_e = instruct_onset;
    catch
        % Segments not available; leave baseline NaN
    end
    % Store baseline in a small struct for provenance (optional)
    pupilData = struct( ...
        'participantID', participantID_num, ...
        'eye', trackedEye, ...
        'syncpulse', syncPulse, ...
        'baseline_s', baseline_s, ...
        'baseline_e', baseline_e ...
    ); %#ok<NASGU> % (kept for debugging; not saved separately here)

    % -----------------------------
    % Save CSV
    % -----------------------------
    csvFileName = sprintf('%s_cface_eyeData.csv', participantID_str);
    csvPath = fullfile(pupilDir, csvFileName);
    writetable(eyeDataTbl, csvPath);
    fprintf('Saved: %s  (%d samples)\n', csvPath, nSamples);
end

end