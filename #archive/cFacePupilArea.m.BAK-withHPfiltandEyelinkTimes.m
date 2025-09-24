function cFacePupilArea(participantID)
% cFacePupilArea
% Parse EyeLink ASC file for the specified participant and save a CSV with:
% - raw pupil area
% - interpolated pupil area (blink-masked ±200 ms, PCHIP)
% - filtered (HP+LP, zero-phase) on the interpolated signal
% - per-trial baseline-corrected (relative to trial start, -0.5..0 s)
%
% Columns: eyeLinkTime, X, Y, pupilArea, convertedTime_s, trial, participantID,
%          message, eye, interpolatedPupilArea, filteredInterpolatedPupilArea,
%          perTrialBaselineCorrInterpFiltPupilArea

% -----------------------------
% Setup: paths
% -----------------------------
options  = specifyOptions;
ascDir   = options.paths.EDFtoASC;
pupilDir = options.paths.eyeData;
eyePlots = options.paths.eyeDataPlots;

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
    if ~isempty(recEye)
        trackedEye = upper(recEye{1});
    else
        trackedEye = 'R'; % fallback
    end
end

% -----------------------------
% Sampling rate (Hz) from header, fallback to 1000
% -----------------------------
% Prefer explicit RATE near SAMPLES/RECORD lines, e.g. "RATE 1000.00"
rateTok = regexp(rawText, '\bRATE\s+(\d+(?:\.\d+)?)', 'tokens', 'once');
if ~isempty(rateTok)
    Fs = str2double(rateTok{1});
elseif ~isempty(regexp(rawText, 'RECORD\s+CR\s+1000', 'once'))
    Fs = 1000;
else
    Fs = 1000; % sensible default for EyeLink CR 1000
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
% Trim header after RECORD line (tracked eye)
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
% Extract TRIAL start times and numbers: "MSG <time> TRIALID <n>"
% -----------------------------
trialMsgs = regexp(dataText, 'MSG\s+(\d+)\s+TRIALID\s+(\d+)', 'tokens');
trialStartTimes = [];
trialNumbers    = [];
if ~isempty(trialMsgs)
    trialStartTimes = cellfun(@(t) str2double(t{1}), trialMsgs);
    trialNumbers    = cellfun(@(t) str2double(t{2}), trialMsgs);
    [trialStartTimes, idx] = sort(trialStartTimes);
    trialNumbers = trialNumbers(idx);
end

% -----------------------------
% Remove saccade message lines (SSACC / ESACC)
% -----------------------------
dataText = regexprep(dataText, 'SSACC[^\n\r]*[\r\n]+', '');
dataText = regexprep(dataText, 'ESACC[^\n\r]*[\r\n]+', '');

% -----------------------------
% Parse numeric sample rows (first 4 numeric fields)
% -----------------------------
lines = regexp(dataText, '\r?\n', 'split');
sampleRows = {};
for i = 1:numel(lines)
    L = strtrim(lines{i});
    if isempty(L) || strncmp(L, 'MSG', 3), continue; end
    if ~isstrprop(L(1), 'digit'), continue; end
    toks = strsplit(L);                                        % split on whitespace
    if numel(toks) < 4, continue; end                          % need time + X + Y + pupil
    sampleRows(end+1,:) = toks(1:4); %#ok<AGROW>               % keep first 4 tokens only
end


assert(~isempty(sampleRows), 'No sample rows parsed from %s.', ascPath);

% Convert to numeric; non-numeric tokens (e.g., '.') become NaN
toNum = @(col) cellfun(@str2double, sampleRows(:,col));

timeVec  = toNum(1);   % EyeLink time (ms)
xVec     = toNum(2);   % X position ('.' -> NaN)
yVec     = toNum(3);   % Y position ('.' -> NaN)
pupilVec = toNum(4);   % Pupil area ('.' -> NaN)

nSamples = numel(timeVec);

% -----------------------------
% Assign trial numbers to each sample by time
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
% Messages (SBLINK, SFIX) – initial tagging
% -----------------------------
msgVec = strings(nSamples,1);
% Tag fixations by repeated X (conservative exact equality)
sameX   = [false; diff(xVec)==0];
sfixMask = sameX | [sameX(2:end); false];
msgVec(sfixMask) = "SFIX";


% =======================================================================
% NEW BLOCK A: Blink mask ±200 ms and PCHIP interpolation
% -----------------------------------------------------------------------
% 1) Build blink mask from raw pupil (==0) and expand it by ±200 ms.
% 2) Interpolate across masked samples using 'pchip'.
% =======================================================================
padSec = 0.200;                           % ±200 ms padding
padN   = max(1, round(padSec * Fs));      % samples to pad on each side

blinkMaskRaw = (pupilVec == 0);           % EyeLink zeros during blinks
% Expand (dilate) mask by convolution with a ones window of length (2*padN+1)
kernel       = ones(2*padN + 1, 1);
blinkMask    = conv(double(blinkMaskRaw), kernel, 'same') > 0;

% Update message to "SBLINK" for the whole padded span
msgVec(blinkMask) = "SBLINK";

% Interpolate across masked samples
interpPupil = pupilVec;
interpPupil(blinkMask) = NaN;
% If NaNs at edges, forward/backward fill small edges before PCHIP
if isnan(interpPupil(1))
    firstValid = find(~isnan(interpPupil), 1, 'first');
    if ~isempty(firstValid), interpPupil(1:firstValid-1) = interpPupil(firstValid); end
end
if isnan(interpPupil(end))
    lastValid = find(~isnan(interpPupil), 1, 'last');
    if ~isempty(lastValid), interpPupil(lastValid+1:end) = interpPupil(lastValid); end
end
% Final shape-preserving interpolation
interpPupil = fillmissing(interpPupil, 'pchip');

% =======================================================================
% NEW BLOCK B: Zero-phase filtering (HP + LP on interpolated signal)
% -----------------------------------------------------------------------
% Use Butterworth with filtfilt to avoid phase delay.
% Defaults: HP=0.05 Hz, LP=4 Hz (safe for 60–1000 Hz sampling).
% =======================================================================
HP = 0.05;   % Hz
LP = 4.00;   % Hz
nyq = Fs/2;
% Guard against pathological settings
HPn = max(HP/nyq, 1e-6);
LPn = min(LP/nyq, 0.999);

% Two-stage filter is numerically stable on long runs
[b1,a1] = butter(2, HPn, 'high');                   % 2nd order high-pass
[b2,a2] = butter(3, LPn, 'low');                    % 3rd order low-pass
tmpSig  = filtfilt(b1, a1, double(interpPupil));    % zero-phase HP
filtPup = filtfilt(b2, a2, tmpSig);                 % zero-phase LP


% =======================================================================
% NEW BLOCK B2 (anchored): Downsample to 100 Hz AFTER filtering,
% keeping each trial's original first timestamp as the first kept sample.
% -----------------------------------------------------------------------
targetFs = 100;
dsFactor = round(Fs / targetFs);

% Only proceed if real downsampling is needed and Fs is close to a multiple
if dsFactor > 1
    nSamples = numel(timeVec);
    keepMask = false(nSamples,1);

    % Helper to mark every dsFactor-th index within [a..b], anchored at a
    mark_every = @(a,b,step) (a:step:b);

    % 1) Trials present? Anchor within each trial block to its first index
    if any(~isnan(trialVec))
        % Find contiguous runs of the same trial ID (so repeated IDs are handled per-run)
        idx = 1;
        while idx <= nSamples
            if isnan(trialVec(idx))
                % handle NaN segment later
                idx = idx + 1;
                continue;
            end
            % start of a trial run
            tID = trialVec(idx);
            j = idx;
            while j <= nSamples && trialVec(j) == tID
                j = j + 1;
            end
            runStart = idx;
            runEnd   = j - 1;

            % Anchor at runStart for this trial
            keepIdx = mark_every(runStart, runEnd, dsFactor);
            keepMask(keepIdx) = true;

            idx = j; % advance to next run
        end
    end

    % 2) Handle NaN (non-trial) segments with a simple global anchor
    % Choose the first available index in each NaN run as its anchor.
    idx = 1;
    while idx <= nSamples
        if ~isnan(trialVec(idx))
            idx = idx + 1;
            continue;
        end
        % start of NaN run
        j = idx;
        while j <= nSamples && isnan(trialVec(j))
            j = j + 1;
        end
        runStart = idx;
        runEnd   = j - 1;

        keepIdx = mark_every(runStart, runEnd, dsFactor);
        keepMask(keepIdx) = true;

        idx = j;
    end

    % Safety: always keep the very first sample
    keepMask(1) = true;

    % Apply mask to ALL time-aligned vectors
    timeVec      = timeVec(keepMask);
    pupilVec     = pupilVec(keepMask);      % raw pupil (table consistency)
    interpPupil  = interpPupil(keepMask);
    filtPup      = filtPup(keepMask);
    trialVec     = trialVec(keepMask);
    msgVec       = msgVec(keepMask);
    xVec         = xVec(keepMask);
    yVec         = yVec(keepMask);

    % Update rate and count
    Fs       = targetFs;
    nSamples = numel(timeVec);
end

% =======================================================================
% NEW BLOCK C: Per-trial baseline correction (relative to trial start)
% -----------------------------------------------------------------------
% For each trial T, compute baseline = mean over [-0.1, 0] s relative to
% TRIALID start time, using the filtered & interpolated signal.
% Samples before first trial or where no baseline window exists -> NaN.
% =======================================================================
baselineWin = [-0.1, 0.0];   % seconds relative to trial start
perTrialBC  = NaN(nSamples,1);

if ~isempty(trialStartTimes)
    % Trial starts in seconds relative to sync pulse
    trialStart_s = (trialStartTimes - syncPulse) / 1000;
    % Map each trial number to its start time (trials can be 0..79)
    uTrials = unique(trialNumbers(:)');
    % Some datasets repeat trial IDs; we use the first occurrence per ID in order
    % but safer is to use the sorted timeline mapping from trialStartTimes/trialNumbers:
    % Build a lookup for the *active* trial for each sample via trialVec already set.
    for i = 1:numel(uTrials)
        tID = uTrials(i);
        % Find time of this trial's start (use first occurrence in time order)
        idxStarts = find(trialNumbers == tID);
        if isempty(idxStarts), continue; end
        tStart = trialStart_s(idxStarts(1));

        % Baseline window indices
        t0 = tStart + baselineWin(1);
        t1 = tStart + baselineWin(2);
        idxBase = ( (timeVec - syncPulse)/1000 >= t0 ) & ( (timeVec - syncPulse)/1000 <= t1 );

        if ~any(idxBase)
            continue; % no baseline samples available
        end
        baseVal = mean(filtPup(idxBase), 'omitnan');

        % Apply to all samples *within this trial occurrence in time*
        % Use trialVec to select rows belonging to this trial ID
        inThisTrial = (trialVec == tID);
        perTrialBC(inThisTrial) = filtPup(inThisTrial) - baseVal;
    end
end

% -----------------------------
% Build output table
% -----------------------------
convertedTime_s = (timeVec - syncPulse) / 1000;

% Create these AFTER downsampling so lengths match nSamples
eyeCol         = repmat(string(trackedEye), nSamples, 1);             % "R" or "L" as strings
participantCol = repmat(str2double(participantID_str), nSamples, 1);  % numeric ID column

% New columns:
%   interpolatedPupilArea                 -> interpPupil
%   filteredInterpolatedPupilArea         -> filtPup
%   perTrialBaselineCorrInterpFiltPupilArea -> perTrialBC
eyeDataTbl = table( ...
    timeVec, xVec, yVec, pupilVec, convertedTime_s, trialVec, ...
    participantCol, msgVec, eyeCol, ...
    interpPupil, filtPup, perTrialBC, ...
    'VariableNames', { ...
        'eyeLinkTime','X','Y','pupilArea','convertedTime_s','trial', ...
        'participantID','message','eye', ...
        'interpolatedPupilArea','filteredInterpolatedPupilArea','perTrialBaselineCorrInterpFiltPupilArea'} ...
);

f = figure; hold on;
plot(eyeDataTbl.convertedTime_s, eyeDataTbl.pupilArea, 'LineWidth', 1.3);
plot(eyeDataTbl.convertedTime_s, eyeDataTbl.interpolatedPupilArea, 'LineWidth', 1.3);
plot(eyeDataTbl.convertedTime_s, eyeDataTbl.filteredInterpolatedPupilArea, 'LineWidth', 1.3);
plot(eyeDataTbl.convertedTime_s, eyeDataTbl.perTrialBaselineCorrInterpFiltPupilArea, 'LineWidth', 1.3);
hold off;

xlabel('Time (s)');
ylabel('Pupil area (a.u.)');
title('Pupil time series');
legend({'pupilArea','interpolatedPupilArea', 'filteredInterpolatedPupilArea', 'perTrialBaselineCorrInterpFiltPupilArea'}, 'Location','best');
grid on;

figPath = fullfile(eyePlots, sprintf('pupil_plot_%s.fig', string(participantID)));
savefig(f, figPath);                  
fprintf('Saved: %s\n', figPath);

% -----------------------------
% Save CSV
% -----------------------------
csvFileName = sprintf('%s_cface_eyeData.csv', participantID_str);
csvPath = fullfile(pupilDir, csvFileName);
writetable(eyeDataTbl, csvPath);
fprintf('Saved: %s (%d samples)\n', csvPath, nSamples);

end