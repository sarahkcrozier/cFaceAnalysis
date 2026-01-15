function [eyeDataTbl, cFacePupil] = cFacePupilArea(participantID)
% cFacePupilArea
% Parse EyeLink ASC file for the specified participant and save a CSV with:
% - raw pupil area
% - interpolated pupil area (blink-masked ±200 ms, PCHIP)
% - filtered (LP, zero-phase) on the interpolated signal
% - per-trial baseline-corrected (relative to trial start, -0.1..0 s) \
% CHECK THIS - HAVE ADDED FURTHER OUTPUTS
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

% get the timeings for calculating baseline correction andpeak windows
segmentsTable = cFaceTrialSegments(participantID)
trial_s = segmentsTable.instruct_onset % an array of all the trial start times for a given participant
stimResponse = segmentsTable.stimMove_onset % an array of all the start times for the stimulus response windows
partEmotion = segmentsTable.ptemot % an array of the participant emoted emotions [AN/HA] per trial
condition = segmentsTable.condition % an array of the condition [congruent/incongruent] for each trial

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
% NEW BLOCK A: Filtering using filterRawData.m
% =======================================================================
% 1. Get default settings from the Kret & Sjak-Shie (2018) library
filtSettings = filterRawData(); 

% 2. Customize padding to match your current ±150ms requirement
filtSettings.gapPadding_backward = 150; 
filtSettings.gapPadding_forward  = 150;

% 3. Run the filter (uses speed, deviation, and island filters)
% timeVec (ms) and pupilVec (raw area) are passed to the function
[isValid, ~, ~] = filterRawData(timeVec, pupilVec, filtSettings);

% 4. Create the cleaned signal for subsequent interpolation
% Mark invalid samples as NaN based on the advanced logical output
interpPupil = pupilVec;
interpPupil(~isValid) = NaN;

% Update the message vector for consistency in your output table
msgVec(~isValid) = "REJECTED_BY_FILTER";


% =======================================================================
% BLOCK B: Zero-phase LOW-PASS filtering on interpolated signal
% -----------------------------------------------------------------------
% Use Butterworth with filtfilt to avoid phase delay.
% Default LP = 4 Hz (safe for 60–1000 Hz sampling).

% This version interpolates internal gaps but leaves NaNs at edges,
% then filters only the valid segment to avoid edge bias.
% =======================================================================

LP = 4.00;                 % low-pass cutoff frequency in Hz
nyq = Fs / 2;              % Nyquist frequency
LPn = min(LP/nyq, 0.999);  % normalized cutoff (guard upper bound)

% 3rd-order low-pass Butterworth filter
[b,a] = butter(3, LPn, 'low');

% SIMPLE FILTER: Apply zero-phase filtering
% filtPup = filtfilt(b, a, double(interpPupil));

% Amended filter: 
% Interpolate internal gaps only; keep edge NaNs (no extrapolation)
interpPupil = fillmissing(interpPupil, 'pchip', 'EndValues', 'none');

% Filter only the contiguous valid portion
filtPup  = interpPupil;
validIdx = find(~isnan(interpPupil));
if ~isempty(validIdx)
    firstValid = validIdx(1); lastValid = validIdx(end);
    seg = double(interpPupil(firstValid:lastValid));

    minLen = 3 * max(length(a), length(b));   % filtfilt safety
    if numel(seg) >= minLen && all(isfinite(seg))
        segFilt = filtfilt(b, a, seg);
        filtPup(firstValid:lastValid) = segFilt;
    else
        filtPup(firstValid:lastValid) = seg;   % too short: skip filtering
    end
end



% =======================================================================
%  BLOCK C: Downsample to 100 Hz AFTER filtering,
% keeping every dsFactor-th sample consecutively (no trial anchoring).
% -----------------------------------------------------------------------
targetFs = 100;
dsFactor = round(Fs / targetFs);

if dsFactor > 1
    % Keep every dsFactor-th sample starting at the very first sample
    keepMask = false(nSamples,1);
    keepMask(1:dsFactor:end) = true;

    % Apply to ALL time-aligned vectors so lengths remain consistent
    timeVec      = timeVec(keepMask);
    pupilVec     = pupilVec(keepMask);      % raw pupil (for table consistency)
    interpPupil  = interpPupil(keepMask);
    filtPup      = filtPup(keepMask);
    trialVec     = trialVec(keepMask);
    msgVec       = msgVec(keepMask);
    xVec         = xVec(keepMask);
    yVec         = yVec(keepMask);

    % Update Fs and sample count
    Fs       = targetFs;
    nSamples = numel(timeVec);
end

% =======================================================================
% BLOCK D: Per-trial baseline correction using segmentsTable strimulus
% response timings
% (stimResponse)
% -----------------------------------------------------------------------
% trial_s: vector (seconds) of trial start times, consecutive (1..N).
% For each trial n:
%   - Baseline window 10.1 to 0 prior to stimulus onset
% Outputs:
%   perTrialBC      : filtPup minus per-stimulus reposne baseline 
%   meanBaselineVec : baseline mean replicated across all samples in that trial window
% =======================================================================

baselineWin = [-0.1, 0.0];            % seconds relative to trial start
perTrialBC      = NaN(nSamples,1);
meanBaselineVec = NaN(nSamples,1);

% Ensure trial_s and stimResponseTimings are clean, ascending column vector in seconds
trial_s = trial_s(:);
stimResponseTimings = stimResponse(:);     

N = numel(trial_s);
N = min(numel(trial_s), numel(stimResponseTimings));                    % guard against length mismatch
if N == 0
    warning('stimResponse or trial_s is empty; skipping baseline correction.');
else
    % Sample times (s) on same reference as trial_s
    t_sample_s = (timeVec - syncPulse) / 1000;

    % Define trial end points: next start, or end of data for the last trial
    lastEnd = max(t_sample_s);   % end of recording
    trial_e = [trial_s(2:end); lastEnd + eps];  % half-open intervals [start, end)

    for n = 1:N
        tStart = trial_s(n);
        tEnd   = trial_e(n);

        % Samples in this trial window
        inTrial = (t_sample_s >= tStart) & (t_sample_s < tEnd);
        if ~any(inTrial) || isnan(stimResponseTimings(n)) % skip if missing stim onset
            continue;
        end

        % Baseline window relative to stimulus response onset for this trial
        t0 = stimResponseTimings(n) + baselineWin(1);                     % -0.10 s relative to stimMove_onset
        t1 = stimResponseTimings(n) + baselineWin(2);                     %  0.00 s relative to stimMove_onset
        idxBase = (t_sample_s >= t0) & (t_sample_s <= t1);

        if ~any(idxBase)
            % No baseline samples available; leave NaNs for this trial
            continue;
        end

        % Baseline from the FILTERED signal
        baseVal = mean(filtPup(idxBase), 'omitnan');

        % Apply baseline to all samples in this trial window
        perTrialBC(inTrial)      = filtPup(inTrial) - baseVal;
        meanBaselineVec(inTrial) = baseVal;
    end
end

% =======================================================================
% BLOCK E: Build trial response window table, save CSV, compute peaks, plots
% -----------------------------------------------------------------------
% Window starts: stimResponse (seconds, absolute, same ref as convertedTime_s)
% Window length: 2.75 s
% Signal used: perTrialBaselineCorrInterpFiltPupilArea  (variable: perTrialBC)
% Table columns: [time_s (relative 0..2.75), trial01 .. trial80]
% CSV name: '[participantID]cFaceResponseWindow.csv'
% Also computes peakPupil (max per trial within window)
% And saves 9 figures under eyePlots as '[participantID]_[plottype].fig'
% =======================================================================

% --- Config ---
winLen_s = 2.75;

% --- Sanity: ensure vectors are proper types/shapes ---
stimResponse = stimResponse(:);                 % trial window starts (s), Nx1
nTrialsAvail = numel(stimResponse);
nTrials      = min(80, nTrialsAvail);           % clamp to 80

% Conditions/emotion to string arrays for indexing
condition   = string(condition(:));
partEmotion = string(partEmotion(:));

% --- Time axis for each window (relative, common across all trials) ---
dt = 1 / Fs;
relTime = (0:dt:winLen_s)';                     % column vector, length nT
nT = numel(relTime);

% --- Absolute sample time (seconds) for the full series ---
t_sample_s = (timeVec - syncPulse) / 1000;

% --- Preallocate window matrix: nT x 80 ---
winMat = NaN(nT, 80);

% --- Fill each trial column by interpolating perTrialBC at t = stim + relTime ---
for n = 1:nTrials
    tAbs = stimResponse(n) + relTime;           % absolute target times for trial n
    winMat(:, n) = interp1(t_sample_s, perTrialBC, tAbs, 'linear', NaN);
end

% --- Peak per trial within window (1..80); NaN for trials without data ---
peakPupil = max(winMat, [], 1, 'omitnan')';     % 80x1 vector

% --- Build table: first col = relTime (0..2.75), then trial01..trial80 ---
trialCols = compose('trial%02d', 1:80);
winTbl = array2table([relTime, winMat], 'VariableNames', [{'time_s'}, trialCols]);

% --- Save CSV ---
csvWinName = sprintf('%scFaceResponseWindow.csv', participantID_str);
csvWinPath = fullfile(pupilDir, csvWinName);
writetable(winTbl, csvWinPath);
fprintf('Saved window table: %s  (rows=%d, trials=80)\n', csvWinPath, nT);

% =========================
% Plotting helper function
% =========================
    function plot_and_save(idx, titleStr, fileTag)
    if isempty(idx), return; end
    
    % Create invisible figure
    f = figure('Visible','off'); hold on;
    plot(relTime, winMat(:, idx), 'LineWidth', 1.0);   % one line per selected trial
    hold off; grid on;
    xlabel('Time from window start (s)');
    ylabel('Baseline-corrected pupil (a.u.)');
    title(sprintf('%s (%d trials)', titleStr, numel(idx)));
    
    % Paths
    figPath = fullfile(eyePlots, sprintf('%s_%s.fig', participantID_str, fileTag));
    pngPath = fullfile(eyePlots, sprintf('%s_%s.png', participantID_str, fileTag));
    
    % Save in MATLAB format and PNG
    savefig(f, figPath);
    exportgraphics(f, pngPath, 'Resolution',300);  % high-quality PNG
    
    close(f);
end

% -------------------------------------------------------------
% Build trial index sets (1..nTrials) using string comparisons
% -------------------------------------------------------------

% Ensure condition and partEmotion are strings, column vectors
condition   = string(condition(:));
partEmotion = string(partEmotion(:));

% Number of valid trials (usually 80, but guard if inputs shorter)
nTrials = min([80, numel(condition), numel(partEmotion)]);

% Universal set of trial indices
allIdx = 1:nTrials;

% Condition-based subsets
congIdx   = find(strcmpi(condition(1:nTrials),   "congruent"));
incongIdx = find(strcmpi(condition(1:nTrials),   "incongruent"));

% Emotion-based subsets
HAIdx = find(strcmpi(partEmotion(1:nTrials), "HA")); % all the trials where participant instructed emotion is HA
ANIdx = find(strcmpi(partEmotion(1:nTrials), "AN")); % all the trials where participant instructed emotion is AN

% Combined subsets
HAcongIdx   = intersect(HAIdx, congIdx);
ANcongIdx   = intersect(ANIdx, congIdx);
HAincongIdx = intersect(HAIdx, incongIdx);
ANincongIdx = intersect(ANIdx, incongIdx);

% --- Make the 9 plots ---
plot_and_save(allIdx,      'All trials',                   'allTrials');
plot_and_save(congIdx,     'Congruent trials',             'congruentTrials');
plot_and_save(incongIdx,   'Incongruent trials',           'incongruentTrials');
plot_and_save(HAIdx,       'HA trials',                    'HAtrials');
plot_and_save(ANIdx,       'AN trials',                    'ANtrials');
plot_and_save(HAcongIdx,   'HA congruent',                 'HAcong');
plot_and_save(ANcongIdx,   'AN congruent',                 'ANcong');
plot_and_save(HAincongIdx, 'HA incongruent',               'HAincong');
plot_and_save(ANincongIdx, 'AN incongruent',               'ANincong');


% -----------------------------
% Build output table
% -----------------------------
convertedTime_s = (timeVec - syncPulse) / 1000;

% Create these AFTER downsampling so lengths match nSamples
eyeSide         = repmat(string(trackedEye), nSamples, 1);             % "R" or "L" as strings
participantCol = repmat(str2double(participantID_str), nSamples, 1);  % numeric ID column

%  columns:
%   interpolatedPupilArea                 -> interpPupil
%   filteredInterpolatedPupilArea         -> filtPup
%   perTrialBaselineCorrInterpFiltPupilArea -> perTrialBC
eyeDataTbl = table( ...
    timeVec, xVec, yVec, pupilVec, convertedTime_s, trialVec, ...
    participantCol, msgVec, eyeSide, ...
    interpPupil, filtPup, perTrialBC, meanBaselineVec, ...
    'VariableNames', { ...
        'eyeLinkTime','X','Y','pupilArea','convertedTime_s','trial', ...
        'participantID','message','eye', ...
        'interpolatedPupilArea','filteredInterpolatedPupilArea','perTrialBaselineCorrInterpFiltPupilArea','perTrial100msBaseline'} ...
);

% =======================================================================
% Task baseline: 10 s prior to trial 1
% -----------------------------------------------------------------------
% trial_s is an array of trial start times (in seconds, same reference as
% convertedTime_s). Take the first element (trial 1 start), and average
% filteredInterpolatedPupilArea over the [-10, 0] s window.
% =======================================================================

if ~isempty(trial_s)
    firstTrialStart = trial_s(1);  % seconds
    baselineStart   = firstTrialStart - 10;
    baselineEnd     = firstTrialStart;

    % Find samples within this window
    idxBaseline = (convertedTime_s >= baselineStart) & (convertedTime_s < baselineEnd);

    if any(idxBaseline)
        cFacePupilBaseline = median(filtPup(idxBaseline), 'omitnan');
    else
        cFacePupilBaseline = NaN;  % no samples available
    end
else
    cFacePupilBaseline = NaN;
end


% =======================================================================
% Build cFacePupil struct (trial-wise + averages)
% -----------------------------------------------------------------------

% Compute per-trial peak pupil (max sample) (NaN if trial has no data)
allTrialPeaks = NaN(1, nTrials);
for t = 1:nTrials
    colData = winMat(:, t);
    if all(isnan(colData))
        continue;  % leave as NaN if no samples
    end
    allTrialPeaks(t) = max(colData, [], 'omitnan');
end

% Build struct with summary measures
cFacePupil = struct();
cFacePupil.allTrialPeaks     = allTrialPeaks;
cFacePupil.peakAverage       = mean(allTrialPeaks, 'omitnan');
cFacePupil.congruentAverage  = mean(allTrialPeaks(congIdx),   'omitnan');
cFacePupil.incongruentAverage= mean(allTrialPeaks(incongIdx), 'omitnan');
cFacePupil.baseline          = cFacePupilBaseline;
cFacePupil.eyeSide           = eyeSide(1);
cFacePupil.HAHA              = mean(allTrialPeaks(HAcongIdx), 'omitnan');
cFacePupil.ANHA              = mean(allTrialPeaks(HAincongIdx), 'omitnan');
cFacePupil.ANAN              = mean(allTrialPeaks(ANcongIdx), 'omitnan');
cFacePupil.HAAN              = mean(allTrialPeaks(ANincongIdx), 'omitnan');

% Print to MATLAB command window
fprintf('Peak pupil summary for participant %s:\n', participantID_str);
fprintf('  preTask baseline:      %.4f\n', cFacePupil.baseline);
fprintf('  Overall average:      %.4f\n', cFacePupil.peakAverage);
fprintf('  Congruent average:    %.4f\n', cFacePupil.congruentAverage);
fprintf('  Incongruent average:  %.4f\n', cFacePupil.incongruentAverage);
fprintf('  (See cFacePupil.allTrialPeaks for trial-wise values)\n');

%%%%%% PLOT FIGURES

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