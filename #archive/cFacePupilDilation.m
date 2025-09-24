function pupil = cFacePupilDilation(participantID, winDur)
% cFacePupilDilation  Plot pupil area for N seconds after stimMove_onset per trial,
%                     coloured by emotion (ptemot) and styled by congruency.
%                     Adds per-trial 200 ms baseline, Δpupil traces, and peak dilation (no smoothing).
%
% Usage:
%   pupil = cFacePupilDilation(133)        % default 2.75 s window
%   pupil = cFacePupilDilation('133', 5.0) % custom window length in seconds
%
% Returns:
%   pupil : struct with fields
%       .baseline.perTrial   (200 ms pre-stim baseline per trial)
%       .baseline.global     (mean pupil over 10 s prior to Trial 1 start)
%       .baseline.mean       (grand mean of per-trial baselines)
%       .peak.perTrial       (peak Δpupil per trial; no smoothing)
%       .peak.timePerTrial   (time-to-peak in seconds, relative to stim)
%       .peak.mean           (average of peak Δpupil across all trials)
%       .avCongruent         (mean peak Δpupil over congruent trials)
%       .avIncongruent       (mean peak Δpupil over incongruent trials)
%       .angry               (mean peak Δpupil over AN trials)
%       .happy               (mean peak Δpupil over HA trials)
%       .ANincongruent       (mean peak Δpupil over AN & incongruent trials)
%       .HAincongruent       (mean peak Δpupil over HA & incongruent trials)
%       .Ns                  (counts for each grouping)

if nargin < 2 || isempty(winDur), winDur = 2.75; end

options = specifyOptions;
ascDir  = options.paths.ASCtoMAT;
plotDir = options.paths.eyeDataPlots;

% Normalize ID
if isnumeric(participantID), participantID = num2str(participantID); end

% Ensure MAT exists
try
    ascToMatExtract(participantID);
catch ME
    warning('ascToMatExtract failed or not on path: %s', ME.message);
end

% Locate newest *_eye.mat
allMats  = dir(fullfile(ascDir, '*_eye.mat'));
mask     = contains({allMats.name}, participantID, 'IgnoreCase', true);
mats     = allMats(mask);
if isempty(mats)
    error('No *_eye.mat found for participant "%s" in %s', participantID, ascDir);
end
[~, newestIdx] = max([mats.datenum]);
matPath = fullfile(mats(newestIdx).folder, mats(newestIdx).name);

% Derive baseName for saving outputs (needed for CSV and plots)
[~, baseName] = fileparts(regexprep(matPath, '_eye\.mat$', ''));
baseName = regexprep(baseName, '_eye$', '');

% Load data
S = load(matPath);
assert(isfield(S,'eyeData') && isfield(S,'eyeDataTrialTimestamp'), ...
    'Expected eyeData and eyeDataTrialTimestamp in %s', matPath);
eyeData = S.eyeData;
eyeTT   = S.eyeDataTrialTimestamp;

% Pupil channel
if ismember('pupilArea_lp4Hz', eyeData.Properties.VariableNames)
    pupilRaw = eyeData.pupilArea_lp4Hz;
else
    pupilRaw = eyeData.pupilArea;
end

% Load behavioral segments
seg = cFaceTrialSegments(participantID);
if ~ismember('trialNo', seg.Properties.VariableNames)
    seg.trialNo = (1:height(seg))';
end

% Must-have columns
needTT  = {'Trial','StartTime_s'};
needSeg = {'stimMove_onset','instruct_onset','ptemot'};
assert(all(ismember(needTT, eyeTT.Properties.VariableNames)), 'eyeDataTrialTimestamp missing needed columns.');
assert(all(ismember(needSeg, seg.Properties.VariableNames)), 'cFaceTrialSegments missing needed columns.');

% Join by trial number
trialMeta = innerjoin(eyeTT, seg, 'LeftKeys','Trial', 'RightKeys','trialNo');

% Compute aligned stim time in EyeLink seconds
offset_s = trialMeta.stimMove_onset - trialMeta.instruct_onset;
stim_s   = trialMeta.StartTime_s + offset_s;
stimEnd  = stim_s + winDur;

fprintf('t range: [%.3f, %.3f]\n', min(eyeData.time_s), max(eyeData.time_s));
fprintf('last 5 stim_s: '); fprintf(' %.3f', stim_s(max(1,end-4):end)); fprintf('\n');
fprintf('last 5 stimEnd:'); fprintf(' %.3f', stimEnd(max(1,end-4):end)); fprintf('\n');

% -------------------------------------------------------------------------
% GLOBAL BASELINE (mean pupil over the 10 s before Trial 1 start) — retained
% -------------------------------------------------------------------------
t = eyeData.time_s;
t1 = eyeTT.StartTime_s(eyeTT.Trial == 1);
if isempty(t1), t1 = min(eyeTT.StartTime_s); end

% baselining window [t1 - 10, t1)
winStart = t1 - 10.0;
t0Avail  = min(t);
lo       = max(winStart, t0Avail);
hi       = t1;

mask10s = (t >= lo) & (t < hi);
if any(mask10s)
    globalBaseline = mean(pupilRaw(mask10s), 'omitnan');
else
    maskAllPre = t < t1;
    if any(maskAllPre)
        globalBaseline = mean(pupilRaw(maskAllPre), 'omitnan');
    else
        maskFirst = t < (t0Avail + 10);
        globalBaseline = mean(pupilRaw(maskFirst), 'omitnan');
    end
end

% -------------------------------------------------------------------------
% GROUP FLAGS (explicit mapping for ptemot; congruency if present)
% -------------------------------------------------------------------------
emoStr = lower(string(trialMeta.ptemot));
isAN = (emoStr == "angry") | (emoStr == "an");
isHA = (emoStr == "happy") | (emoStr == "ha");

hasCong  = ismember('cong', trialMeta.Properties.VariableNames);
if hasCong
    isCong   = (trialMeta.cong == 1);
    isIncong = ~isCong;
else
    isCong   = false(height(trialMeta),1);
    isIncong = false(height(trialMeta),1);
end

% -------------------------------------------------------------------------
% PER-TRIAL 200 ms BASELINE, ΔPUPIL (no smoothing), PEAK DILATION
% -------------------------------------------------------------------------
fs = 1/median(diff(t), 'omitnan'); % ~100 Hz
nTrials = height(trialMeta);

perTrialBaseline = nan(nTrials,1);
peakDelta        = nan(nTrials,1);
peakTime         = nan(nTrials,1);

% Collect for plotting y-lims
yCandidatesAN = [];
yCandidatesHA = [];

for k = 1:nTrials
    % 200 ms baseline window: [-0.2, 0) s relative to stim
    bmask = (t >= (stim_s(k) - 0.2)) & (t < stim_s(k));
    if ~any(bmask)
        % fallback: last ~0.2 s worth of samples available before stim
        preIdx = find(t < stim_s(k));
        if ~isempty(preIdx)
            lastN = max(1, round(0.2 * fs));
            bmask = preIdx(max(1, numel(preIdx) - lastN + 1):end);
        end
    end
    perTrialBaseline(k) = mean(pupilRaw(bmask), 'omitnan');

% Clamp the post-stimulus window to the available data range
tmin = min(t);
tmax = max(t);
st  = max(stim_s(k), tmin);   % stim cannot be earlier than first sample
en  = min(stimEnd(k), tmax);  % stimEnd cannot be later than last sample

wmask = (t >= st) & (t <= en);

% Require at least some samples (e.g., 100 ms worth) to count as evaluable
minSamples = max(1, round(0.1 * fs));   % fs is ~100 Hz
if nnz(wmask) < minSamples
    peakDelta(k) = NaN;
    peakTime(k)  = NaN;
    continue;   % skip this trial
end

rt = t(wmask) - stim_s(k);   % still keep time relative to intended stim
rp = pupilRaw(wmask);

% Δpupil without smoothing
dp = rp - perTrialBaseline(k);

% Peak and time-to-peak
[peakDelta(k), maxIdx] = max(dp);
peakTime(k) = rt(maxIdx);

    % For y-lims
    if isAN(k), yCandidatesAN = [yCandidatesAN; dp]; end %#ok<AGROW>
    if isHA(k), yCandidatesHA = [yCandidatesHA; dp]; end %#ok<AGROW>
end

% ---------- TRIAL-LEVEL AUDIT TABLE (diagnose dropped trials) ----------
% Prepare label columns
Trial       = trialMeta.Trial;                        % 1..N
Emotion     = strings(nTrials,1);
Emotion(isAN) = "AN";
Emotion(isHA) = "HA";
Emotion(~isAN & ~isHA) = "Other";

if hasCong
    Congruent   = logical(isCong);
    Incongruent = logical(isIncong);
else
    Congruent   = false(nTrials,1);
    Incongruent = false(nTrials,1);
end

% Re-compute simple validity flags per trial for diagnostics
hasBaselineWin = false(nTrials,1);
hasPostWin     = false(nTrials,1);
dpAnyData      = false(nTrials,1);
dpAllNaN       = false(nTrials,1);
dropReason     = strings(nTrials,1);

for k = 1:nTrials
    % baseline window [-0.2,0)
    bmask = (t >= (stim_s(k) - 0.2)) & (t < stim_s(k));
    if ~any(bmask)
        preIdx = find(t < stim_s(k));
        if ~isempty(preIdx)
            lastN = max(1, round(0.2 * fs));
            bmask = preIdx(max(1, numel(preIdx) - lastN + 1):end);
        end
    end
    hasBaselineWin(k) = any(bmask);

    % post-stim window [0, winDur]
    wmask = (t >= stim_s(k)) & (t <= stimEnd(k));
    hasPostWin(k) = any(wmask);

    if hasPostWin(k)
        rp = pupilRaw(wmask);
        dp = rp - perTrialBaseline(k);
        dpAnyData(k) = any(~isnan(dp));
        dpAllNaN(k)  = all(isnan(dp));
    else
        dpAnyData(k) = false;
        dpAllNaN(k)  = true;
    end

    % Drop reason (only if peak is NaN / not evaluable)
    if isnan(peakDelta(k))
        if ~hasBaselineWin(k) && ~hasPostWin(k)
            dropReason(k) = "no baseline & no post window";
        elseif ~hasBaselineWin(k)
            dropReason(k) = "no baseline window";
        elseif ~hasPostWin(k)
            dropReason(k) = "no post-stim window";
        elseif dpAllNaN(k)
            dropReason(k) = "all Δpupil samples NaN";
        else
            dropReason(k) = "peak not found (other)";
        end
    else
        dropReason(k) = "";
    end
end

Evaluable = ~isnan(peakDelta);

% Build the table
trialTbl = table( ...
    Trial, ...
    Emotion, Congruent, Incongruent, ...
    stim_s, stimEnd, ...
    perTrialBaseline, ...
    peakDelta, peakTime, ...
    hasBaselineWin, hasPostWin, dpAnyData, dpAllNaN, ...
    Evaluable, dropReason, ...
    'VariableNames', { ...
        'Trial','Emotion','Congruent','Incongruent', ...
        'StimTime_s','StimEnd_s', ...
        'Baseline200ms','PeakDelta','PeakTime_s', ...
        'HasBaselineWindow','HasPostWindow','DeltaHasData','DeltaAllNaN', ...
        'Evaluable','DropReason'});

% Save CSV next to plots (and include in struct)
csvPath = fullfile(plotDir, sprintf('%s_trial_level_pupil.csv', baseName));
try
    writetable(trialTbl, csvPath);
    fprintf('Wrote trial-level audit CSV: %s\n', csvPath);
catch ME
    warning('Could not write trial-level CSV: %s', ME.message);
end

% -------------------------------------------------------------------------
% DropReason + group diagnostics (toolbox-free)
% -------------------------------------------------------------------------
fprintf('Trial counts — total=%d | evaluable=%d | dropped=%d\n', ...
    height(trialTbl), nnz(trialTbl.Evaluable), nnz(~trialTbl.Evaluable));

fprintf('Drop reasons (counts):\n');
dr = unique(trialTbl.DropReason);
for i = 1:numel(dr)
    if dr(i) == ""  % evaluable trials
        continue
    end
    fprintf('  %-28s : %d\n', dr(i), sum(trialTbl.DropReason == dr(i)));
end

if hasCong
    fprintf('Counts by Congruent (Eval=1): Cong=1: %d | Cong=0: %d\n', ...
        sum(trialTbl.Congruent & trialTbl.Evaluable), ...
        sum(~trialTbl.Congruent & trialTbl.Evaluable));
end

fprintf('Counts by Emotion (Eval=1): AN: %d | HA: %d | Other: %d\n', ...
    sum(trialTbl.Evaluable & trialTbl.Emotion=="AN"), ...
    sum(trialTbl.Evaluable & trialTbl.Emotion=="HA"), ...
    sum(trialTbl.Evaluable & trialTbl.Emotion=="Other"));

% Attach to output struct later:
% pupil.trials = trialTbl;

% OPTIONAL: quick sanity checks in console
fprintf('Trial counts — total=%d | evaluable=%d | dropped=%d\n', ...
    height(trialTbl), nnz(trialTbl.Evaluable), nnz(~trialTbl.Evaluable));

if hasCong
    fprintf('Congruent vs Evaluable (rows=Congruent [0/1], cols=Evaluable [0/1]):\n');
    try
        % Requires Statistics & ML Toolbox
        disp(crosstab(trialTbl.Congruent, trialTbl.Evaluable));
    catch
        % Fallback without toolbox: build 2x2 counts manually
        % Rows: Congruent = 0,1 ; Cols: Evaluable = 0,1
        ct = [ ...
            sum(~trialTbl.Congruent & ~trialTbl.Evaluable), sum(~trialTbl.Congruent & trialTbl.Evaluable); ...
            sum( trialTbl.Congruent & ~trialTbl.Evaluable), sum( trialTbl.Congruent & trialTbl.Evaluable) ...
        ];
        T = array2table(ct, ...
            'VariableNames', {'Eval_0','Eval_1'}, ...
            'RowNames', {'Cong_0','Cong_1'});
        disp(T)
    end
end
fprintf('Emotion vs Evaluable (rows=Emotion, cols=Evaluable[0/1]):\n');
try
    disp(crosstab(categorical(trialTbl.Emotion), trialTbl.Evaluable));
catch
    % crosstab requires Statistics & Machine Learning Toolbox; ignore if missing
end

% -------------------------------------------------------------------------
% PLOTTING: Δpupil overlays per emotion (no smoothing)
% -------------------------------------------------------------------------
plotAN = find(isAN);
plotHA = find(isHA);

mkFig = @(name) figure('Color','w','Units','normalized','Position',[0.2 0.2 0.55 0.5], ...
                       'Name', sprintf('%s — ID %s', name, participantID));
[folderOnly, baseName] = fileparts(regexprep(matPath, '_eye\.mat$', ''));
baseName = regexprep(baseName, '_eye$', '');
if ~exist(plotDir,'dir'), mkdir(plotDir); end

if ~isempty(plotAN)
    figAN = mkFig('ΔPupil after stim — AN trials');
    hold on; box on; grid on;
    if ~isempty(yCandidatesAN)
        yl = [prctile(yCandidatesAN,2) prctile(yCandidatesAN,98)];
        if all(isfinite(yl)), ylim(yl); end
    end
    for k = plotAN(:)'
        wmask = (t >= stim_s(k)) & (t <= stimEnd(k)); if ~any(wmask), continue; end
        rt = t(wmask) - stim_s(k);
        dp = pupilRaw(wmask) - perTrialBaseline(k);
        if hasCong && isCong(k)
            plot(rt, dp, '-', 'LineWidth', 1.2, 'Color', [0 0.4470 0.7410 0.35]); % congruent
        else
            plot(rt, dp, '-', 'LineWidth', 1.2, 'Color', [0.8500 0.3250 0.0980 0.35]); % incongruent/other
        end
    end
    yline(0, ':', 'Δ from 200 ms pre-stim', 'Color',[0.3 0.3 0.3], 'LineWidth', 1);
    xlabel('Time since stim (s)'); ylabel('ΔPupil (baseline-subtracted)');
    title(sprintf('ΔPupil %0.1fs after stim — AN (ID %s)', winDur, participantID), 'Interpreter','none');
    savefig(figAN, fullfile(plotDir, sprintf('%s_overlay_AN_dP_%0.1fs.fig', baseName, winDur)));
    exportgraphics(figAN, fullfile(plotDir, sprintf('%s_overlay_AN_dP_%0.1fs.jpg', baseName, winDur)), 'Resolution', 150);
end

if ~isempty(plotHA)
    figHA = mkFig('ΔPupil after stim — HA trials');
    hold on; box on; grid on;
    if ~isempty(yCandidatesHA)
        yl = [prctile(yCandidatesHA,2) prctile(yCandidatesHA,98)];
        if all(isfinite(yl)), ylim(yl); end
    end
    for k = plotHA(:)'
        wmask = (t >= stim_s(k)) & (t <= stimEnd(k)); if ~any(wmask), continue; end
        rt = t(wmask) - stim_s(k);
        dp = pupilRaw(wmask) - perTrialBaseline(k);
        if hasCong && isCong(k)
            plot(rt, dp, '-', 'LineWidth', 1.2, 'Color', [0 0.4470 0.7410 0.35]); % congruent
        else
            plot(rt, dp, '-', 'LineWidth', 1.2, 'Color', [0.8500 0.3250 0.0980 0.35]); % incongruent/other
        end
    end
    yline(0, ':', 'Δ from 200 ms pre-stim', 'Color',[0.3 0.3 0.3], 'LineWidth', 1);
    xlabel('Time since stim (s)'); ylabel('ΔPupil (baseline-subtracted)');
    title(sprintf('ΔPupil %0.1fs after stim — HA (ID %s)', winDur, participantID), 'Interpreter','none');
    savefig(figHA, fullfile(plotDir, sprintf('%s_overlay_HA_dP_%0.1fs.fig', baseName, winDur)));
    exportgraphics(figHA, fullfile(plotDir, sprintf('%s_overlay_HA_dP_%0.1fs.jpg', baseName, winDur)), 'Resolution', 150);
end

% -------------------------------------------------------------------------
% GROUPED AVERAGES OF PEAK ΔPUPIL (no smoothing)
% -------------------------------------------------------------------------
meanOrNaN = @(x) mean(x, 'omitnan');

avCongruent     = meanOrNaN(peakDelta(isCong));
avIncongruent   = meanOrNaN(peakDelta(isIncong));
angryAvg        = meanOrNaN(peakDelta(isAN));
happyAvg        = meanOrNaN(peakDelta(isHA));
ANincongAvg     = meanOrNaN(peakDelta(isAN & isIncong));
HAincongAvg     = meanOrNaN(peakDelta(isHA & isIncong));

Ns = struct( ...
    'congruent',     nnz(isCong & ~isnan(peakDelta)), ...
    'incongruent',   nnz(isIncong & ~isnan(peakDelta)), ...
    'angry',         nnz(isAN & ~isnan(peakDelta)), ...
    'happy',         nnz(isHA & ~isnan(peakDelta)), ...
    'ANincongruent', nnz(isAN & isIncong & ~isnan(peakDelta)), ...
    'HAincongruent', nnz(isHA & isIncong & ~isnan(peakDelta)) );

% -------------------------------------------------------------------------
% BUILD RETURN STRUCT
% -------------------------------------------------------------------------
pupil = struct();
pupil.baseline = struct('perTrial', perTrialBaseline, ...
                        'global',   globalBaseline, ...
                        'mean',     meanOrNaN(perTrialBaseline));
pupil.peak     = struct('perTrial', peakDelta, ...
                        'timePerTrial', peakTime, ...
                        'mean',     meanOrNaN(peakDelta));   % <- average of peaks across all trials
pupil.avCongruent   = avCongruent;
pupil.avIncongruent = avIncongruent;
pupil.angry         = angryAvg;
pupil.happy         = happyAvg;
pupil.ANincongruent = ANincongAvg;
pupil.HAincongruent = HAincongAvg;
pupil.Ns            = Ns;
pupil.trials        = trialTbl;

% Console summary
fprintf('Global baseline (10 s pre-Trial1): %.4f | Mean per-trial baseline (200 ms): %.4f\n', ...
    pupil.baseline.global, pupil.baseline.mean);
fprintf('Mean peak Δpupil (all trials): %.4f | Cong=%.4f (n=%d) | Incong=%.4f (n=%d) | AN=%.4f (n=%d) | HA=%.4f (n=%d) | AN&Incong=%.4f (n=%d) | HA&Incong=%.4f (n=%d)\n', ...
    pupil.peak.mean, pupil.avCongruent, Ns.congruent, pupil.avIncongruent, Ns.incongruent, ...
    pupil.angry, Ns.angry, pupil.happy, Ns.happy, pupil.ANincongruent, Ns.ANincongruent, pupil.HAincongruent, Ns.HAincongruent);

end