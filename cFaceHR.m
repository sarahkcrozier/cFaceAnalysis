function HR = cFaceHR(participantID)
% cFaceHR  Build HR-based dataset and plots from PPG for one participant.
% Uses robust PPG preprocessing, adaptive peak detection, PPI editing,
% beat-domain HR for trial windows, and a smoothed HR line for plotting.

options  = specifyOptions();
IDstring = sprintf('%03d', participantID);

% -------- Default outputs (for early returns) --------
HR.baseline             = NaN;
HR.postTask             = NaN;
HR.incongruentAverage   = NaN;
HR.congruentAverage     = NaN;
HR.baselineHighpassedPeakPPG      = NaN;
HR.incongruentHighpassedPeakPPG   = NaN;
HR.congruentHighpassedPeakPPG     = NaN;
% -----------------------------------------------------

logPath = fullfile(options.paths.analysis, 'output_HR_cface.txt');
diary(logPath)
try
    %% Locate participant folder
    participantFolders = dir(options.paths.data);
    folderNames = {participantFolders.name};
    folderNames = folderNames(~ismember(folderNames, {'.','..'}));
    matchIdx = contains(folderNames, IDstring);
    if ~any(matchIdx)
        warning('No folder matched for participant %s.', IDstring);
        return
    end
    if sum(matchIdx) > 1
        warning('Multiple folders matched for ID %s. Using the first.', IDstring);
    end
    folderName = folderNames{find(matchIdx, 1)};
    folderPath = fullfile(options.paths.data, folderName);
    disp(['Using folder: ', folderName]);

    %% ---- STEP 1: Locate timings (baseline + post-task) ----
    filePattern   = fullfile(folderPath, 'beh', 'cface*MH*', '*out.csv');
    summaryFile   = dir(filePattern);
    if isempty(summaryFile)
        warning('No *out.csv found for %s.', IDstring);
        return
    end
    summaryFilePath = fullfile(summaryFile.folder, summaryFile.name);

    opts = detectImportOptions(summaryFilePath);
    % these columns must exist in the CSV
    opts.SelectedVariableNames = {'instruct_onset','instruct_duration', ...
                                  'postinstruct_onset','fixation_onset','fixation_duration'};
    segmentsTable = readtable(summaryFilePath, opts);
    
    % Add sequential trial numbers based on row order (1..N)
    if ~ismember('trialNo', segmentsTable.Properties.VariableNames)
        segmentsTable.trialNo = (1:height(segmentsTable)).';
    end

    % Baseline timing (first instruct onset, with guard for 0)
    if segmentsTable.instruct_onset(1) == 0
        if segmentsTable.postinstruct_onset(1) == 0
            baselineTiming = 10; % fallback
        else
            baselineTiming = segmentsTable.postinstruct_onset(1) - segmentsTable.instruct_duration(1);
        end
    else
        baselineTiming = segmentsTable.instruct_onset(1);
    end

    % Post-task: last 10 s of the block, based on the final fixation row
    lastIdx = height(segmentsTable);
    postTaskEnd   = segmentsTable.fixation_onset(lastIdx) + segmentsTable.fixation_duration(lastIdx);
    postTaskStart = max(0, postTaskEnd - 10);
    postTaskTimingStart = postTaskStart;
    postTaskTimingEnd   = postTaskEnd;



%% ---- STEP 2 (inline): read PPG, high-pass, detect & clean beats ----
ppgFilePattern = fullfile(folderPath, 'beh', 'cface*MH*', '*ppg.csv');
ppgFile = dir(ppgFilePattern);
if isempty(ppgFile)
    warning('No PPG file found for participant %s.', IDstring);
    return
end

ppgFilePath = fullfile(ppgFile.folder, ppgFile.name);
ppgOpts = detectImportOptions(ppgFilePath);
ppgOpts.SelectedVariableNames = {'time','PPG'};
ppgTable = readtable(ppgFilePath, ppgOpts);

time   = ppgTable.time(:);
signal = ppgTable.PPG(:);

% --- (1) sampling rate from time vector
dt = median(diff(time), 'omitnan');
if ~isfinite(dt) || dt<=0
    error('Bad time vector in %s', ppgFilePath);
end
Fs = 1/dt;

% --- (2) zero-phase high-pass at 0.01 Hz (no low-pass, no interpolation)
Wn = 0.01 / (Fs/2);                 % normalized cutoff
Wn = max(min(Wn, 0.99), 1e-6);      % guard against extreme Fs
[b,a] = butter(3, Wn, 'high');
highpassedSignal = filtfilt(b, a, signal);

% --- (3) fixed-distance peak detection (assumes upright systolic peaks)
minPeakDistanceSamples = 56;        % hard-coded, as requested. Alternative: minPeakDistanceSamples = max(1, round(0.30 * Fs));  % instead of 56
[~, peakIdx] = findpeaks(highpassedSignal, ...
    'MinPeakDistance', minPeakDistanceSamples);

peakTimes = time(peakIdx);

% --- (4) artifact handling in interval domain (keeps only clean peaks)
[peakTimes, ~, ~] = cleanPPI(peakTimes);  % cleaner is under subfuctions below

% ---- Rolling HR for 5 s windows (for trial means) ----
rollingWindowSec = 5.0;                           % <<< CHANGE ME if needed
HRrolling5 = buildHRLine(peakTimes, time, rollingWindowSec);  % bpm, time-aligned


% ---- Baseline HR (from clean peaks) ----
inBaseline = peakTimes <= baselineTiming;   % Select peaks that occur before the baseline cutoff
if nnz(inBaseline) >= 2                     % Require at least 2 beats
    tb = peakTimes(inBaseline);
    baselineDuration = tb(end) - tb(1);
    if baselineDuration > 0
        HR.baseline = ((numel(tb)-1) / baselineDuration) * 60;  % Compute baseline HR from beat INTERVALS (ie numel(tb)-1) and convert beats per second to bpm
        ib = (time >= tb(1)) & (time <= tb(end));
        HR.baselineHighpassedPeakPPG = max(highpassedSignal(ib),[],'omitnan');    % PEAK PPG - baseline - Compute baseline PPG amplitude
    end
else
    warning('No clean peaks before baseline for %s.', IDstring);
end


    % ---- Post-task HR from clean peaks ----
    inPost = (peakTimes >= postTaskTimingStart) & (peakTimes <= postTaskTimingEnd);
    if nnz(inPost) >= 2
        tp = peakTimes(inPost); % Select peaks that fall inside the window
        postDuration = tp(end) - tp(1); % Compute the duration covered by those peaks
        if postDuration > 0
            HR.postTask = ((numel(tp)-1) / postDuration) * 60; % Count the number of INTERVALS inside the window (if beats only, change to numel(tp)) and convert beats per second to bpm
        end
    end

%% ---- STEP 3: Trial-window HR in beat domain + per-trial max PPG ----
segmentsTableIncongruent = cFaceIncongruentTrials(participantID);   % Get incongruent trial windows 
segmentsTableCongruent   = cFaceCongruentTrials(participantID);     % Get congruent trial windows 

% Helper: mean HR (bpm) from clean peaks in [t0, t1]
% function found in SUBFUNCTIONS below
meanHRinWindow = @(t0,t1) localMeanHRfromPeaks(peakTimes, t0, t1);   % (kept for future use)

% Incongruent trials:
incongruentHRs    = NaN(height(segmentsTableIncongruent),1);
incongruentMaxPPG = NaN(height(segmentsTableIncongruent),1);
for i = 1:height(segmentsTableIncongruent)
    t0 = segmentsTableIncongruent.stimMove_onset(i);
    t1 = t0 + segmentsTableIncongruent.responseWindow(i);

    % NEW: mean of rolling 5 s HR within the trial window (bpm)
    idx = (time >= t0) & (time <= t1);
    %% try max HR for each trial window, replace with line below: 
    % incongruentHRs(i) = max(HRrolling5(idx), [], 'omitnan');   % per-trial maximum HR
    incongruentHRs(i) = mean(HRrolling5(idx), 'omitnan');            % trial-level HR (bpm) for that incongruent trial.

    % PPG amplitude (unchanged)
    if any(idx)
        incongruentMaxPPG(i) = max(highpassedSignal(idx),[],'omitnan'); % the maximum filtered PPG amplitude within that trial's window
    end
end

% Congruent trials
congruentHRs    = NaN(height(segmentsTableCongruent),1);
congruentMaxPPG = NaN(height(segmentsTableCongruent),1);
for i = 1:height(segmentsTableCongruent)
    t0 = segmentsTableCongruent.stimMove_onset(i);
    t1 = t0 + segmentsTableCongruent.responseWindow(i);

    % NEW: mean of rolling 5 s HR within the trial window (bpm)
    idx = (time >= t0) & (time <= t1);
    %% try max HR for each trial window, replace with line below: 
    % congruentHRs(i) = max(HRrolling5(idx), [], 'omitnan');   % per-trial maximum HR
    congruentHRs(i) = mean(HRrolling5(idx), 'omitnan');              % trial-level HR (bpm) for that congruent trial.

    % PPG amplitude (unchanged)
    if any(idx)
        congruentMaxPPG(i) = max(highpassedSignal(idx),[],'omitnan'); % the maximum filtered PPG amplitude within that trial's window
    end
end

%% FINAL CONDITION SUMMARIES: (unchanged field names)
kme = mean(incongruentHRs, 'omitnan');    % bpm, from rolling-5s HR
HR.congruentAverage   = mean(congruentHRs,   'omitnan');    % bpm, from rolling-5s HR
HR.incongruentHighpassedPeakPPG = mean(incongruentMaxPPG, 'omitnan'); % a.u.
HR.congruentHighpassedPeakPPG   = mean(congruentMaxPPG,   'omitnan'); % a.u.


%% ---- STEP 3b: 3-beat ΔPPI metrics per trial (all in seconds) ----
% Uses cleaned peakTimes from earlier.

% Preallocate per-trial containers
nI = height(segmentsTableIncongruent);
nC = height(segmentsTableCongruent);

% Incongruent
pre3_meanPPI_s_inc      = NaN(nI,1);
post3_meanPPI_s_inc     = NaN(nI,1);
diffPPI_s_inc           = NaN(nI,1);
percentageDiffPPI_inc   = NaN(nI,1);
maxChange_s_inc         = NaN(nI,1);
nPreIntervals_inc       = NaN(nI,1);
nPostIntervals_inc      = NaN(nI,1);
onset_inc               = NaN(nI,1);

for i = 1:nI
    t0 = segmentsTableIncongruent.stimMove_onset(i);
    [pre3, post3, dPPI_s, pct, maxchg_s, npre, npost] = deltaPPI3beat_seconds(peakTimes, t0);
    pre3_meanPPI_s_inc(i)    = pre3;
    post3_meanPPI_s_inc(i)   = post3;
    diffPPI_s_inc(i)         = dPPI_s;
    percentageDiffPPI_inc(i) = pct;
    maxChange_s_inc(i)       = maxchg_s;
    nPreIntervals_inc(i)     = npre;
    nPostIntervals_inc(i)    = npost;
    onset_inc(i)             = t0;
end

% Congruent
pre3_meanPPI_s_con      = NaN(nC,1);
post3_meanPPI_s_con     = NaN(nC,1);
diffPPI_s_con           = NaN(nC,1);
percentageDiffPPI_con   = NaN(nC,1);
maxChange_s_con         = NaN(nC,1);
nPreIntervals_con       = NaN(nC,1);
nPostIntervals_con      = NaN(nC,1);
onset_con               = NaN(nC,1);

for i = 1:nC
    t0 = segmentsTableCongruent.stimMove_onset(i);
    [pre3, post3, dPPI_s, pct, maxchg_s, npre, npost] = deltaPPI3beat_seconds(peakTimes, t0);
    pre3_meanPPI_s_con(i)    = pre3;
    post3_meanPPI_s_con(i)   = post3;
    diffPPI_s_con(i)         = dPPI_s;
    percentageDiffPPI_con(i) = pct;
    maxChange_s_con(i)       = maxchg_s;
    nPreIntervals_con(i)     = npre;
    nPostIntervals_con(i)    = npost;
    onset_con(i)             = t0;
end

% ---------- Build a single TRIAL-LEVEL table (no nested structs) ----------
participantCol_inc = repmat({IDstring}, nI, 1);
participantCol_con = repmat({IDstring}, nC, 1);

cond_inc = repmat({'incongruent'}, nI, 1);
cond_con = repmat({'congruent'},   nC, 1);

trialNo_inc = segmentsTableIncongruent.trialNo;
trialNo_con = segmentsTableCongruent.trialNo;

T_inc = table( ...
    participantCol_inc, cond_inc, trialNo_inc, onset_inc, ...
    pre3_meanPPI_s_inc, post3_meanPPI_s_inc, diffPPI_s_inc, ...
    percentageDiffPPI_inc, maxChange_s_inc, ...
    nPreIntervals_inc, nPostIntervals_inc, ...
    'VariableNames', {'participantID','condition','trialNumber','onsetTime_s', ...
                      'pre3_meanPPI_s','post3_meanPPI_s','diffPPI_s', ...
                      'percentageDiffPPI','maxChange_s','nPre','nPost'});

T_con = table( ...
    participantCol_con, cond_con, trialNo_con, onset_con, ...
    pre3_meanPPI_s_con, post3_meanPPI_s_con, diffPPI_s_con, ...
    percentageDiffPPI_con, maxChange_s_con, ...
    nPreIntervals_con, nPostIntervals_con, ...
    'VariableNames', {'participantID','condition','trialNumber','onsetTime_s', ...
                      'pre3_meanPPI_s','post3_meanPPI_s','diffPPI_s', ...
                      'percentageDiffPPI','maxChange_s','nPre','nPost'});

ppiTrialsTbl = [T_inc; T_con];   % one row per trial, with condition + participantID

% (Optional) keep in HR as a table (not nested structs)
HR.PPImetrics_trials = ppiTrialsTbl;

% ---------- Condition-level summaries (means across trials; not saved) ----------
summary_cond   = {'incongruent'; 'congruent'; 'allTrials'};
summary_id     = repmat({IDstring}, 3, 1);
all_mask       = true(height(ppiTrialsTbl),1);

mu = @(x,mask) mean(x(mask), 'omitnan');

mu_inc = strcmp(ppiTrialsTbl.condition,'incongruent');
mu_con = strcmp(ppiTrialsTbl.condition,'congruent');

summaryTbl = table( ...
    summary_id, summary_cond, ...
    [mu(ppiTrialsTbl.pre3_meanPPI_s, mu_inc); mu(ppiTrialsTbl.pre3_meanPPI_s, mu_con); mu(ppiTrialsTbl.pre3_meanPPI_s, all_mask)], ...
    [mu(ppiTrialsTbl.post3_meanPPI_s, mu_inc); mu(ppiTrialsTbl.post3_meanPPI_s, mu_con); mu(ppiTrialsTbl.post3_meanPPI_s, all_mask)], ...
    [mu(ppiTrialsTbl.diffPPI_s,      mu_inc); mu(ppiTrialsTbl.diffPPI_s,      mu_con); mu(ppiTrialsTbl.diffPPI_s,      all_mask)], ...
    [mu(ppiTrialsTbl.percentageDiffPPI, mu_inc); mu(ppiTrialsTbl.percentageDiffPPI, mu_con); mu(ppiTrialsTbl.percentageDiffPPI, all_mask)], ...
    [mu(ppiTrialsTbl.maxChange_s,    mu_inc); mu(ppiTrialsTbl.maxChange_s,    mu_con); mu(ppiTrialsTbl.maxChange_s,    all_mask)], ...
    'VariableNames', {'participantID','condition', ...
                      'pre3_meanPPI_s','post3_meanPPI_s','diffPPI_s', ...
                      'percentageDiffPPI','maxChange_s'});
HR.PPImetrics_summary = summaryTbl;


% ---- Copy condition-level means into flat HR fields for reuse (all seconds / %) ----
rowInc = strcmp(HR.PPImetrics_summary.condition,'incongruent');
rowCon = strcmp(HR.PPImetrics_summary.condition,'congruent');
rowAll = strcmp(HR.PPImetrics_summary.condition,'allTrials');

% Averages of diffPPI_s (s)
HR.IncongruentDiffPPI = HR.PPImetrics_summary.diffPPI_s(rowInc);   % seconds
HR.CongruentDiffPPI   = HR.PPImetrics_summary.diffPPI_s(rowCon);   % seconds
HR.AllTrialsDiffPPI   = HR.PPImetrics_summary.diffPPI_s(rowAll);   % seconds

% Averages of percentageDiffPPI (%)
HR.IncongruentPercentageDiffPPI = HR.PPImetrics_summary.percentageDiffPPI(rowInc); % percent
HR.CongruentPercentageDiffPPI   = HR.PPImetrics_summary.percentageDiffPPI(rowCon); % percent
HR.AllTrialsPercentageDiffPPI   = HR.PPImetrics_summary.percentageDiffPPI(rowAll); % percent

% Averages of maxChange_s (s)
HR.IncongruentMaxChangePPI = HR.PPImetrics_summary.maxChange_s(rowInc);  % seconds
HR.CongruentMaxChangePPI   = HR.PPImetrics_summary.maxChange_s(rowCon);  % seconds
HR.AllTrialsMaxChangePPI   = HR.PPImetrics_summary.maxChange_s(rowAll);  % seconds


% ---------- Write TRIAL-LEVEL CSV (one file per participant) ----------
if ~exist(options.paths.HRdata, 'dir'), mkdir(options.paths.HRdata); end
outFile = fullfile(options.paths.HRdata, sprintf('%s_PPImetrics.csv', IDstring));
writetable(ppiTrialsTbl, outFile);   % no rounding


    %% ---- Plot with shaded windows (HRline vs time) ----
    HRline = HRrolling5;                  % use the same 5-s rolling HR used for trial means
plotWindowSec = rollingWindowSec;     % optional: carry the window value into the legend/title

    % make sure plot folder exists
    if ~exist(options.paths.plots, 'dir'), mkdir(options.paths.plots); end

    fig = figure('Color','w'); hold on; box on;
    xlabel('Time (s)'); ylabel('HR (bpm)');
    title(sprintf('Rolling 5-second HR (cleaned) with Congruent & Incongruent Windows — ID %s', IDstring), ...
          'Interpreter','none');

    yl = [min(HRline,[],'omitnan') max(HRline,[],'omitnan')];
    if ~all(isfinite(yl)) || yl(1)==yl(2), yl = [40 140]; end
    ylim(yl);
    yfill = [yl(1) yl(1) yl(2) yl(2)];

    % Incongruent windows
    if ~isempty(segmentsTableIncongruent) && height(segmentsTableIncongruent)>0
        for i = 1:height(segmentsTableIncongruent)
            x0 = segmentsTableIncongruent.stimMove_onset(i);
            x1 = x0 + segmentsTableIncongruent.responseWindow(i);
            patch('XData',[x0 x1 x1 x0], 'YData',yfill, ...
                  'FaceColor',[0.85 0.33 0.10], 'FaceAlpha',0.13, ...
                  'EdgeColor','none', 'DisplayName','Incongruent window');
        end
    end

    % Congruent windows
    if ~isempty(segmentsTableCongruent) && height(segmentsTableCongruent)>0
        for i = 1:height(segmentsTableCongruent)
            x0 = segmentsTableCongruent.stimMove_onset(i);
            x1 = x0 + segmentsTableCongruent.responseWindow(i);
            patch('XData',[x0 x1 x1 x0], 'YData',yfill, ...
                  'FaceColor',[0.10 0.60 0.80], 'FaceAlpha',0.13, ...
                  'EdgeColor','none', 'DisplayName','Congruent window');
        end
    end

    % HR line on top
    hHR = plot(time, HRline, 'b-', 'LineWidth', 1.2, ...
           'DisplayName', sprintf('HR (rolling %.1f s)', plotWindowSec));
    uistack(hHR,'top');

    % Legend
    legendObjects = findobj(gca,'-property','DisplayName');
    [~, uniqIdx] = unique(string(get(legendObjects,'DisplayName')));
    legend(legendObjects(uniqIdx), 'Location','best');

    % Save .fig
    filename = sprintf('cFacePlot_%s.fig', IDstring);
    saveas(fig, fullfile(options.paths.plots, filename));
    close(fig);

    %% Report
    fprintf('Baseline timing for %s: %.3f seconds\n', IDstring, baselineTiming);
    fprintf('PostTaskHR (bpm) for %s: %.3f\n', IDstring, HR.postTask);
    fprintf('Mean HR baseline (bpm): %.3f\n', HR.baseline);
    fprintf('Mean HR in incongruent windows (bpm): %.3f\n', HR.incongruentAverage);
    fprintf('Mean HR in congruent windows (bpm): %.3f\n', HR.congruentAverage);
    fprintf('BaselineHighpassedPeakPPG: %.3f | IncongruentHighpassedPeakPPG: %.3f | CongruentHighpassedPeakPPG: %.3f\n', ...
        HR.baselineHighpassedPeakPPG, HR.incongruentHighpassedPeakPPG, HR.congruentHighpassedPeakPPG);
   
    fprintf('ΔPPI (s): mean inc=%.4f, con=%.4f, all=%.4f | %%ΔPPI: mean inc=%.2f, con=%.2f, all=%.2f\n', ...
         HR.IncongruentDiffPPI, HR.CongruentDiffPPI, HR.AllTrialsDiffPPI, ...
         HR.IncongruentPercentageDiffPPI, HR.CongruentPercentageDiffPPI, HR.AllTrialsPercentageDiffPPI);

    fprintf('Max-change PPI (s): mean inc=%.4f, con=%.4f, all=%.4f\n', ...
         HR.IncongruentMaxChangePPI, HR.CongruentMaxChangePPI, HR.AllTrialsMaxChangePPI);

catch ME
    warning('cFaceHR(%s) failed: %s', IDstring, ME.message);
    rethrow(ME)  % comment this out if you prefer soft-fail
finally
    diary off
end
end  % ===== end main function =====


% ===================== SUBFUNCTIONS =====================

function [cleanPeakTimes, PPI, keepMask] = cleanPPI(peakTimes)
% Remove implausible/outlier pulse intervals (PPI).
    if numel(peakTimes) < 3
        cleanPeakTimes = peakTimes;
        PPI = diff(peakTimes);
        keepMask = true(size(peakTimes));
        return
    end

    PPIraw = diff(peakTimes);                         % seconds
    physOK = (PPIraw >= 0.30) & (PPIraw <= 2.00);     % 30–200 bpm

    ref = PPIraw(physOK);
    if isempty(ref) || all(~isfinite(ref))
        keepMask = false(size(peakTimes)); keepMask(1) = true;
        cleanPeakTimes = peakTimes(keepMask);
        PPI = diff(cleanPeakTimes);
        return
    end

    deviation = abs(PPIraw - median(ref,'omitnan')) / (1.4826*mad(ref,1));
    statOK = deviation <= 3.5;

    intervalOK    = physOK & statOK;
    keepMask      = [true; intervalOK];   % map interval mask to peaks
    cleanPeakTimes = peakTimes(keepMask);
    PPI = diff(cleanPeakTimes);
end

function HRline = buildHRLine(peakTimes, time, smoothingWindow)
% Make a smooth HR (bpm) line from cleaned peaks for plotting only.
% smoothingWindow (sec) controls moving-average length (default 1.0).
    if nargin < 3 || isempty(smoothingWindow), smoothingWindow = 1.0; end
    if numel(peakTimes) < 3
        HRline = nan(size(time)); return
    end

    PPI    = diff(peakTimes);
    instHR = 60 ./ PPI;                 % bpm per interval
    instHR = [instHR(1); instHR];       % align with peakTimes length

    % Convert seconds to ~samples via beats/sec estimate
    medianPPI = median(PPI,'omitnan'); % seconds per beat
    if ~isfinite(medianPPI) || medianPPI <= 0, medianPPI = 1.0; end
              
    beatsPerSec = 1./medianPPI;                      % ~samples per second (in the beat domain)
    winSamples  = max(1, round(smoothingWindow * beatsPerSec));
    HRsmooth    = movmean(instHR, winSamples, 'SamplePoints', peakTimes);

    HRline   = interp1(peakTimes, HRsmooth, time, 'pchip', 'extrap');
end


function v = localMeanHRfromPeaks(peakTimes, t0, t1)
% Mean HR (bpm) from clean peaks in [t0, t1].
    tt = peakTimes(peakTimes>=t0 & peakTimes<=t1);
    if numel(tt) < 2
        v = NaN;
    else
        v = ((numel(tt)-1) / (tt(end)-tt(1))) * 60;
    end
end

function [pre3, post3, dPPI_ms, pct_dPPI, maxChange_ms, nPre, nPost] = deltaPPI3beat(peakTimes, t0)
% deltaPPI3beat  Compute 3-beat pre/post PPI metrics around onset t0.
% Inputs:
%   peakTimes : cleaned beat times (s), ascending
%   t0        : onset time (s)
% Outputs:
%   pre3, post3       : mean PPI (s) over last up to 3 pre intervals and first up to 3 post intervals
%   dPPI_ms           : (post3 - pre3) in ms
%   pct_dPPI          : 100 * (post3 - pre3)/pre3  (NaN if pre3 is NaN or 0)
%   maxChange_ms      : max |PPI_post_i - pre3| within the first up to 3 post intervals, signed (ms)
%   nPre, nPost       : how many PPIs contributed to pre3 and post3 (0..3)

    if numel(peakTimes) < 2
        pre3 = NaN; post3 = NaN; dPPI_ms = NaN; pct_dPPI = NaN; maxChange_ms = NaN;
        nPre = 0; nPost = 0; return
    end

    % Intervals and their start/end times
    PPI      = diff(peakTimes);             % seconds
    tStart   = peakTimes(1:end-1);          % interval start time (s)
    tEnd     = peakTimes(2:end);            % interval end time (s)

    % PRE: intervals whose end time is <= t0 (completed before onset)
    preMask  = (tEnd <= t0);
    preIdx   = find(preMask);
    if isempty(preIdx)
        pre3 = NaN; nPre = 0;
    else
        take    = preIdx(max(1, numel(preIdx)-2) : numel(preIdx));   % last up to 3
        pre3    = mean(PPI(take), 'omitnan');
        nPre    = numel(take);
    end

    % POST: intervals whose start time is >= t0 (start after onset)
    postMask = (tStart >= t0);
    postIdx  = find(postMask);
    if isempty(postIdx)
        post3 = NaN; nPost = 0; dPPI_ms = NaN; pct_dPPI = NaN; maxChange_ms = NaN; return
    else
        take    = postIdx(1 : min(3, numel(postIdx)));               % first up to 3
        post3   = mean(PPI(take), 'omitnan');
        nPost   = numel(take);
    end

    % ΔPPI and %ΔPPI
    if isfinite(pre3) && isfinite(post3)
        dPPI_ms  = (post3 - pre3) * 1000.0;
        if pre3 ~= 0
            pct_dPPI = 100.0 * (post3 - pre3) / pre3;
        else
            pct_dPPI = NaN;
        end
    else
        dPPI_ms  = NaN;
        pct_dPPI = NaN;
    end

    % Max-change: pick the single post interval (among first up to 3) whose deviation from pre3 is largest in |.|, report signed ms
    if isfinite(pre3) && ~isempty(postIdx)
        candIdx = postIdx(1 : min(3, numel(postIdx)));
        diffs   = (PPI(candIdx) - pre3) * 1000.0;   % ms, signed
        [~, k]  = max(abs(diffs));
        maxChange_ms = diffs(k);
    else
        maxChange_ms = NaN;
    end
end


function [pre3_meanPPI_s, post3_meanPPI_s, diffPPI_s, percentageDiffPPI, maxChange_s, nPreIntervals, nPostIntervals] = deltaPPI3beat_seconds(peakTimes, t0)
% deltaPPI3beat_seconds
% Computes 3-beat pre/post PPI metrics around onset t0, all in seconds.
% PRE = last up to 3 intervals whose END time ≤ t0
% POST = first up to 3 intervals whose START time ≥ t0

    if numel(peakTimes) < 2
        pre3_meanPPI_s = NaN; post3_meanPPI_s = NaN;
        diffPPI_s = NaN; percentageDiffPPI = NaN; maxChange_s = NaN;
        nPreIntervals = 0; nPostIntervals = 0; 
        return
    end

    % Intervals and their start/end times
    PPI    = diff(peakTimes);        % seconds
    tStart = peakTimes(1:end-1);     % start of each interval
    tEnd   = peakTimes(2:end);       % end of each interval

    % ----- PRE (<= t0)
    preIdx = find(tEnd <= t0);
    if isempty(preIdx)
        pre3_meanPPI_s = NaN; nPreIntervals = 0;
    else
        take = preIdx(max(1, numel(preIdx)-2) : numel(preIdx));   % last up to 3
        pre3_meanPPI_s = mean(PPI(take), 'omitnan');
        nPreIntervals  = numel(take);
    end

    % ----- POST (≥ t0)
    postIdx = find(tStart >= t0);
    if isempty(postIdx)
        post3_meanPPI_s = NaN; nPostIntervals = 0;
        diffPPI_s = NaN; percentageDiffPPI = NaN; maxChange_s = NaN; 
        return
    else
        take = postIdx(1 : min(3, numel(postIdx)));               % first up to 3
        post3_meanPPI_s = mean(PPI(take), 'omitnan');
        nPostIntervals  = numel(take);
    end

    % ----- Differences
    if isfinite(pre3_meanPPI_s) && isfinite(post3_meanPPI_s)
        diffPPI_s = post3_meanPPI_s - pre3_meanPPI_s;
        if pre3_meanPPI_s ~= 0
            percentageDiffPPI = 100.0 * diffPPI_s / pre3_meanPPI_s;
        else
            percentageDiffPPI = NaN;
        end
    else
        diffPPI_s        = NaN;
        percentageDiffPPI = NaN;
    end

    % ----- Max-change (signed, seconds): largest |pre3 - PPI_post_i| among first up to 3 post intervals
    if isfinite(pre3_meanPPI_s) && ~isempty(postIdx)
        cand = postIdx(1 : min(3, numel(postIdx)));     % Get indices of the first up to 3 post-stimulus PPIs
        diffs = pre3_meanPPI_s - PPI(cand);  % Compute baseline − post for each interval (seconds)
        maxChange_s = max(diffs);
    else
        maxChange_s = NaN;
    end
    %% Below code  does the same thing as above BUT takes the largest change (could be decelaration).
    %% Takes the abs max, then puts the sign back on. Note, this original code minuses the post from the pre, so accelaration is negative
    % if isfinite(pre3_meanPPI_s) && ~isempty(postIdx)
    % cand  = postIdx(1 : min(3, numel(postIdx)));
    % diffs = PPI(cand) - pre3_meanPPI_s;   % post − pre (s), signed
    % [~, k] = max(abs(diffs));
    % maxChange_s = diffs(k);
    % else
    %    maxChange_s = NaN;
    % end
end
