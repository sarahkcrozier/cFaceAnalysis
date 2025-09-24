function HR = cFaceHR_originalHRprocessing(participantID)
% cFaceHR_oldMatch  Reproduce the OLD rolling-HR-based trial means.
% - Savitzky–Golay smoothing for PPG
% - Peak detection with MinPeakDistance = 56 (samples)
% - NO PPI cleaning
% - Instantaneous HR from IBIs, then 5-beat movmean (beat domain)
% - Linear interpolation (NaN outside) to time base
% - Trial means: average HR at peakTimes in window; fallback to interp
%
% Outputs match the old fields:
%   HR.baseline, HR.postTask, HR.incongruentAverage, HR.congruentAverage
%   HR.baselinePeakPPG, HR.incongruentPeakPPG, HR.congruentPeakPPG

options  = specifyOptions();
IDstring = sprintf('%03d', participantID);

% -------- Default outputs (for early returns) --------
HR.baseline             = NaN;
HR.postTask             = NaN;
HR.incongruentAverage   = NaN;
HR.congruentAverage     = NaN;
HR.baselinePeakPPG      = NaN;
HR.incongruentPeakPPG   = NaN;
HR.congruentPeakPPG     = NaN;
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
    opts.SelectedVariableNames = {'instruct_onset','instruct_duration', ...
                                  'postinstruct_onset','fixation_onset','fixation_duration'};
    segmentsTable = readtable(summaryFilePath, opts);

    % Baseline timing (first instruct onset, with guard for 0) — matches old logic
    if segmentsTable.instruct_onset(1) == 0
        if segmentsTable.postinstruct_onset(1) == 0
            baselineTiming = 10; % fallback
        else
            baselineTiming = segmentsTable.postinstruct_onset(1) - segmentsTable.instruct_duration(1);
        end
    else
        baselineTiming = segmentsTable.instruct_onset(1);
    end

    % Post-task window = last 10 s of the task (old code used last fixation row)
    lastIdx = height(segmentsTable);
    postTaskTimingStart = (segmentsTable.fixation_onset(lastIdx) + segmentsTable.fixation_duration(lastIdx)) - 10;
    postTaskTimingEnd   =  segmentsTable.fixation_onset(lastIdx) + segmentsTable.fixation_duration(lastIdx);

    %% ---- STEP 2: Read PPG, Savitzky–Golay smoothing (OLD), baseline/post-task HR ----
    ppgFilePattern = fullfile(folderPath, 'beh', 'cface*MH*', '*ppg.csv');
    ppgFile = dir(ppgFilePattern);
    if isempty(ppgFile)
        warning('No PPG file found for participant %s. Skipping PPG calculation.', IDstring);
        return
    end

    ppgFilePath = fullfile(ppgFile.folder, ppgFile.name);
    ppgOpts = detectImportOptions(ppgFilePath);
    ppgOpts.SelectedVariableNames = {'time','PPG'};
    ppgTable = readtable(ppgFilePath, ppgOpts);

    time_raw = ppgTable.time(:);
    PPG_raw  = ppgTable.PPG(:);

    % Old pipeline: Savitzky–Golay smoothing
    ppgData = smoothdata(PPG_raw, 'sgolay');

    % ---- Baseline HR via peak count on smoothed signal (OLD) ----
    baselineRows = time_raw <= baselineTiming;
    if any(baselineRows)
        ppgWindow = ppgData(baselineRows);

        % OLD: MinPeakDistance = 56 (samples)
        [peaksB, ~] = findpeaks(ppgWindow, 'MinPeakDistance', 56);
        peakCount   = numel(peaksB);
        HR.baseline = (peakCount / baselineTiming) * 60;

        % OLD: baseline max PPG amplitude within baseline window (smoothed)
        HR.baselinePeakPPG = max(ppgWindow, [], 'omitnan');

        % ---- Post-task HR (OLD style, last 10 s) ----
        postTaskRows = (time_raw >= postTaskTimingStart) & (time_raw <= postTaskTimingEnd);
        if any(postTaskRows)
            ppgWindowPost = ppgData(postTaskRows);
            [peaksP, ~]   = findpeaks(ppgWindowPost, 'MinPeakDistance', 56);
            windowDuration = postTaskTimingEnd - postTaskTimingStart;
            HR.postTask    = (numel(peaksP) / windowDuration) * 60;
        end
    else
        warning('No PPG data before baseline timing for participant %s.', IDstring);
    end

    %% ---- STEP 3: Incongruent / Congruent rolling-HR means (OLD logic) ----
    segmentsTableIncongruent = cFaceIncongruentTrials(participantID);
    segmentsTableCongruent   = cFaceCongruentTrials(participantID);

    % Deduplicate time (OLD did this before rolling HR)
    ppgMatrix = [time_raw, ppgData];
    [~, uniqueIdx] = unique(ppgMatrix(:,1), 'stable');
    ppgMatrix = ppgMatrix(uniqueIdx, :);
    time      = ppgMatrix(:,1);
    ppgDataU  = ppgMatrix(:,2);

    % Peak detection for rolling-HR path (OLD settings, no cleaning)
    [~, peakLocs] = findpeaks(ppgDataU, 'MinPeakDistance', 56);
    peakTimes = time(peakLocs);

    % Instantaneous HR from IBIs, align with peakTimes (OLD)
    IBIs      = diff(peakTimes);
    IBIs      = [NaN; IBIs];                 % align to peaks
    HR_inst   = 60 ./ IBIs;

    % 5-beat rolling mean in the beat domain (OLD)
    HR_rolling = movmean(HR_inst, 5, 'SamplePoints', peakTimes);

    % Linear interpolation to time base; NaN outside (OLD)
    HR_interp  = interp1(peakTimes, HR_rolling, time, 'linear', NaN);

    % ---------- Incongruent windows ----------
    onsets   = segmentsTableIncongruent.stimMove_onset;
    windows  = segmentsTableIncongruent.responseWindow;
    numTrials = height(segmentsTableIncongruent);
    meanHRs   = NaN(numTrials,1);

    % Also compute per-trial max smoothed PPG (OLD behavior)
    incMaxPPG = NaN(numTrials,1);

    for i = 1:numTrials
        t0 = onsets(i);
        t1 = t0 + windows(i);

        % Primary (OLD): mean rolling HR at peaks inside window
        inWinPeaks = (HRTableTimeSafe(peakTimes) >= t0) & (HRTableTimeSafe(peakTimes) <= t1);
        if any(inWinPeaks)
            meanHRs(i) = mean(HR_rolling(inWinPeaks), 'omitnan');
        else
            % Fallback (OLD): mean of interpolated values on the time grid
            inWinInterp = (time >= t0) & (time <= t1);
            meanHRs(i) = mean(HR_interp(inWinInterp), 'omitnan');
        end

        % Max smoothed PPG within the window (OLD)
        inWindowPPG = (time >= t0) & (time <= t1);
        if any(inWindowPPG)
            incMaxPPG(i) = max(ppgDataU(inWindowPPG), [], 'omitnan');
        end
    end

    % ---------- Congruent windows ----------
    onsetsC   = segmentsTableCongruent.stimMove_onset;
    windowsC  = segmentsTableCongruent.responseWindow;
    numTrialsC = height(segmentsTableCongruent);
    meanHRC    = NaN(numTrialsC,1);
    congMaxPPG = NaN(numTrialsC,1);

    for i = 1:numTrialsC
        t0 = onsetsC(i);
        t1 = t0 + windowsC(i);

        inWinPeaks = (HRTableTimeSafe(peakTimes) >= t0) & (HRTableTimeSafe(peakTimes) <= t1);
        if any(inWinPeaks)
            meanHRC(i) = mean(HR_rolling(inWinPeaks), 'omitnan');
        else
            inWinInterp = (time >= t0) & (time <= t1);
            meanHRC(i) = mean(HR_interp(inWinInterp), 'omitnan');
        end

        inWindowPPG = (time >= t0) & (time <= t1);
        if any(inWindowPPG)
            congMaxPPG(i) = max(ppgDataU(inWindowPPG), [], 'omitnan');
        end
    end

    % ---------- Condition summaries (OLD fields) ----------
    HR.incongruentAverage = mean(meanHRs,  'omitnan');
    HR.congruentAverage   = mean(meanHRC,  'omitnan');

    HR.incongruentPeakPPG = mean(incMaxPPG,  'omitnan');
    HR.congruentPeakPPG   = mean(congMaxPPG, 'omitnan');

    %% ---- Plot (optional; mirrors the old look) ----
    if ~exist(options.paths.plots, 'dir'), mkdir(options.paths.plots); end

    fig = figure('Color','w'); hold on; box on;
    yl = [min(HR_interp,[],'omitnan') max(HR_interp,[],'omitnan')];
    if ~all(isfinite(yl)) || yl(1) == yl(2), yl = ylim; end
    yfill = [yl(1) yl(1) yl(2) yl(2)];

    % Incongruent windows (light red)
    if ~isempty(onsets) && ~isempty(windows)
        for i = 1:numel(onsets)
            x = [onsets(i), onsets(i) + windows(i)];
            patch('XData',[x(1) x(2) x(2) x(1)], 'YData',yfill, ...
                  'FaceColor',[0.85 0.33 0.10], 'FaceAlpha',0.13, 'EdgeColor','none', ...
                  'DisplayName','Incongruent window');
        end
    end
    % Congruent windows (light teal)
    if ~isempty(onsetsC) && ~isempty(windowsC)
        for i = 1:numel(onsetsC)
            x = [onsetsC(i), onsetsC(i) + windowsC(i)];
            patch('XData',[x(1) x(2) x(2) x(1)], 'YData',yfill, ...
                  'FaceColor',[0.10 0.60 0.80], 'FaceAlpha',0.13, 'EdgeColor','none', ...
                  'DisplayName','Congruent window');
        end
    end

    % Rolling HR line (OLD)
    hHR = plot(time, HR_interp, 'b-', 'LineWidth',1.2, 'DisplayName','Rolling HR (5-beat mean)');
    xlabel('Time (s)'); ylabel('Rolling HR (bpm)');
    title(sprintf('Rolling HR with Congruent & Incongruent Windows — ID %s', IDstring), 'Interpreter','none');

    [~, uniqIdx] = unique(string(get(findobj(gca,'-property','DisplayName'),'DisplayName')));
    legendObjs = findobj(gca,'-property','DisplayName');
    legend(legendObjs(uniqIdx), 'Location','best');
    uistack(hHR,'top');

    % Save figure
    filename = sprintf('cFacePlot_%s.fig', IDstring);
    saveas(fig, fullfile(options.paths.plots, filename));
    close(fig);

    %% Report (for quick checking)
    fprintf('Baseline timing for %s: %.3f seconds\n', IDstring, baselineTiming);
    fprintf('PostTaskHR (bpm) for %s: %.3f\n', IDstring, HR.postTask);
    fprintf('Mean HR baseline (bpm): %.3f\n', HR.baseline);
    fprintf('Mean HR in incongruent windows (bpm): %.3f\n', HR.incongruentAverage);
    fprintf('Mean HR in congruent windows (bpm): %.3f\n', HR.congruentAverage);
    fprintf('BaselinePeakPPG: %.3f | IncongruentPeakPPG: %.3f | CongruentPeakPPG: %.3f\n', ...
        HR.baselinePeakPPG, HR.incongruentPeakPPG, HR.congruentPeakPPG);

catch ME
    warning('cFaceHR_oldMatch(%s) failed: %s', IDstring, ME.message);
    rethrow(ME)
finally
    diary off
end
end

% --- tiny helper to keep code robust if peakTimes are empty (as in old code)
function t = HRTableTimeSafe(peakTimes)
    if isempty(peakTimes), t = NaN(0,1); else, t = peakTimes; end
end