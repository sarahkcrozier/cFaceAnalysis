function HR = cFaceHR(participantID)

% Extracts the first instruct_onset timing from cFace baseline for a given participant

options = specifyOptions;
IDstring = sprintf('%03d', participantID);

%% save text of Matlab session
diary(fullfile(options.paths.analysis,'output_HR_cface.txt'))

% ------------------ Default outputs (for early returns) ------------------
HR.baseline             = NaN;
HR.postTask             = NaN;
HR.incongruentAverage   = NaN;
HR.congruentAverage     = NaN;
HR.baselinePeakPPG      = NaN;  
HR.incongruentPeakPPG   = NaN;  
HR.congruentPeakPPG     = NaN;  
% ------------------------------------------------------------------------

%% Locate participant folder
participantFolders = dir(options.paths.data);
folderNames = {participantFolders.name};
folderNames = folderNames(~ismember(folderNames, {'.','..'}));  % skip system entries

% Find participant folder name and path
matchIdx = contains(folderNames, IDstring);  % logical index
if ~any(matchIdx)
    % keep NaNs as initialized
    return
end
if sum(matchIdx) > 1
    warning('Multiple folders matched for ID %s. Using the first.', IDstring);
end
folderName = folderNames{find(matchIdx, 1)};
folderPath = fullfile(options.paths.data, folderName);
disp(['Using folder: ', folderName]);

%% ---- STEP 1: Locate instruct_onset ----
filePattern = fullfile(folderPath, 'beh', 'cface*MH*', '*out.csv');
summaryFile = dir(filePattern);
if isempty(summaryFile)
    % keep NaNs as initialized
    return
end
summaryFilePath = fullfile(summaryFile.folder, summaryFile.name);

% Read timings from baseline and postTask windows from CSV
opts = detectImportOptions(summaryFilePath);
opts.SelectedVariableNames = {'instruct_onset','instruct_duration','postinstruct_onset','fixation_onset','fixation_duration'};
segmentsTable = readtable(summaryFilePath, opts);

if segmentsTable.instruct_onset(1) == 0 % some initial onset timings incorrectly 0
    if segmentsTable.postinstruct_onset(1) == 0
        baselineTiming = 10;
    else
        baselineTiming = (segmentsTable.postinstruct_onset(1) - segmentsTable.instruct_duration(1));
    end
else
    baselineTiming = segmentsTable.instruct_onset(1);  % first onset
end

% Define post-task window = last 10 s of task
postTaskTimingStart = (segmentsTable.fixation_onset(80) + segmentsTable.fixation_duration(1)) - 10;  
postTaskTimingEnd   =  segmentsTable.fixation_onset(80) + segmentsTable.fixation_duration(1);

%% ---- STEP 2: Locate PPG file, smooth, extract baseline ----
ppgFilePattern = fullfile(folderPath, 'beh', 'cface*MH*', '*ppg.csv');
ppgFile = dir(ppgFilePattern);
if isempty(ppgFile)
    warning('No PPG file found for participant %s. Skipping PPG calculation.', IDstring);
    % leave NaNs
    diary off
    return
end

ppgFilePath = fullfile(ppgFile.folder, ppgFile.name);
ppgOpts = detectImportOptions(ppgFilePath);
ppgOpts.SelectedVariableNames = {'time','PPG'};
ppgTable = readtable(ppgFilePath, ppgOpts);

% Smooth data (Savitzky–Golay)
ppgData = smoothdata(ppgTable.PPG,'sgolay');

%% ---- Extract baseline ----
baselineRows = ppgTable.time <= baselineTiming;
if any(baselineRows)
    % smoothed PPG in baseline window
    ppgWindow = ppgData;
    ppgWindow(~baselineRows,:) = [];

    % Baseline heart rate via peak count
    [peaks,locations,~,~] = findpeaks(ppgWindow,'MinPeakDistance',56); %#ok<ASGLU>
    fig = figure;
    findpeaks(ppgWindow, 'MinPeakDistance', 56);  % plot with peaks
    filename = sprintf('ppgBaselinePeaksPlot_%s.fig', IDstring);
    savePath = fullfile(options.paths.plots, filename);
    saveas(fig, savePath);
    close(fig);

    peakCount   = numel(peaks);
    HR.baseline = (peakCount / baselineTiming) * 60;

    % ADDED: max PPG amplitude within the baseline window
    if ~isempty(ppgWindow)
        HR.baselinePeakPPG = max(ppgWindow, [], 'omitnan');
    end

    %% extract postTask 10 sec HR
    postTaskRows = (ppgTable.time >= postTaskTimingStart) & (ppgTable.time <= postTaskTimingEnd);
    if any(postTaskRows)
        ppgWindow = ppgData;
        ppgWindow(~postTaskRows,:) = [];
        [peaks,locations,~,~] = findpeaks(ppgWindow,'MinPeakDistance',56); %#ok<ASGLU>

        fig = figure;
        findpeaks(ppgWindow, 'MinPeakDistance', 56);
        filename = sprintf('ppgPostTaskPeaksPlot_%s.fig', IDstring);
        savePath = fullfile(options.paths.plots, filename);
        saveas(fig, savePath);
        close(fig);

        windowDuration = postTaskTimingEnd - postTaskTimingStart;
        peakCount      = numel(peaks);
        HR.postTask    = (peakCount / windowDuration) * 60;
    end
else
    warning('No PPG data before baseline timing for participant %s.', IDstring);
end

%% Initialise segment tables
segmentsTableIncongruent = cFaceIncongruentTrials(participantID);
segmentsTableCongruent   = cFaceCongruentTrials(participantID);

%% ---- STEP 3: Compute rolling HR & means in INCONGRUENT windows ----
if length(ppgTable.time) ~= length(ppgData)
    error('Mismatch: time and PPG data must have the same length');
end

% Combine time & smoothed PPG, ensure unique times
ppgMatrix = [ppgTable.time, ppgData];
[~, uniqueIdx] = unique(ppgMatrix(:,1), 'stable');
ppgMatrix = ppgMatrix(uniqueIdx, :);

% Peaks → instantaneous HR → rolling HR
[peaks,locations,~,~] = findpeaks(ppgMatrix(:,2),'MinPeakDistance',56); %#ok<ASGLU>
peakTimes = ppgMatrix(locations, 1);
IBIs      = diff(peakTimes);
IBIs      = [NaN; IBIs];                       % align with peakTimes
HR_inst   = 60 ./ IBIs;
HR_rolling = movmean(HR_inst, 5, 'SamplePoints', peakTimes);
HR_interp  = interp1(peakTimes, HR_rolling, ppgTable.time, 'linear', NaN);

% Incongruent windows
onsets   = segmentsTableIncongruent.stimMove_onset;
windows  = segmentsTableIncongruent.responseWindow;
trialNo  = segmentsTableIncongruent.trialNo;
numTrials = height(segmentsTableIncongruent);
meanHRs   = NaN(numTrials,1);

% ADDED: per-trial max PPG (smoothed) within each incongruent window
incMaxPPG = NaN(numTrials,1);

for i = 1:numTrials
    startTime = onsets(i);
    endTime   = startTime + windows(i);

    % mean rolling HR within window (using peak times)
    inWindowHR = (HRTableTimeSafe(peakTimes) >= startTime) & (HRTableTimeSafe(peakTimes) <= endTime);
    if any(inWindowHR)
        meanHRs(i) = mean(HR_rolling(inWindowHR), 'omitnan');
    else
        % fall back to interpolate on full time base
        inWinInterp = (ppgTable.time >= startTime) & (ppgTable.time <= endTime);
        meanHRs(i) = mean(HR_interp(inWinInterp), 'omitnan');
    end

    % max PPG (smoothed) within window using full ppgMatrix time base
    inWindowPPG = (ppgMatrix(:,1) >= startTime) & (ppgMatrix(:,1) <= endTime);
    if any(inWindowPPG)
        incMaxPPG(i) = max(ppgMatrix(inWindowPPG, 2), [], 'omitnan');
    end
end

%% Calculate mean HR in windows after each CONGRUENT trial onset
onsetsC   = segmentsTableCongruent.stimMove_onset;
windowsC  = segmentsTableCongruent.responseWindow;
trialNoC  = segmentsTableCongruent.trialNo;
numTrialsC = height(segmentsTableCongruent);
meanHRC    = NaN(numTrialsC,1);

% ADDED: per-trial max PPG within each congruent window
congMaxPPG = NaN(numTrialsC,1);

for i = 1:numTrialsC
    startTime = onsetsC(i);
    endTime   = startTime + windowsC(i);

    inWindowHR = (HRTableTimeSafe(peakTimes) >= startTime) & (HRTableTimeSafe(peakTimes) <= endTime);
    if any(inWindowHR)
        meanHRC(i) = mean(HR_rolling(inWindowHR), 'omitnan');
    else
        inWinInterp = (ppgTable.time >= startTime) & (ppgTable.time <= endTime);
        meanHRC(i) = mean(HR_interp(inWinInterp), 'omitnan');
    end

    inWindowPPG = (ppgMatrix(:,1) >= startTime) & (ppgMatrix(:,1) <= endTime);
    if any(inWindowPPG)
        congMaxPPG(i) = max(ppgMatrix(inWindowPPG, 2), [], 'omitnan');
    end
end

%% plot task block HR, with windows marked (unchanged except variable scope)
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

% Rolling HR line
hHR = plot(ppgTable.time, HR_interp, 'b-', 'LineWidth',1.2, 'DisplayName','Rolling HR (5-peak mean)');
xlabel('Time (s)'); ylabel('Rolling HR (bpm)');
title(sprintf('Rolling HR with Congruent & Incongruent Windows — ID %s', IDstring), 'Interpreter','none');

[~, uniqIdx] = unique(string(get(findobj(gca,'-property','DisplayName'),'DisplayName')));
legendObjs = findobj(gca,'-property','DisplayName');
legend(legendObjs(uniqIdx), 'Location','best');
uistack(hHR,'top');

% Save figure
filename = sprintf('cFacePlot_%s.fig', IDstring);
savePath = fullfile(options.paths.plots, filename);
saveas(fig, savePath);
close(fig);

%% Final summaries
overallMeanRollingHR      = mean(meanHRs,  'omitnan');
overallMeanRollingHR_cong = mean(meanHRC,  'omitnan');
HR.incongruentAverage     = overallMeanRollingHR;
HR.congruentAverage       = overallMeanRollingHR_cong;

% ADDED: mean of per-trial max PPGs in each class
HR.incongruentPeakPPG     = mean(incMaxPPG,  'omitnan');
HR.congruentPeakPPG       = mean(congMaxPPG, 'omitnan');

% Report
fprintf('Baseline timing for %s: %.3f seconds\n', IDstring, baselineTiming);
fprintf('PostTaskHR (bpm) for %s: %.3f\n', IDstring, HR.postTask);
fprintf('Mean HR baseline (bpm): %.3f\n', HR.baseline);
fprintf('Mean HR in incongruent windows (bpm): %.3f\n', HR.incongruentAverage);
fprintf('Mean HR in congruent windows (bpm): %.3f\n', HR.congruentAverage);
fprintf('BaselinePeakPPG: %.3f | IncongruentPeakPPG: %.3f | CongruentPeakPPG: %.3f\n', ...
    HR.baselinePeakPPG, HR.incongruentPeakPPG, HR.congruentPeakPPG);

diary off
end

% --- tiny helper to keep code robust if peakTimes are empty
function t = HRTableTimeSafe(peakTimes)
    if isempty(peakTimes), t = NaN(0,1); else, t = peakTimes; end
end