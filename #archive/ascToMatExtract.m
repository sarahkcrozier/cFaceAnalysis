function ascToMatExtract(participantID)
% ascToMatExtract  Convert EyeLink .asc files to .mat tables of gaze and pupil data (robust timeline).
%
%   ascToMatExtract(participantID) finds all .asc files in options.paths.ASCtoMAT that
%   contain the participant ID (case-insensitive), parses samples and events, aligns
%   times to the SyncPulse (time zero), constructs a 100 Hz time-aware grid preserving
%   the true recording span, and saves <base>_eye.mat with eyeData, eyeDataTrialTimestamp,
%   and meta diagnostics.
%
% Usage:
%   ascToMatExtract(133)
%   ascToMatExtract('133')
%
% Output per matching .asc:
%   <base>_eye.mat containing:
%     - eyeData: table(time_s, gazeX, gazeY, pupilArea, pupilArea_lp4Hz, Input)
%     - eyeDataTrialTimestamp: table(Trial, StartTime_ms, StartTime_s)
%     - meta: struct with sync_time_ms, raw first/last sample times, etc.
%
% Requirements:
%   - specifyOptions (must define options.paths.ASCtoMAT, options.paths.eyeDataPlots)
%   - Signal Processing Toolbox (butter, filtfilt)

options = specifyOptions;
ascDir  = options.paths.ASCtoMAT;

% Ensure plot dir exists (used by downstream tools / optional figures)
if isfield(options.paths, 'eyeDataPlots') && ~exist(options.paths.eyeDataPlots, 'dir')
    mkdir(options.paths.eyeDataPlots);
end

% Normalize ID to string
if isnumeric(participantID), participantID = num2str(participantID); end

% Find .asc files for this participant
allFiles  = dir(fullfile(ascDir, '*.asc'));
matchMask = contains({allFiles.name}, participantID, 'IgnoreCase', true);
files     = allFiles(matchMask);

if isempty(files)
    warning('No .asc files found for ID "%s" in %s', participantID, ascDir);
    return;
end

for f = 1:numel(files)
    ascPath = fullfile(files(f).folder, files(f).name);
    fprintf('\nProcessing %s\n', ascPath);

    raw      = fileread(ascPath);
    linesAll = regexp(raw, '\r\n|\n', 'split')'; % full text
    lines    = linesAll;                         % working copy

    % ---------------------------------------------------------------------
    % 1) Find SyncPulse (anchor) BEFORE stripping
    % ---------------------------------------------------------------------
    sync_time = NaN; % ms
    syncMS = [];     % collect all, pick best if multiple
    for i = 1:numel(linesAll)
        L = char(linesAll{i});
        L(~isstrprop(L,'print') & L ~= char(9)) = ' ';
        if contains(L,'MSG') && contains(lower(L),'syncpulse')
            parts = strsplit(strtrim(L));
            if numel(parts) >= 2 && all(isstrprop(parts{2}, 'digit'))
                tms = str2double(parts{2});
                if ~isnan(tms), syncMS(end+1) = tms; end %#ok<AGROW>
            end
        end
    end

    % Find TRIALID 0 (for choosing nearest sync if multiple)
    trial0_ms = NaN;
    for i = 1:numel(linesAll)
        tok0 = regexp(linesAll{i}, '^\s*MSG\s+(\d+)\s+TRIALID\s+0\b', 'tokens', 'once');
        if ~isempty(tok0)
            trial0_ms = str2double(tok0{1}{1});
            break;
        end
    end

    if ~isempty(syncMS)
        if ~isnan(trial0_ms)
            % choose the last SyncPulse at/before TRIALID 0 (or last overall)
            cand = syncMS(syncMS <= trial0_ms);
            if ~isempty(cand), sync_time = cand(end); else, sync_time = syncMS(end); end
        else
            sync_time = syncMS(end);
        end
    end
    if isnan(sync_time)
        warning('No SyncPulse found; falling back to TRIALID 0 as anchor.');
        sync_time = trial0_ms;
    end
    if isnan(sync_time)
        warning('No SyncPulse or TRIALID 0 found; cannot anchor time. Skipping file.');
        continue;
    end
    fprintf('ANCHOR: sync_time (ms) = %g\n', sync_time);

    % START (for logging only; do not use to anchor)
    startTS = NaN;
    for i = 1:numel(linesAll)
        C = char(linesAll{i});
        C(~isstrprop(C,'print') & C ~= char(9)) = ' ';
        tok = regexp(C, '^\s*START\s+(\d+)\b', 'tokens', 'once');
        if ~isempty(tok)
            startTS = str2double(tok{1});
            break;
        end
    end
    if ~isnan(startTS), fprintf('INFO: START stamp (ms) = %g\n', startTS); end
    if ~isnan(trial0_ms), fprintf('INFO: TRIALID 0 (ms) = %g\n', trial0_ms); end

    % ---------------------------------------------------------------------
    % 2) Strip header (speeds up passes)
    % ---------------------------------------------------------------------
    firstIdx = find(~cellfun(@isempty, regexp(lines, '^\s*(?:\d+|MSG|START)\b', 'once')), 1, 'first');
    if isempty(firstIdx), firstIdx = 1; end
    lines = lines(firstIdx:end);

    % ---------------------------------------------------------------------
    % 3) Collect TRIALID starts and INPUT events
    % ---------------------------------------------------------------------
    trialIds   = [];
    trialTimes = []; % ms
    inputTimes = []; % ms
    
    for i = 1:numel(lines)
        L = lines{i};
        % normalise hidden chars (keep TAB)
        C = char(L);
        C(~isstrprop(C,'print') & C ~= char(9)) = ' ';
    
        % TRIALID: allow leading whitespace/tabs
        tokTrial = regexp(C, '^\s*MSG\s+(\d+)\s+TRIALID\s+(-?\d+)\b', 'tokens', 'once');
        if ~isempty(tokTrial)
            trialTimes(end+1,1) = str2double(tokTrial{1}); %#ok<AGROW>
            trialIds(end+1,1)   = str2double(tokTrial{2}); %#ok<AGROW>
            continue;
        end
    
        % INPUT (optional)
        tokInput = regexp(C, '^\s*MSG\s+(\d+)\s+INPUT\b', 'tokens', 'once');
        if ~isempty(tokInput)
            inputTimes(end+1,1) = str2double(tokInput{1}); %#ok<AGROW>
            continue;
        end
    end

    % Diagnostics + non-fatal handling
    has0  = any(trialIds == 0);
    has79 = any(trialIds == 79);
    if ~has0 || ~has79
        warning('TRIALID 0 present? %d | TRIALID 79 present? %d — continuing with whatever TRIALIDs are present.', has0, has79);
    else
        fprintf('Found TRIALID 0 at %g ms and TRIALID 79 at %g ms\n', ...
            trialTimes(find(trialIds==0,1,'first')), trialTimes(find(trialIds==79,1,'first')));
    end
    
    % Build the trial timestamp table from whatever TRIALIDs you found (in time order)
    if isempty(trialIds)
        warning('No TRIALID messages found after header; making empty trial table.');
        eyeDataTrialTimestamp = table([], [], [], 'VariableNames', {'Trial','StartTime_ms','StartTime_s'});
    else
        % sort by time; keep first occurrence of each TRIALID
        [trialTimes, order] = sort(trialTimes);
        trialIds            = trialIds(order);
        [~, firstIdx]       = unique(trialIds, 'stable');
        trialIds            = trialIds(firstIdx);
        trialTimes          = trialTimes(firstIdx);
    
        % Renumber trials to 1..N in *time order*
        Ntr         = numel(trialIds);
        Trials      = (1:Ntr).';
        StartTime_s = (trialTimes - sync_time)/1000;
    
        eyeDataTrialTimestamp = table(Trials, trialTimes, StartTime_s, ...
            'VariableNames', {'Trial','StartTime_ms','StartTime_s'});
        fprintf('TRIALS: %d unique TRIALID messages found (min ID=%d, max ID=%d). Using time order as Trials=1..%d.\n', ...
            Ntr, min(trialIds), max(trialIds), Ntr);
    end

    % ---------------------------------------------------------------------
    % 4) Parse sample rows (raw time series)
    % ---------------------------------------------------------------------
    isEvent = @(s) ~isempty(regexp(s, '^\s*(EFIX|ESACC|SFIX|SSACC|EBLINK|MSG)\b', 'once'));
    times  = []; xs = []; ys = []; pupils = [];
    for i = 1:numel(lines)
        L = lines{i};
        if isEvent(L), continue; end
        if isempty(regexp(L, '^\s*\d+', 'once')), continue; end
        nums = sscanf(L, '%f');
        if numel(nums) < 4, continue; end
        times(end+1,1)  = nums(1);  %#ok<AGROW> % ms
        xs(end+1,1)     = nums(2);  %#ok<AGROW>
        ys(end+1,1)     = nums(3);  %#ok<AGROW>
        pupils(end+1,1) = nums(4);  %#ok<AGROW>
    end
    if isempty(times)
        warning('File %s: No sample data parsed.', files(f).name);
        continue;
    end

    % DIAG: raw span vs sync
    fprintf('RAW since sync: first=%.3f s, last=%.3f s, span=%.3f s\n', ...
        (times(1)-sync_time)/1000, (times(end)-sync_time)/1000, (times(end)-times(1))/1000);

    % ---------------------------------------------------------------------
    % 5) Drop samples earlier than sync (do NOT re-anchor with START)
    % ---------------------------------------------------------------------
    keep = times >= sync_time;
    if any(~keep)
        fprintf('Dropping %d sample(s) t < sync_time (%g ms)\n', nnz(~keep), sync_time);
    end
    times  = times(keep);
    xs     = xs(keep);
    ys     = ys(keep);
    pupils = pupils(keep);
    if isempty(times)
        warning('All samples dropped by sync filter in %s', files(f).name);
        continue;
    end

    % ---------------------------------------------------------------------
    % 6) Time-aware resampling to 100 Hz (preserve true end; keep gaps as NaN)
    % ---------------------------------------------------------------------
    % Seconds since sync (may include gaps, may include duplicate stamps)
    t_s = (times - sync_time) / 1000;      % raw timestamps in seconds since sync
    
    % Deduplicate any repeated timestamps to keep interp1 happy
    [t_s_u, ia] = unique(t_s, 'stable');
    xs_u       = double(xs(ia));
    ys_u       = double(ys(ia));
    pup_u      = double(pupils(ia));
    
    % Build a 100 Hz uniform grid covering the true timestamp span
    targetFs = 100;                         % Hz
    dt       = 1/targetFs;
    tStart   = ceil(t_s_u(1) * targetFs) / targetFs;
    tEnd     = floor(t_s_u(end) * targetFs) / targetFs;
    if tEnd <= tStart, tEnd = t_s_u(end); end
    tDS      = (tStart:dt:tEnd).';         % column vector, seconds since sync
    
    % Interpolate gaze/pupil onto the grid; keep out-of-range points as NaN
    xDS = interp1(t_s_u, xs_u,  tDS, 'linear', NaN);
    yDS = interp1(t_s_u, ys_u,  tDS, 'linear', NaN);
    pDS = interp1(t_s_u, pup_u, tDS, 'linear', NaN);
    
    % Map INPUT events to nearest grid sample (only those within [tStart,tEnd])
    inputDS = zeros(size(tDS));
    if ~isempty(inputTimes)
        inputTimes_s = (inputTimes - sync_time) / 1000;   % seconds since sync
        inRange = (inputTimes_s >= tDS(1)) & (inputTimes_s <= tDS(end));
        if any(inRange)
            [~, j] = min(abs(tDS - inputTimes_s(inRange).'), [], 1);
            inputDS(j) = 1;
        end
    end

    % ---------------------------------------------------------------------
    % 7) Low-pass pupil at 4 Hz with NaN-safe handling (zero-phase)
    % ---------------------------------------------------------------------
    fs = targetFs; fc = 4;
    [b,a] = butter(4, fc/(fs/2), 'low');
    nanMask = isnan(pDS);
    if any(~nanMask)
        pFill = pDS;
        pFill = fillmissing(pFill, 'linear', 'EndValues','nearest');
        pLP4  = filtfilt(b, a, pFill);
        pLP4(nanMask) = NaN; % restore gaps
    else
        pLP4 = pDS; % degenerate all-NaN case
    end

    % ---------------------------------------------------------------------
    % 8) Assemble output table & diagnostics
    % ---------------------------------------------------------------------
    eyeData = table(tDS, xDS, yDS, pDS, pLP4, inputDS, ...
        'VariableNames', {'time_s','gazeX','gazeY','pupilArea','pupilArea_lp4Hz','Input'});

    fprintf('\nTrial Timestamp Table for %s:\n', files(f).name);
    disp(eyeDataTrialTimestamp(1:min(6,height(eyeDataTrialTimestamp)),:));
    fprintf('... (%d trials total)\n', height(eyeDataTrialTimestamp));

    fprintf('eyeData: first rows:\n');
    disp(eyeData(1:min(5,height(eyeData)),:));
    fprintf('eyeData: last rows:\n');
    disp(eyeData(max(1,height(eyeData)-4):end,:));

    fprintf('RANGE CHECK: eyeData.time_s ∈ [%.3f, %.3f] s since sync\n', min(eyeData.time_s), max(eyeData.time_s));
    fprintf('CHECK: raw last sample since sync: %.3f s\n', (times(end) - sync_time)/1000);

    % Meta diagnostics to save
    meta = struct();
    meta.sync_time_ms           = sync_time;
    meta.startTS_ms             = startTS;
    meta.trial0_ms              = trial0_ms;
    meta.first_raw_ms           = times(1);
    meta.last_raw_ms            = times(end);
    meta.first_s_after_sync     = (times(1)-sync_time)/1000;
    meta.last_s_after_sync      = (times(end)-sync_time)/1000;
    meta.grid_first_s_after_sync= tDS(1);
    meta.grid_last_s_after_sync = tDS(end);
    meta.targetFs               = targetFs;

    % ---------------------------------------------------------------------
    % 9) Save MAT next to the ASC
    % ---------------------------------------------------------------------
    [~, baseName, ~] = fileparts(ascPath);  % e.g., '133'
    outMat = fullfile(files(f).folder, [baseName, '_eye.mat']);
    save(outMat, 'eyeData', 'eyeDataTrialTimestamp', 'meta');
    fprintf('Saved %s\n', outMat);
end
end