function plotAscRaw(participantID)
% plotAscRaw  Plot raw pupil area from an EyeLink ASC for a participant.
%             Marks trial starts and shades last 2.75 s of each trial.
%
% Usage: plotAscRaw(133)

options = specifyOptions;
ascDir  = options.paths.EDFtoASC;

if isnumeric(participantID), participantID = num2str(participantID); end

% Find newest matching .asc
allAsc = dir(fullfile(ascDir, '*.asc'));
mask   = contains({allAsc.name}, participantID, 'IgnoreCase', true);
ascs   = allAsc(mask);
assert(~isempty(ascs), 'No .asc file containing "%s" found in %s', participantID, ascDir);

[~, newest] = max([ascs.datenum]);
ascPath = fullfile(ascs(newest).folder, ascs(newest).name);
fprintf('Using ASC: %s\n', ascPath);

% Read and split
raw      = fileread(ascPath);
linesAll = regexp(raw, '\r\n|\n', 'split')';

% --- Detect SyncPulse and START (robust to tabs/unicode) ---
sync_time = NaN;  % ms
startTS   = NaN;  % ms
for i = 1:numel(linesAll)
    C = char(linesAll{i});
    if isempty(C), continue; end
    C(~isstrprop(C,'print') & C ~= char(9)) = ' '; % normalize; keep tabs

    if isnan(sync_time) && strncmp(C,'MSG',3) && contains(lower(C),'syncpulse')
        parts = strsplit(strtrim(C));
        if numel(parts) >= 2 && all(isstrprop(parts{2},'digit'))
            sync_time = str2double(parts{2});
        end
    end
    if isnan(startTS)
        tokS = regexp(C, '^\s*START\s+(\d+)\b', 'tokens','once');
        if ~isempty(tokS), startTS = str2double(tokS{1}); end
    end
    if ~isnan(sync_time) && ~isnan(startTS), break; end
end

% --- Collect TRIALIDs and raw samples (no filtering/decimation) ---
trialIds = []; trialTimes = [];  % ms
times = []; pupils = [];         % ms; raw pupil area

isEvent = @(s) ~isempty(regexp(s, '^\s*(EFIX|ESACC|SFIX|SSACC|EBLINK|MSG)\b', 'once'));

for i = 1:numel(linesAll)
    L = linesAll{i};
    if isempty(L), continue; end

    tokT = regexp(L, '^MSG\s+(\d+)\s+TRIALID\s+(-?\d+)', 'tokens','once');
    if ~isempty(tokT)
        trialTimes(end+1,1) = str2double(tokT{1}); %#ok<AGROW>
        trialIds(end+1,1)   = str2double(tokT{2}); %#ok<AGROW>
        continue
    end

    if isEvent(L), continue; end
    if isempty(regexp(L, '^\s*\d+', 'once')), continue; end
    nums = sscanf(L, '%f');
    if numel(nums) < 4, continue; end

    times(end+1,1)  = nums(1); %#ok<AGROW>
    pupils(end+1,1) = nums(4); %#ok<AGROW>
end

assert(~isempty(times), 'No sample rows parsed from %s', ascPath);

% --- Choose anchor for dropping, and origin for plotting/labels ---
% Anchor for dropping: prefer START; else SyncPulse; else no drop
if ~isnan(startTS)
    anchorMS = startTS;
elseif ~isnan(sync_time)
    anchorMS = sync_time;
else
    anchorMS = -inf; % don't drop anything
end

% Drop pre-anchor samples (if anchor is finite)
if isfinite(anchorMS)
    keep = times >= anchorMS;
    dropped = nnz(~keep);
    if dropped > 0
        fprintf('Dropping %d sample(s) with t < %g ms (anchor)\n', dropped, anchorMS);
    end
    times  = times(keep);
    pupils = pupils(keep);
end

% Origin for plotting/labels: prefer SyncPulse; else START; else first sample
if ~isnan(sync_time)
    originMS = sync_time;
elseif ~isnan(startTS)
    originMS = startTS;
else
    originMS = times(1);
end

% Convert to seconds
t_s = (times - originMS) / 1000;

if isempty(t_s)
    warning('No samples left after filtering; nothing to plot.');
    return;
end

% --- Prepare trial starts in seconds (label 1..80 for TRIALID 0..79) ---
trialStarts_s = [];
trialNums     = [];
if ~isempty(trialIds)
    want   = (trialIds >= 0) & (trialIds <= 79);
    trIds  = trialIds(want);
    trTime = trialTimes(want);
    [trIds, ord] = sort(trIds);
    trTime = trTime(ord);
    trialNums     = trIds + 1;
    trialStarts_s = (trTime - originMS) / 1000;
end

% --- Plot ---
figure('Color','w','Name', sprintf('Raw pupil — %s', ascs(newest).name), ...
       'Units','normalized','Position',[0.15 0.15 0.7 0.55]);
plot(t_s, pupils, '-', 'LineWidth', 0.9); hold on; grid on; box on;
xlabel('Time (s)');
ylabel('Pupil area (raw)');
title(sprintf('Raw pupil vs time — %s', ascs(newest).name), 'Interpreter','none');

% --- Shade last 2.75 s of each trial & mark starts ---
if ~isempty(trialStarts_s)
    % define trial ends = next start; last ends at last sample
    lastTime_s = t_s(end);
    trialEnds_s = [trialStarts_s(2:end); lastTime_s];

    yl = ylim;
    shadeDur = 2.75;  % seconds

    for k = 1:numel(trialStarts_s)
        t0 = trialStarts_s(k);
        t1 = trialEnds_s(k);

        % Shade last 2.75 s of trial k
        x1 = max(t0, t1 - shadeDur);
        x2 = t1;
        if x2 > x1 && isfinite(x1) && isfinite(x2)
            patch([x1 x2 x2 x1], [yl(1) yl(1) yl(2) yl(2)], [0.85 0.90 1.00], ...
                  'FaceAlpha', 0.18, 'EdgeColor','none');
        end

        % Trial start line with label
        xline(t0, '--', sprintf('T%d', trialNums(k)), ...
              'LabelVerticalAlignment','bottom', 'LabelOrientation','horizontal', ...
              'Color',[0.4 0.4 0.4 0.85]);
    end

    % keep the trace visible on top of patches
    hLine = findobj(gca,'Type','line');
    uistack(hLine,'top');
end

outDir = options.paths.eyeDataPlots;

% Derive a clean base name from the ASC filename
[~, baseName] = fileparts(ascs(newest).name);

figPath = fullfile(outDir, sprintf('%s_raw.fig', baseName));
pngPath = fullfile(outDir, sprintf('%s_raw.png', baseName));

savefig(gcf, figPath);
exportgraphics(gcf, pngPath, 'Resolution',150);

fprintf('Saved figure to:\n  %s\n  %s\n', figPath, pngPath);

end