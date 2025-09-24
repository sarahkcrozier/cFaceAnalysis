function shadeCfaceWindows(participantID, durationSec, ax)
% Shade 2.75 s (default) windows after stimMove_onset for congruent/incongruent trials
% and overlay pupilArea vs convertedTime_s from eyeData CSV.
%
% Usage:
%   shadeCfaceWindows(25);   % will look for 025_cface_eyeData.csv
%   shadeCfaceWindows(133);  % will look for 133_cface_eyeData.csv

    if nargin < 2 || isempty(durationSec), durationSec = 2.75; end
    if nargin < 3 || isempty(ax) || ~isgraphics(ax,'axes')
        figure; ax = gca;
    end

    % --- Load eyeData file for this participant
    options = specifyOptions();  % make sure this returns struct with paths.eyeData
    pidStr  = sprintf('%03d', participantID);  % zero-pad to 3 digits
    eyeFile = fullfile(options.paths.eyeData, sprintf('%s_cface_eyeData.csv', pidStr));
    assert(exist(eyeFile,'file')==2, 'Eye data file not found: %s', eyeFile);

    eyeData = readtable(eyeFile);
    assert(all(ismember({'convertedTime_s','pupilArea'}, eyeData.Properties.VariableNames)), ...
        'Eye data file must contain columns convertedTime_s and pupilArea.');

    % --- Plot pupil trace
    plot(ax, eyeData.convertedTime_s, eyeData.pupilArea, 'k-');
    xlabel(ax,'Converted time (s)'); ylabel(ax,'Pupil area');
    title(ax, sprintf('Participant %s', pidStr));
    hold(ax,'on');

    % --- Get trial segments
seg  = cFaceTrialSegments(participantID);
vars = seg.Properties.VariableNames;

assert(ismember('stimMove_onset', vars), ...
    'cFaceTrialSegments must contain a stimMove_onset column (seconds).');
assert(ismember('cong', vars), ...
    'cFaceTrialSegments must contain a ''cong'' column (1 = congruent, 0 = incongruent).');

% Onsets (seconds)
on = seg.stimMove_onset(:);
if isdatetime(on), on = seconds(on - on(1)); end

% Coerce cong -> numeric 0/1
c = seg.cong;
if islogical(c)
    congNum = double(c);
elseif isnumeric(c)
    congNum = c;
elseif iscategorical(c)
    congNum = str2double(string(c));
elseif isstring(c) || iscellstr(c)
    congNum = str2double(string(c));
else
    error('Unsupported type for seg.cong: %s', class(c));
end

% --- Congruency masks from 'cong'
maskCong   = (congNum == 1) & ~isnan(on);
maskIncong = (congNum == 0) & ~isnan(on);

% Unique onset times per class
onCong   = unique(on(maskCong));
onIncong = unique(on(maskIncong));

    % --- Draw shaded windows
    yl = ax.YLim;
    drawPatches(ax, onCong,   durationSec, yl, [0.85 0.93 1.00], 0.25); % blue
    drawPatches(ax, onIncong, durationSec, yl, [1.00 0.88 0.88], 0.25); % red

    % --- Guide lines
    for x = onCong(:).'
        xline(ax, x, '-', 'Color',[0.30 0.55 0.85], 'LineWidth',0.75, ...
              'Alpha',0.5, 'HandleVisibility','off');
    end
    for x = onIncong(:).'
        xline(ax, x, '-', 'Color',[0.85 0.35 0.35], 'LineWidth',0.75, ...
              'Alpha',0.5, 'HandleVisibility','off');
    end

    % --- Legend
    if isempty(get(ax,'Legend'))
        lbl = {};
        if ~isempty(onCong),   lbl{end+1} = 'Congruent window';   end %#ok<AGROW>
        if ~isempty(onIncong), lbl{end+1} = 'Incongruent window'; end %#ok<AGROW>
        if ~isempty(lbl), legend(ax, lbl, 'Location','best'); end
    end

    drawnow;
end

function drawPatches(ax, onsets, dur, yl, faceRGB, alphaVal)
    if isempty(onsets), return; end
    onsets = onsets(~isnan(onsets));
    for k = 1:numel(onsets)
        x1 = onsets(k); x2 = x1 + dur;
        patch(ax, [x1 x2 x2 x1], [yl(1) yl(1) yl(2) yl(2)], faceRGB, ...
              'FaceAlpha',alphaVal,'EdgeColor','none','HitTest','off', ...
              'HandleVisibility','off');
    end
end