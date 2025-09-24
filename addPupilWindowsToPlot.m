function addPupilWindowsToPlot(participantID, durationSec, figDir)
% Add congruent/incongruent windows (from cFaceTrialSegments) to
% pupil_plot_[ID].fig and update the corresponding PNG (no zero-padding).
%
% Usage:
%   addPupilWindowsToPlot(28);   % uses pupil_plot_28.fig/.png

    if nargin < 2 || isempty(durationSec), durationSec = 2.75; end

    options = specifyOptions();
    if nargin < 3 || isempty(figDir), figDir = options.paths.eyeDataPlots; end

    % ---- filenames (NO zero padding)
    pidStr  = sprintf('%d', participantID);
    figFile = fullfile(figDir, sprintf('pupil_plot_%s.fig', pidStr));
    pngFile = fullfile(figDir, sprintf('pupil_plot_%s.png', pidStr));
    assert(exist(figFile,'file')==2, 'Could not find figure: %s', figFile);

    % ---- get segments and build onset lists
    seg = cFaceTrialSegments(participantID);
    assert(ismember('stimMove_onset', seg.Properties.VariableNames), ...
        'cFaceTrialSegments must include stimMove_onset.');
    assert(ismember('cong', seg.Properties.VariableNames), ...
        'cFaceTrialSegments must include cong (1=congruent, 0=incongruent).');

    on = seg.stimMove_onset(:);
    if isdatetime(on), on = seconds(on - on(1)); end

    % cong → numeric 0/1
    c = seg.cong;
    if islogical(c),        congNum = double(c);
    elseif isnumeric(c),    congNum = c;
    elseif iscategorical(c) || isstring(c) || iscellstr(c)
        congNum = str2double(string(c));
    else, error('Unsupported type for seg.cong: %s', class(c));
    end

    onCong   = unique(on(congNum==1 & ~isnan(on)));
    onIncong = unique(on(congNum==0 & ~isnan(on)));

    % ---- open and modify figure
    fig = openfig(figFile, 'invisible');
    ax  = findobj(fig, 'Type','axes'); if isempty(ax), ax = axes('Parent',fig); end
    ax = ax(1);
    hold(ax,'on'); drawnow;

    % Y-lims safety
    yl = ax.YLim;
    if any(~isfinite(yl)) || diff(yl)==0
        L = findobj(ax,'Type','line');
        if ~isempty(L)
            ydata = vertcat(L.YData);
            yl = [min(ydata(:)) max(ydata(:))];
            if diff(yl)==0, yl = yl + [-1 1]*eps; end
            ylim(ax, yl);
        else
            yl = [0 1]; ylim(ax, yl);
        end
    end

    % patches
    drawPatches(ax, onCong,   durationSec, yl, [0.85 0.93 1.00], 0.25);
    drawPatches(ax, onIncong, durationSec, yl, [1.00 0.88 0.88], 0.25);

    % guide lines (scalar xline compatibility)
    for x = onCong(:).'
        xline(ax, x, '-', 'Color',[0.30 0.55 0.85], 'LineWidth',0.75, 'Alpha',0.5, 'HandleVisibility','off');
    end
    for x = onIncong(:).'
        xline(ax, x, '-', 'Color',[0.85 0.35 0.35], 'LineWidth',0.75, 'Alpha',0.5, 'HandleVisibility','off');
    end

    % expand x-lims to include windows
    allStarts = [onCong(:); onIncong(:)];
    if ~isempty(allStarts)
        xmin = min([ax.XLim(1), allStarts']);
        xmax = max([ax.XLim(2), (allStarts'+durationSec)]);
        if isfinite(xmin) && isfinite(xmax) && xmin < xmax, xlim(ax, [xmin xmax]); end
    end

    % legend (only if none)
    if isempty(get(ax,'Legend'))
        tags = {};
        if ~isempty(onCong),   tags{end+1} = 'Congruent window';   end %#ok<AGROW>
        if ~isempty(onIncong), tags{end+1} = 'Incongruent window'; end %#ok<AGROW>
        if ~isempty(tags), legend(ax, tags, 'Location','best'); end
    end

    % save .fig and .png (overwrite)
    drawnow;
    savefig(fig, figFile);
    try
        exportgraphics(fig, pngFile, 'Resolution',300);
    catch
        print(fig, pngFile, '-dpng', '-r300');
    end
    set(fig,'Visible','on');

    fprintf('Updated:\n  %s\n  %s\n', figFile, pngFile);
end

function drawPatches(ax, onsets, dur, yl, faceRGB, alphaVal)
    if isempty(onsets), return; end
    onsets = onsets(~isnan(onsets));
    for k = 1:numel(onsets)
        x1 = onsets(k); x2 = x1 + dur;
        patch(ax, [x1 x2 x2 x1], [yl(1) yl(1) yl(2) yl(2)], faceRGB, ...
              'FaceAlpha',alphaVal,'EdgeColor','none','HitTest','off','HandleVisibility','off');
    end
end