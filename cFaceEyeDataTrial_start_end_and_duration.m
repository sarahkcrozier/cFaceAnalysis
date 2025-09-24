function trialWindows = cFaceEyeDataTrial_start_end_and_duration
% TRIAL_START_END_AND_DURATION
% Given a CSV with columns convertedTime_s and trial,
% returns a table with t_start, t_end, and duration for trials 0..70.

filePath = ['/Users/yamaan/Projects/PCNS/Data/eyeData/025_cface_eyeData']

    % --- Read & sanity checks ---
    T = readtable(filePath);
    req = {'convertedTime_s','trial'};
    assert(all(ismember(req, T.Properties.VariableNames)), ...
        'CSV must contain columns: convertedTime_s and trial');

    % Normalize 'trial' to numeric (handles categorical/strings)
    if iscategorical(T.trial)
        T.trial = str2double(string(T.trial));  % convert category labels like '0','1',...
    elseif isstring(T.trial) || iscellstr(T.trial)
        T.trial = str2double(string(T.trial));
    end

    % Normalize time to numeric seconds
    if isdatetime(T.convertedTime_s)
        tSec = seconds(T.convertedTime_s - T.convertedTime_s(1));
    elseif isduration(T.convertedTime_s)
        tSec = seconds(T.convertedTime_s);
    else
        tSec = T.convertedTime_s; % assume already numeric seconds
    end
    validateattributes(tSec, {'numeric'},{'vector','real'});

    % --- Sort by time to preserve order within trials ---
    T = sortrows(table(tSec, T.trial, 'VariableNames', {'t','trial'}), 't');

    % --- Keep only trials 0..70, drop NaNs ---
    keep = ~isnan(T.trial) & T.trial >= 0 & T.trial <= 79;
    t  = T.t(keep);
    tr = T.trial(keep);

    % Early exit if nothing to do
    if isempty(t)
        trialWindows = table((0:70)', nan(71,1), nan(71,1), nan(71,1), zeros(71,1), ...
            'VariableNames', {'trial','t_start_s','t_end_s','duration_s','n_samples'});
        return
    end

    % --- Group by trial and take first/last time (start/end) ---
    [g, trialVals] = findgroups(tr);
    t_start = splitapply(@(x) x(1),  t, g);   % first time per trial
    t_end   = splitapply(@(x) x(end), t, g);  % last time  per trial
    n_samp  = splitapply(@numel,       t, g);
    dur     = t_end - t_start;

    % --- Assemble a full 0..70 table; fill missing trials with NaN ---
    allTrials = (0:79)';
    [present, loc] = ismember(allTrials, trialVals);

    t_start_full = nan(size(allTrials));
    t_end_full   = nan(size(allTrials));
    dur_full     = nan(size(allTrials));
    n_full       = zeros(size(allTrials));

    t_start_full(present) = t_start(loc(present));
    t_end_full(present)   = t_end(loc(present));
    dur_full(present)     = dur(loc(present));
    n_full(present)       = n_samp(loc(present));

    trialWindows = table(allTrials, t_start_full, t_end_full, dur_full, n_full, ...
        'VariableNames', {'trial','t_start_s','t_end_s','duration_s','n_samples'});

    % --- (Optional) save next to the CSV ---
    [folder, base, ~] = fileparts(filePath);
    outPath = fullfile(folder, [base '_trialWindows_0to80.csv']);
    writetable(trialWindows, outPath);
    fprintf('Wrote: %s\n', outPath);
end