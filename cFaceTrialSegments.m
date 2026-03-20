function segmentsTable = cFaceTrialSegments(participantID)
% cFaceTrialSegments  Read cFace behavioral trial timings (congruent + incongruent)
%                     for one participant from the *out.csv file.
%
% Usage:
%   T = cFaceTrialSegments(133);
%
% Returns a table with (at least) the columns:
%   - trialNo
%   - cong              (0=incongruent, 1=congruent)
%   - ptemot            (if present in CSV)
%   - stimMove_onset
%   - fixation_onset
%   - fixation_duration
%   - instruct_onset    (from CSV; if first-trial is 0, it is repaired)
%   - instruct_duration (if present)
%   - postinstruct_onset (if present)
%   - responseWindow    (= (fixation_onset + fixation_duration) - stimMove_onset)
%
% Notes:
%   * Looks for summary file at: <data>/<participant>/beh/cface*MH*/*out.csv
%   * Uses specifyOptions(). Make sure your options.paths.data is set.

options = specifyOptions;

% Normalise/format participant ID to 3 digits like '133'
if isnumeric(participantID)
    IDstring = sprintf('%03d', participantID);
else
    IDnum = str2double(regexp(char(participantID), '\d+', 'match','once'));
    if isnan(IDnum)
        error('participantID must contain digits (e.g., 133 or "PCNS_133").');
    end
    IDstring = sprintf('%03d', IDnum);
end

% Find participant folder containing the ID (case-insensitive)
d      = dir(options.paths.rawData);
names  = {d.name};
names  = names(~ismember(names,{'.','..'}));
hits   = contains(names, IDstring, 'IgnoreCase', true);
if ~any(hits)
    error('No matching participant folder found for "%s" under %s', IDstring, options.paths.data);
end
folderName = names{find(hits,1)};
folderPath = fullfile(options.paths.rawData, folderName);

% Find the *out.csv file
filePattern   = fullfile(folderPath, 'beh', 'cface*MH*out.csv');
summaryFile   = dir(filePattern);
if isempty(summaryFile)
    warning('No *out.csv found for %s.', IDstring);
    return
end
summaryFilePath = fullfile(summaryFile.folder, summaryFile.name);

% Import required columns (select what we need, but tolerate missing optional vars)
reqVars = {'stimMove_onset','fixation_onset','fixation_duration','cong'};
optVars = {'ptemot','instruct_onset','instruct_duration','postinstruct_onset','ParticipantID'};

opts = detectImportOptions(summaryFilePath);
csvVars = opts.VariableNames;

% Build the final list to read: required + any of the optional that exist
selVars = reqVars;
for v = 1:numel(optVars)
    if any(strcmpi(csvVars, optVars{v}))
        selVars{end+1} = optVars{v}; %#ok<AGROW>
    end
end
opts.SelectedVariableNames = selVars;

segmentsTable = readtable(summaryFilePath, opts);

% Sanity checks for required columns
for v = 1:numel(reqVars)
    if ~ismember(reqVars{v}, segmentsTable.Properties.VariableNames)
        error('Required variable "%s" not found in %s', reqVars{v}, summaryFilePath);
    end
end

% Add trial number (1..N)
segmentsTable.trialNo = (1:height(segmentsTable))';

% Ensure 'cong' is numeric/logical
if iscell(segmentsTable.cong) || isstring(segmentsTable.cong)
    % try to coerce (0/1) from text if necessary
    segmentsTable.cong = double(strcmpi(string(segmentsTable.cong),'1') | strcmpi(string(segmentsTable.cong),'true'));
end

% Create a human-readable condition label
segmentsTable.condition = repmat("incongruent", height(segmentsTable), 1);
segmentsTable.condition(segmentsTable.cong == 1) = "congruent";
segmentsTable.condition = categorical(segmentsTable.condition);

% Response window (as in your original function)
segmentsTable.responseWindow = ...
    (segmentsTable.fixation_onset + segmentsTable.fixation_duration) - segmentsTable.stimMove_onset;

% --- Fix missing first-trial instruct_onset if we have the columns ---
has_instruct_onset     = ismember('instruct_onset', segmentsTable.Properties.VariableNames);
has_instruct_duration  = ismember('instruct_duration', segmentsTable.Properties.VariableNames);
has_postinstruct_onset = ismember('postinstruct_onset', segmentsTable.Properties.VariableNames);

if has_instruct_onset && ~isempty(segmentsTable.instruct_onset)
    if segmentsTable.instruct_onset(1) == 0
        % Apply your fix rule
        if has_postinstruct_onset && ~isempty(segmentsTable.postinstruct_onset) && segmentsTable.postinstruct_onset(1) ~= 0 ...
           && has_instruct_duration && ~isempty(segmentsTable.instruct_duration)
            baselineTiming = segmentsTable.postinstruct_onset(1) - segmentsTable.instruct_duration(1);
        else
            baselineTiming = 10;  % default
        end
        segmentsTable.instruct_onset(1) = baselineTiming;
    end
end

% Reorder columns
wantOrder = {'trialNo','cong','condition','ptemot', ...
             'stimMove_onset','fixation_onset','fixation_duration', ...
             'instruct_onset','instruct_duration','postinstruct_onset', ...
             'responseWindow','ParticipantID'};
have = intersect(wantOrder, segmentsTable.Properties.VariableNames, 'stable');
rest = setdiff(segmentsTable.Properties.VariableNames, have, 'stable');
segmentsTable = segmentsTable(:, [have, rest]);

csvpath = fullfile(options.paths.trialTimings,strcat(num2str(participantID),...
    '_cface_TrialSegments.csv'));  
try
    writetable(segmentsTable, csvpath);
    fprintf('Saved trial-level table to %s\n', csvpath);
catch ME
    warning('Could not save segmentsTable: %s', ME.message);
end

end