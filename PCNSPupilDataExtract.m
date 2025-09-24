function PCNSPupilDataExtract
% Build pupil-based dataset and write CSV, including shared REDCap/HBD/PANSS columns.

options = specifyOptions();  % must return .paths.DBExport and any others you use

% ----- Get shared participant rows (with inclusion criteria applied)
shared = getSharedPCNSData_(options);    % table with ID and shared columns
N = height(shared);

% ----- Preallocate pupil vars
baselinePupil           = NaN(N,1);
averagePupil            = NaN(N,1);
incongruentPupilAverage = NaN(N,1);
congruentPupilAverage   = NaN(N,1);
eyeSide_cell            = repmat({''}, N, 1);   % store as 1-char strings ("R"/"L")

% ----- Fill pupil metrics
for i = 1:N
    currentID = shared.ID(i);
    try
        % [~, cFacePupil] per your convention
        [~, cFacePupil] = cFacePupilArea(currentID);

        if isfield(cFacePupil,'baseline'),             baselinePupil(i)            = cFacePupil.baseline;            end
        if isfield(cFacePupil,'peakAverage'),          averagePupil(i)             = cFacePupil.peakAverage;         end
        if isfield(cFacePupil,'incongruentAverage'),   incongruentPupilAverage(i)  = cFacePupil.incongruentAverage;  end
        if isfield(cFacePupil,'congruentAverage'),     congruentPupilAverage(i)    = cFacePupil.congruentAverage;    end
        if isfield(cFacePupil,'eyeSide')
            % coerce to 1-char string cell
            val = cFacePupil.eyeSide;
            if ischar(val) && ~isempty(val)
                eyeSide_cell{i} = val(1);
            elseif isstring(val) && strlength(val)>=1
                eyeSide_cell{i} = char(val(1));
            end
        end
    catch ME
        warning('cFacePupilArea failed for ID %d: %s', currentID, ME.message);
    end
end

% ----- Assemble pupil table (shared + pupil)
pupilTbl = shared;
pupilTbl.baselinePupil           = baselinePupil;
pupilTbl.eyeSide                 = eyeSide_cell;         % 1-char strings
pupilTbl.averagePupil            = averagePupil;
pupilTbl.incongruentPupilAverage = incongruentPupilAverage;
pupilTbl.congruentPupilAverage   = congruentPupilAverage;

% Optional: drop rows with missing ID (shouldn’t happen after shared filter)
pupilTbl = pupilTbl(~isnan(pupilTbl.ID), :);

% ----- Write CSV
outPath = fullfile(options.paths.DBExport, 'PCNS_PupilData.csv');
writetable(pupilTbl, outPath);
fprintf('Wrote pupil data: %s (%d rows)\n', outPath, height(pupilTbl));
end

function shared = getSharedPCNSData_(options)
% Returns a table with one row per included participant containing shared fields:
% ID, group, sex, age, education, FSIQ, meds_chlor, valid_cfacei,
% absInteroThreshold, absExteroThreshold, panss, panssPositive, panssNegative.

% --- Load REDCap
file = dir(fullfile(options.paths.DBExport, 'PCNS_RedCap_Export.csv'));
assert(~isempty(file), 'REDCap export not found in %s', options.paths.DBExport);
data = readtable(fullfile(options.paths.DBExport, file(1).name));

minID = min(data.record_id);
maxID = max(data.record_id);
IDspan = maxID - minID + 1;

% Preallocate working arrays
record_id     = NaN(IDspan,1);
group         = NaN(IDspan,1);
sex           = NaN(IDspan,1);
age_years     = NaN(IDspan,1);
edu_cat       = NaN(IDspan,1);
FSIQ          = NaN(IDspan,1);
meds_chlor    = NaN(IDspan,1);
valid_cfacei  = NaN(IDspan,1);
panss         = NaN(IDspan,1);
panssPositive = NaN(IDspan,1);
panssNegative = NaN(IDspan,1);

% --- Load HBD outcomes
HBDfile = dir(fullfile(options.paths.HBDExport, 'outcomes_myhrd.csv'));
assert(~isempty(HBDfile), 'HBD export not found in %s', options.paths.HBDExport);
HBDdata = readtable(fullfile(options.paths.HBDExport, HBDfile(1).name));
absInteroThreshold = NaN(IDspan,1);
absExteroThreshold = NaN(IDspan,1);

% --- Iterate REDCap IDs with inclusion criteria
rowIdx = 1;
for n = minID:maxID
    id_rows = find(data.record_id == n);
    if isempty(id_rows), continue; end

    % first occurrences
    details_row  = [];
    demogr_row   = [];
    clinical_row = [];

    k = find(strcmp(data.redcap_repeat_instrument(id_rows), 'participant_details'), 1, 'first');
    if ~isempty(k), details_row = id_rows(k); end
    k = find(strcmp(data.redcap_repeat_instrument(id_rows), 'demographics'), 1, 'first');
    if ~isempty(k), demogr_row = id_rows(k); end
    k = find(strcmp(data.redcap_repeat_instrument(id_rows), 'clinical'), 1, 'first');
    if ~isempty(k), clinical_row = id_rows(k); end

    % require demographics & details rows
    if isempty(demogr_row) || isempty(details_row), continue; end

    % Exclusions (demographics row)
    if data.valid_any(demogr_row) ~= 1,   continue; end
    if data.valid_cfacei(demogr_row) ~= 1, continue; end
    if data.fsiq2(demogr_row) < 80,        continue; end
    if data.pilotorreal(demogr_row) ~= 2,  continue; end

    % --- Basics
    currentID = data.record_id(details_row);
    record_id(rowIdx)   = currentID;
    age_years(rowIdx)   = data.age_years(details_row);
    group(rowIdx)       = data.group(demogr_row);
    sex(rowIdx)         = data.sex(demogr_row);
    edu_cat(rowIdx)     = data.edu_cat(demogr_row);
    FSIQ(rowIdx)        = data.fsiq2(demogr_row);
    meds_chlor(rowIdx)  = data.meds_chlor(demogr_row);
    if isempty(meds_chlor(rowIdx)) || isnan(meds_chlor(rowIdx)), meds_chlor(rowIdx) = 0; end
    valid_cfacei(rowIdx)= data.valid_cfacei(demogr_row);

    % --- PANSS (clinical row if present)
    if ~isempty(clinical_row)
        sumP = 0; sumN = 0; sumG = 0;
        for i = 1:7
            sumP = sumP + data.(['panss_p', num2str(i)])(clinical_row);
            sumN = sumN + data.(['panss_n', num2str(i)])(clinical_row);
        end
        for i = 1:16
            sumG = sumG + data.(['panss_g', num2str(i)])(clinical_row);
        end
        panss(rowIdx)         = sumP + sumN + sumG;
        panssPositive(rowIdx) = sumP;
        panssNegative(rowIdx) = sumN;
    else
        panss(rowIdx)         = 0;
        panssPositive(rowIdx) = 0;
        panssNegative(rowIdx) = 0;
    end

    % --- HBD
    idxH = find(HBDdata.record_id == currentID, 1, 'first');
    if ~isempty(idxH)
        absInteroThreshold(rowIdx) = abs(HBDdata.hrd_Intero_threshold_Bay(idxH));
        absExteroThreshold(rowIdx) = abs(HBDdata.hrd_Extero_threshold_Bay(idxH));
    end

    rowIdx = rowIdx + 1;
end

% keep only filled rows
filled = ~isnan(record_id);

% Assemble shared table
shared = table( ...
    record_id(filled), group(filled), age_years(filled), sex(filled), edu_cat(filled), FSIQ(filled), ...
    meds_chlor(filled), valid_cfacei(filled), ...
    absInteroThreshold(filled), absExteroThreshold(filled), ...
    panss(filled), panssPositive(filled), panssNegative(filled), ...
    'VariableNames', ...
    {'ID','group','age','sex','education','FSIQ', ...
     'meds_chlor','valid_cfacei', ...
     'absInteroThreshold','absExteroThreshold', ...
     'panss','panssPositive','panssNegative'} ...
);
end