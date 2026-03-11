function shared = getSharedPCNSData(options)
% Returns a table with one row per included participant containing shared fields:
% ID, group, sex, age, education, FSIQ, meds_chlor, valid_cfacei,
% absInteroThreshold, absExteroThreshold, panss, panssPositive, panssNegative.

% --- Load REDCap

data = readtable(fullfile(options.data.DBFileName));

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
HBDdata = readtable(fullfile(options.data.hbdOutcomesFile));
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
    if data.valid_any(demogr_row) ~= 1,   continue; end  % 1 = some data usable, 2 = none of the data usable
    if data.valid_cfacei(demogr_row) ~= 1, continue; end % 1 = cface data usable, 2 = cface data not usable
    if data.fsiq2(demogr_row) < 80,        continue; end
    if data.pilotorreal(demogr_row) ~= 2,  continue; end % 1 = pilot, 2 = participant, NaN = not filled out 

    % --- Basics
    currentID           = data.record_id(details_row);
    record_id(rowIdx)   = currentID;
    age_years(rowIdx)   = data.age_years(details_row);
    group(rowIdx)       = data.group(demogr_row); % 1 = control, 2 = patient
    sex(rowIdx)         = data.sex(demogr_row);   % 1 = male, 2 = female
    edu_cat(rowIdx)     = data.edu_cat(demogr_row); % 1 = didnt finish HS, 2 = HS, 3 = non-uni edu, 4 = BA, 5 = MA, 6 = PhD 
    FSIQ(rowIdx)        = data.fsiq2(demogr_row); 
    meds_chlor(rowIdx)  = data.meds_chlor(demogr_row);
    if isempty(meds_chlor(rowIdx)) || isnan(meds_chlor(rowIdx)), meds_chlor(rowIdx) = 0; end
    valid_cfacei(rowIdx)= data.valid_cfacei(demogr_row);

    % --- PANSS (clinical row if present)
    if ~isempty(clinical_row)
        sumP = 0; sumN = 0; sumG = 0;

        % PANSS positive symptoms (sumP) and negative symptoms (sumN)
        for i = 1:options.data.nPannsPItems
            sumP = sumP + data.(['panss_p', num2str(i)])(clinical_row);
            sumN = sumN + data.(['panss_n', num2str(i)])(clinical_row);
        end
        for i = 1:options.data.nPannsGItems
            sumG = sumG + data.(['panss_g', num2str(i)])(clinical_row);
        end
        panss(rowIdx)         = sumP + sumN + sumG;
        panssPositive(rowIdx) = sumP;
        panssNegative(rowIdx) = sumN;
    else
        panss(rowIdx)         = 0; % would NaN or [] be better?
        panssPositive(rowIdx) = 0; % would NaN or [] be better?
        panssNegative(rowIdx) = 0; % would NaN or [] be better?
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