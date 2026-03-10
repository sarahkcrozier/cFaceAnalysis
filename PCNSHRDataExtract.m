function PCNSHRDataExtract
% Build HR-based dataset and write CSV, including shared REDCap/HBD/PANSS columns,
% HR summary measures, PeakPPG metrics, and PPI summary metrics from cFaceHR.

options = specifyOptions();  % must return .paths.DBExport (and others you use)

% ----- Get shared participant rows (with inclusion criteria applied)
shared = getSharedPCNSData(options);    % table with ID and shared columns
N      = height(shared);

%% Initualize data rows
% ----- Preallocate HR vars
baselineHR            = NaN(N,1);
postTaskHR            = NaN(N,1);
incongruentHRaverage  = NaN(N,1);
congruentHRaverage    = NaN(N,1);

% ----- PeakPPG vars (supports old/new field names from cFaceHR)
baselinePeakPPG       = NaN(N,1);
incongruentPeakPPG    = NaN(N,1);
congruentPeakPPG      = NaN(N,1);

% ----- New PPI summary metrics (seconds / percent as noted)
IncongruentDiffPPI            = NaN(N,1);   % s
CongruentDiffPPI              = NaN(N,1);   % s
AllTrialsDiffPPI              = NaN(N,1);   % s
IncongruentPercentageDiffPPI  = NaN(N,1);   % %
CongruentPercentageDiffPPI    = NaN(N,1);   % %
AllTrialsPercentageDiffPPI    = NaN(N,1);   % %
IncongruentMaxChangePPI       = NaN(N,1);   % s
CongruentMaxChangePPI         = NaN(N,1);   % s
AllTrialsMaxChangePPI         = NaN(N,1);   % s
baselinePPI                   = NaN(N,1);   % s
incongruentBaselinePPI        = NaN(N,1);   % s
incongruentResponsePPI        = NaN(N,1);   % s

%% get data
% ----- Fill metrics per participant
for i = 1:N
    currentID = shared.ID(i);
    try
        HR = cFaceHR(currentID);

        % HR measures
        baselineHR(i)           = HR.baseline;
        postTaskHR(i)           = HR.postTask;
        incongruentHRaverage(i) = HR.incongruentAverage;
        congruentHRaverage(i)   = HR.congruentAverage;

        % PeakPPG measures (support old/new field names)
        if     isfield(HR,'baselineHighpassedPeakPPG'),      baselinePeakPPG(i)    = HR.baselineHighpassedPeakPPG;
        elseif isfield(HR,'baselinePeakPPG'),                baselinePeakPPG(i)    = HR.baselinePeakPPG;
        end
        if     isfield(HR,'incongruentHighpassedPeakPPG'),   incongruentPeakPPG(i) = HR.incongruentHighpassedPeakPPG;
        elseif isfield(HR,'incongruentPeakPPG'),             incongruentPeakPPG(i) = HR.incongruentPeakPPG;
        end
        if     isfield(HR,'congruentHighpassedPeakPPG'),     congruentPeakPPG(i)   = HR.congruentHighpassedPeakPPG;
        elseif isfield(HR,'congruentPeakPPG'),               congruentPeakPPG(i)   = HR.congruentPeakPPG;
        end

        % New PPI summary metrics (flat fields created in cFaceHR)
        if isfield(HR,'IncongruentDiffPPI'),           IncongruentDiffPPI(i)           = HR.IncongruentDiffPPI;           end
        if isfield(HR,'CongruentDiffPPI'),             CongruentDiffPPI(i)             = HR.CongruentDiffPPI;             end
        if isfield(HR,'AllTrialsDiffPPI'),             AllTrialsDiffPPI(i)             = HR.AllTrialsDiffPPI;             end

        if isfield(HR,'IncongruentPercentageDiffPPI'), IncongruentPercentageDiffPPI(i) = HR.IncongruentPercentageDiffPPI; end
        if isfield(HR,'CongruentPercentageDiffPPI'),   CongruentPercentageDiffPPI(i)   = HR.CongruentPercentageDiffPPI;   end
        if isfield(HR,'AllTrialsPercentageDiffPPI'),   AllTrialsPercentageDiffPPI(i)   = HR.AllTrialsPercentageDiffPPI;   end

        if isfield(HR,'IncongruentMaxChangePPI'),      IncongruentMaxChangePPI(i)      = HR.IncongruentMaxChangePPI;      end
        if isfield(HR,'CongruentMaxChangePPI'),        CongruentMaxChangePPI(i)        = HR.CongruentMaxChangePPI;        end
        if isfield(HR,'AllTrialsMaxChangePPI'),        AllTrialsMaxChangePPI(i)        = HR.AllTrialsMaxChangePPI;        end

       
        % ---- Baseline/response PPI exports (seconds) ----
        % baselinePPI (mean PPI during baseline beats)
        if isfield(HR,'baselinePPI'), baselinePPI(i) = HR.baselinePPI; end

        % incongruent baseline/response PPI
        % (handle both spellings of the response field to be safe)
        if isfield(HR,'incongruentBaselinePPI')
            incongruentBaselinePPI(i) = HR.incongruentBaselinePPI;
        end
        if isfield(HR,'incongruentResponsePPI')
            incongruentResponsePPI(i) = HR.incongruentResponsePPI;
        elseif isfield(HR,'incongruentrespontPPI')  % fallback if earlier code used this name
            incongruentResponsePPI(i) = HR.incongruentrespontPPI;
        end


        
        % Optional concise per-participant print (no rounding enforced)
        fprintf(['ID %d | base=%.6f bpm, post=%.6f bpm | incHR=%.6f, congHR=%.6f | ' ...
                 'PeakPPG: base=%.6f, inc=%.6f, cong=%.6f | ' ...
                 'ΔPPI(s): inc=%.6f, cong=%.6f, all=%.6f | %%ΔPPI: inc=%.6f, cong=%.6f, all=%.6f | ' ...
                 'maxΔPPI(s): inc=%.6f, cong=%.6f, all=%.6f\n'], ...
            currentID, baselineHR(i), postTaskHR(i), ...
            incongruentHRaverage(i), congruentHRaverage(i), ...
            baselinePeakPPG(i), incongruentPeakPPG(i), congruentPeakPPG(i), ...
            IncongruentDiffPPI(i), CongruentDiffPPI(i), AllTrialsDiffPPI(i), ...
            IncongruentPercentageDiffPPI(i), CongruentPercentageDiffPPI(i), AllTrialsPercentageDiffPPI(i), ...
            IncongruentMaxChangePPI(i), CongruentMaxChangePPI(i), AllTrialsMaxChangePPI(i));
    catch ME
        warning('cFaceHR failed for ID %d: %s', currentID, ME.message);
    end
end

%% create table
% ----- Assemble output table (shared + HR + PeakPPG + PPI summaries)
hrTbl = shared;

% HR
hrTbl.baselineHR            = baselineHR;
hrTbl.postTaskHR            = postTaskHR;
hrTbl.incongruentHRaverage  = incongruentHRaverage;
hrTbl.congruentHRaverage    = congruentHRaverage;

% Peak PPG
hrTbl.baselinePeakPPG       = baselinePeakPPG;
hrTbl.incongruentPeakPPG    = incongruentPeakPPG;
hrTbl.congruentPeakPPG      = congruentPeakPPG;

% PPI summary metrics
hrTbl.IncongruentDiffPPI            = IncongruentDiffPPI;            % seconds
hrTbl.CongruentDiffPPI              = CongruentDiffPPI;              % seconds
hrTbl.AllTrialsDiffPPI              = AllTrialsDiffPPI;              % seconds
hrTbl.IncongruentPercentageDiffPPI  = IncongruentPercentageDiffPPI;  % percent
hrTbl.CongruentPercentageDiffPPI    = CongruentPercentageDiffPPI;    % percent
hrTbl.AllTrialsPercentageDiffPPI    = AllTrialsPercentageDiffPPI;    % percent
hrTbl.IncongruentMaxChangePPI       = IncongruentMaxChangePPI;       % seconds
hrTbl.CongruentMaxChangePPI         = CongruentMaxChangePPI;         % seconds
hrTbl.AllTrialsMaxChangePPI         = AllTrialsMaxChangePPI;         % seconds

hrTbl.baselinePPI            = baselinePPI;
hrTbl.incongruentBaselinePPI = incongruentBaselinePPI;
hrTbl.incongruentResponsePPI = incongruentResponsePPI;

% Optional: drop rows with missing ID (shouldn’t happen after shared filter)
hrTbl = hrTbl(~isnan(hrTbl.ID), :);

% ----- Write CSV
outPath = fullfile(options.paths.DBExport, 'PCNS_HRData.csv');
writetable(hrTbl, outPath);
fprintf('Wrote HR data: %s (%d rows)\n', outPath, height(hrTbl));
end


function shared = getSharedPCNSData(options)
% Returns a table with one row per included participant containing shared fields:
% ID, group, sex, age, education, FSIQ, meds_chlor, valid_cfacei,
% absInteroThreshold, absExteroThreshold, panss, panssPositive, panssNegative.

% --- Load REDCap
file = dir(fullfile(options.data.DBFileName));
assert(~isempty(file), 'REDCap export not found in %s', options.paths.DBExport);
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