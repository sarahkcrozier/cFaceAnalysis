function pcnsDataExtract

options = specifyOptions;

%% LOAD REDCap datafile
file = dir([options.paths.DBExport,'PCNS_RedCap_Export.csv']);
data = readtable([options.paths.DBExport,file.name]);
minID = min(data.record_id);
maxID = max(data.record_id);
IDspan = maxID - minID + 1;

record_id   = NaN(IDspan,1);
group       = NaN(IDspan,1);
sex         = NaN(IDspan,1);
age_years   = NaN(IDspan,1);
FSIQ   = NaN(IDspan,1);
edu_cat   = NaN(IDspan,1);
meds_chlor   = NaN(IDspan,1);
valid_cfacei   = NaN(IDspan,1);
panss       = NaN(IDspan,1);
panssPositive = NaN(IDspan,1);
panssNegative = NaN(IDspan,1);
baselineHR = NaN(IDspan,1);
postTaskHR = NaN(IDspan,1);
congruentHRaverage = NaN(IDspan,1); 
incongruentHRaverage    = NaN(IDspan,1);
baselinePupil           = NaN(IDspan,1);
averagePupil            = NaN(IDspan,1);
incongruentPupilAverage = NaN(IDspan,1);
congruentPupilAverage   = NaN(IDspan,1);
eyeSide = repmat(' ', IDspan, 1);

%%NEW 6 Aug 2025
%% Load HeartBeat Discrimination Data 
HBDfile = dir([options.paths.HBDExport,'outcomes_myhrd.csv']);
HBDdata = readtable([options.paths.HBDExport,HBDfile.name]);

absInteroThreshold   = NaN(IDspan,1);
absExteroThreshold   = NaN(IDspan,1);

%% EXTRACT data

% loop through REDCap IDs
% get data
% write table


%% 
rowIdx = 1;

for n = minID:maxID
    % Find all rows for this REDCap record_id
    id_rows = find(data.record_id == n);
    if isempty(id_rows), continue; end

    % Locate first occurrence of each instrument for this subject
    details_row  = [];
    demogr_row   = [];
    clinical_row = [];

    k = find(strcmp(data.redcap_repeat_instrument(id_rows), 'participant_details'), 1, 'first');
    if ~isempty(k), details_row = id_rows(k); end

    k = find(strcmp(data.redcap_repeat_instrument(id_rows), 'demographics'), 1, 'first');
    if ~isempty(k), demogr_row = id_rows(k); end

    k = find(strcmp(data.redcap_repeat_instrument(id_rows), 'clinical'), 1, 'first');
    if ~isempty(k), clinical_row = id_rows(k); end

    % Must have demographics & details rows to proceed
    if isempty(demogr_row) || isempty(details_row), continue; end

    % ----- Exclusions (demographics row) -----
    if data.valid_any(demogr_row) ~= 1,   continue; end
    if data.valid_cfacei(demogr_row) ~= 1, continue; end
    if data.fsiq2(demogr_row) < 80,        continue; end
    if data.pilotorreal(demogr_row) ~= 2,  continue; end

    % =========================================================
    % DEMOGRAPHICS / BASICS (define currentID here)
    % =========================================================
    currentID = data.record_id(details_row);

    record_id(rowIdx,:)  = currentID;
    age_years(rowIdx,:)  = data.age_years(details_row);
    group(rowIdx,:)      = data.group(demogr_row);
    sex(rowIdx,:)        = data.sex(demogr_row);
    edu_cat(rowIdx,:)    = data.edu_cat(demogr_row);
    FSIQ(rowIdx,:)       = data.fsiq2(demogr_row);
    meds_chlor(rowIdx,:) = data.meds_chlor(demogr_row);
    if isempty(meds_chlor(rowIdx,:)) || isnan(meds_chlor(rowIdx,:))
        meds_chlor(rowIdx,:) = 0;
    end
    valid_cfacei(rowIdx,:) = data.valid_cfacei(demogr_row);

    % =========================================================
    % HEART RATE (use currentID)
    % =========================================================
    HR = cFaceHR(currentID);
    baselineHR(rowIdx,:)          = HR.baseline;
    postTaskHR(rowIdx,:)          = HR.postTask;
    incongruentHRaverage(rowIdx,:)= HR.incongruentAverage;
    congruentHRaverage(rowIdx,:)  = HR.congruentAverage;

    % =========================================================
    % PUPIL METRICS (use currentID; keep NaNs if missing)
    % =========================================================
    try
        % Preferred: cFacePupilArea returns [eyeDataTbl, cFacePupil]
        [~, cFacePupil] = cFacePupilArea(currentID);

        if isfield(cFacePupil,'baseline'),             baselinePupil(rowIdx,:)            = cFacePupil.baseline;           end
        if isfield(cFacePupil,'eyeSide'),              eyeSide(rowIdx,:)                  = cFacePupil.eyeSide;           end
        if isfield(cFacePupil,'peakAverage'),          averagePupil(rowIdx,:)             = cFacePupil.peakAverage;        end
        if isfield(cFacePupil,'incongruentAverage'),   incongruentPupilAverage(rowIdx,:)  = cFacePupil.incongruentAverage; end
        if isfield(cFacePupil,'congruentAverage'),     congruentPupilAverage(rowIdx,:)    = cFacePupil.congruentAverage;   end
    catch ME
        warning('cFacePupilArea failed for ID %d: %s', currentID, ME.message);
        % leave NaNs in the pupil arrays for this participant
    end

    % =========================================================
    % HBD outcomes (if present)
    % =========================================================
    hbd_match_idx = find(HBDdata.record_id == currentID, 1, 'first');
    if ~isempty(hbd_match_idx)
        absInteroThreshold(rowIdx,:) = abs(HBDdata.hrd_Intero_threshold_Bay(hbd_match_idx));
        absExteroThreshold(rowIdx,:) = abs(HBDdata.hrd_Extero_threshold_Bay(hbd_match_idx));
    end

    % =========================================================
    % PANSS sums (clinical row, if present)
    % =========================================================
    if ~isempty(clinical_row)
        sum_panssPositive = 0;
        sum_panssNegative = 0;
        sum_panssGeneral  = 0;

        for i = 1:7
            sum_panssPositive = sum_panssPositive + data.(['panss_p', num2str(i)])(clinical_row);
            sum_panssNegative = sum_panssNegative + data.(['panss_n', num2str(i)])(clinical_row);
        end
        for i = 1:16
            sum_panssGeneral  = sum_panssGeneral  + data.(['panss_g', num2str(i)])(clinical_row);
        end

        panss(rowIdx,:)         = sum_panssPositive + sum_panssNegative + sum_panssGeneral;
        panssPositive(rowIdx,:) = sum_panssPositive;
        panssNegative(rowIdx,:) = sum_panssNegative;
    end

    % Fill zeros if any of the panss totals are empty
    if isempty(panss(rowIdx,:))         || isnan(panss(rowIdx,:)),         panss(rowIdx,:)         = 0; end
    if isempty(panssPositive(rowIdx,:)) || isnan(panssPositive(rowIdx,:)), panssPositive(rowIdx,:) = 0; end
    if isempty(panssNegative(rowIdx,:)) || isnan(panssNegative(rowIdx,:)), panssNegative(rowIdx,:) = 0; end

    % ----- advance row AFTER finishing the participant -----
    rowIdx = rowIdx + 1;
end


mask = ~isnan(record_id);

pcnsDataTable = table( ...
    record_id(mask), group(mask), age_years(mask), sex(mask), edu_cat(mask), FSIQ(mask), meds_chlor(mask), ...
    baselineHR(mask), postTaskHR(mask), incongruentHRaverage(mask), congruentHRaverage(mask), ...
    baselinePupil(mask),eyeSide(mask), averagePupil(mask), incongruentPupilAverage(mask), congruentPupilAverage(mask), ...
    absInteroThreshold(mask), absExteroThreshold(mask), panss(mask), panssPositive(mask), panssNegative(mask), ...
    'VariableNames', ...
    {'ID','group','age','sex','education','FSIQ','Chlorpromazine equivalents (mg)', ...
     'baselineHR','postTaskHR','incongruentHRaverage','congruentHRaverage', ...
     'baselinePupil','eyeSide','averagePupil','incongruentPupilAverage','congruentPupilAverage', ...
     'absInteroThreshold','absExteroThreshold','panss','panssPositive','panssNegative'} ...
);

writetable(pcnsDataTable,[options.paths.DBExport,'pcnsDataTable.csv'])
end

%{

%% GET and ORGANIZE participant data from REDCap export

    Participant ID          = record_id (corresponds with PCNS_ID_BL in MRI folders
                            = starts at 0 in HRD analysis output
                            outcomes_myhrd_reduced

    **DEMOGRAPHICS:
    Age                     = age_years
    Sex                     = sex (1 male, 2 female)
    FSIQ WASI II            = fsiq2
    Education               = education (1, Didn't finish HS; 2, High
    school; 3, Non-university qualification; 4, Bachelor's; 5, Master's; 6, Doctorate)


    Psych medication        = meds_psych (text)
    Diagnosis               = dx_dsm (0==none?, 1 schizophrenia, 2 schizoaffective, 3 bipolar, 4 MDD, 5 delusional disorder, 6 drug-induced psychosis)



    **EXCLUSIONS: 
    Exclusion MH            = ex1hc_mental (If control, 0 == No history of
                              mental health issues)
    Exclusion TBI/neuro     = ex2_neuro (0 == no; 1 == yes)
    Exclusion SUD           = ex1_substance (0 == no)
    Not pilot               = pilotreal (1 == pilot, 2 == study)
    Completed (all?)details = participant_details_complete (2 = complete)
    Attended session        = attended (1 == attended, 2 or nothing == did
                              not)
    Include in analysis     = valid_any (1 = include, others had too much missing data/tasks etc)

    completed cface?        = valid_cfacei (1 == true)


    **MAIN ANALYSIS VARIABLES
    Group                   = group (control == 1, psychosis == 2)
    Pupil                   = pupil average (x second window following
    incongruent trials?)

    **MAIN COVARIATES
    baselineHR
    baselinePupil
    
   
%}


%% loop through pupil records - identify incongruent trials
% calculate average - save to participant row

%% loop through ppg records - identify incongruent trials
% calculate average - save to participant row
%%

%%

