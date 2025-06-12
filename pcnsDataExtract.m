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
edu_cat   = NaN(IDspan,1);
meds_chlor   = NaN(IDspan,1);
panss       = NaN(IDspan,1);
panssPositive = NaN(IDspan,1);
panssNegative = NaN(IDspan,1);

%% EXTRACT data

% loop through REDCap IDs
% get data
% write table


%% NEED TO ADD:
rowIdx = 1;

for n = 1:max(data.record_id)
    id_rows   = find(data.record_id==n);
    if ~isempty(id_rows)
        details_row = intersect(id_rows,find(strcmp(data.redcap_repeat_instrument,'participant_details')));
        demogr_row = intersect(id_rows,find(strcmp(data.redcap_repeat_instrument,'demographics')));
        clinical_row = intersect(id_rows,find(strcmp(data.redcap_repeat_instrument,'clinical')));
        if numel(details_row)>1
            details_row = details_row(1);
        end
        if numel(demogr_row)>1
            demogr_row = demogr_row(1);
        end
        if numel(clinical_row)>1
            clinical_row = clinical_row(1);
        end

        %% omit exclusions: not pilot
        % note that this doesn't entirely work as this loop inlcudes the participant details row,
        % which does not have pilot info, hence adding those pilot ids
        % and age to the final table

        %% add more exclusions (see end of file)?
        if data.valid_any(demogr_row) ~= 1
            continue;
        end
        if rowIdx == 1
            record_id(rowIdx,:) = data.record_id(details_row);
            age_years(rowIdx,:) = data.age_years(details_row);
            group(rowIdx,:)    = data.group(demogr_row);
            sex(rowIdx,:)      = data.sex(demogr_row);
            edu_cat(rowIdx,:)  = data.edu_cat(demogr_row);
            meds_chlor(rowIdx,:)  = data.meds_chlor(demogr_row);

            %% GET and ORGANIZE PANSS data from REDCap export
            if ~isempty(clinical_row)
                sum_panssPositive = 0;
                sum_panssNegative = 0;
                sum_panssGeneral = 0;
                for i = 1:7
                    varname = ['panss_p', num2str(i)];
                    sum_panssPositive = sum_panssPositive + data.(varname)(clinical_row);
                end
                for i = 1:7
                    varname = ['panss_n', num2str(i)];
                    sum_panssNegative = sum_panssNegative + data.(varname)(clinical_row);
                end
                for i = 1:16
                    varname = ['panss_g', num2str(i)];
                    sum_panssGeneral = sum_panssGeneral + data.(varname)(clinical_row);
                end
                panss(rowIdx,:) = sum_panssPositive + sum_panssNegative + sum_panssGeneral;
                panssPositive(rowIdx,:) = sum_panssPositive;
                panssNegative(rowIdx,:) = sum_panssNegative;

            end
            % update index for the first time
            rowIdx = rowIdx+1;
        else
            record_id(rowIdx,:) = data.record_id(details_row);
            age_years(rowIdx,:) = data.age_years(details_row);
            group(rowIdx,:)    = data.group(demogr_row);
            sex(rowIdx,:)      = data.sex(demogr_row);
            edu_cat(rowIdx,:)  = data.edu_cat(demogr_row);
            meds_chlor(rowIdx,:)  = data.meds_chlor(demogr_row);
            %% GET and ORGANIZE PANSS data from REDCap export
            if ~isempty(clinical_row)
                sum_panssPositive = 0;
                sum_panssNegative = 0;
                sum_panssGeneral = 0;
                for i = 1:7
                    varname = ['panss_p', num2str(i)];
                    sum_panssPositive = sum_panssPositive + data.(varname)(clinical_row);
                end
                for i = 1:7
                    varname = ['panss_n', num2str(i)];
                    sum_panssNegative = sum_panssNegative + data.(varname)(clinical_row);
                end
                for i = 1:16
                    varname = ['panss_g', num2str(i)];
                    sum_panssGeneral = sum_panssGeneral + data.(varname)(clinical_row);
                end
                panss(rowIdx,:) = sum_panssPositive + sum_panssNegative + sum_panssGeneral;
                panssPositive(rowIdx,:) = sum_panssPositive;
                panssNegative(rowIdx,:) = sum_panssNegative;
            end
            rowIdx = rowIdx+1;
        end
    end
end

pcnsDataTable = table(record_id, group, age_years, sex, edu_cat, meds_chlor, panss, panssPositive, panssNegative, 'VariableNames',...
    {'ID','group', 'age','sex','education','Chlorpromazine equivalents (mg)','panss','panssPositive','panssNegative'});

deleteIds = find(isnan(pcnsDataTable.ID));
pcnsDataTable(deleteIds,:) = [];

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

