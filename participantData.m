%{

    Participant ID          = record_id (corresponds with PCNS_ID_BL in MRI folders

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

options.pcns.record_id

%% options function

function options = specifyOptions
%% 

options.study.acronym = 'PCNS';
options.study.tasks = {'cFace','FF1','HBD'};

options.paths.workingDir = pwd;
options.paths.analysis   = ['/Users/yamaan/Projects/',options.study.acronym,filesep,options.study.tasks{1},'Analysis',filesep];
% options.paths.data     = ['smb://macuncle.newcastle.edu.au/entities/research/NEWYSNG/PCNS'];
options.paths.data       = [options.paths.analysis,'TestData',filesep]; %temp local data path for testing
options.paths.DBExport   = [options.paths.analysis,'REDCapExport',filesep];
options.paths.plots      = [options.paths.analysis,'Plots',filesep];

options = getQuestionnaireDetails(options);

%% options function end


% for n = 1:max(data.record_id)

%% loop through ppg records and plot data


%% identify incongruent trials - Jayson's Python code in PCNS?

%% function to calculate average 

