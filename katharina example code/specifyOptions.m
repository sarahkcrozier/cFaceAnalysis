function options = specifyOptions

options.study.tasks = {'SAP','SAPC','AAA'};
options.study.acronym = 'SAPS';

options.paths.workingDir = pwd;
options.paths.toolboxDir = ['..',filesep,'..',filesep,'Toolboxes',filesep,'tapas-6.0.1',filesep];
options.paths.data       = ['/Volumes/Samsung_T5/SNG/projects/',options.study.acronym,filesep,'data',filesep];
options.paths.results    = [options.paths.data,'results',filesep];
options.paths.DBExport   = [options.paths.data,'REDCapExport',filesep];
options.paths.questData  = [options.paths.data,'questionnaires',filesep];
options.paths.figDir     = [options.paths.results,'figures',filesep];

options = getQuestionnaireDetails(options);

participants  = dir(options.paths.data);
options.PPIDs = string(nan(numel(participants),1));

options.task(1).inputs = load('/Users/kwellste/projects/SEPAB/tasks/social_affective_prediction_task/task/+eventCreator/input_sequence.csv');
options.task(2).inputs = load('/Users/kwellste/projects/SEPAB/tasks/social_affective_prediction_task/control_task/+eventCreator/input_sequence.csv');
options.task(3).inputs = load('/Users/kwellste/projects/SEPAB/tasks/approach_avoid_task/task/+eventCreator/input_sequence.csv');

for i = 1:numel(participants)
    if ~startsWith(participants(i).name,'.')
        options.PPIDs(i) = participants(i).name;
    end
end

options.PPIDs = rmmissing(options.PPIDs);
options.PIDs  = extract(options.PPIDs,digitsPattern);
options.PIDs  = char(options.PIDs);
options.nParticipants = numel(options.PPIDs);

options.dataFilePrefix = 'SNG_';
options.dataFileSuffix = '_behav_dataFile.mat';

% % optimization algorithm
% addpath(genpath(options.toolboxDir));
% options.hgf.opt_config = eval('tapas_quasinewton_optim_config');
% 
% % seed for random number generator
% options.rng.idx        = 1; % Set counter for random number states
% options.rng.settings   = rng(123, 'twister');
% options.rng.nRandInit  = 100;
% 
% %% SPECIFY MODELS and related functions
% options.setupModels       = [];
% options.model.space       = {'HGF_3L','HGF_2L','RW'}; % all models in modelspace
% options.model.prc         = {'tapas_ehgf_binary','tapas_ehgf_binary','tapas_rw_binary'};
% options.model.prc_config  = {'tapas_ehgf_binary_config_3L','tapas_ehgf_binary_config_2L','tapas_rw_binary_config'};
% options.model.obs	      = {'tapas_unitsq_sgm'};
% options.model.obs_config  = {'tapas_unitsq_sgm_config'};
% options.model.opt_config  = {'tapas_quasinewton_optim_config'};
% options.plot(1).plot_fits = @tapas_ehgf_binary_plotTraj;
% options.plot(2).plot_fits = @tapas_ehgf_binary_plotTraj;
% options.plot(3).plot_fits = @tapas_rw_binary_plotTraj;
end