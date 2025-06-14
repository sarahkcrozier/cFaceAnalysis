function options = specifyOptions

%% SPECIFY Options
% function that includes paths, file names, participant IDs and other (hard coded) specifications

%% 

options.study.acronym = 'PCNS';
options.study.tasks = {'cFace','FF1','HBD'};

options.paths.workingDir = pwd;
options.paths.analysis   = ['/Users/yamaan/Projects/',options.study.acronym,filesep,options.study.tasks{1},'Analysis',filesep];
% options.paths.data     = ['smb://macuncle.newcastle.edu.au/entities/research/NEWYSNG/PCNS'];
options.paths.data       = ['/Users/yamaan/Projects/',options.study.acronym,filesep,'Data/RawData',filesep]; %temp local data path for testing
options.paths.DBExport   = ['/Users/yamaan/Projects/',options.study.acronym,filesep,'Data/REDCapExport',filesep];
options.paths.plots      = [options.paths.analysis,'Plots',filesep];

end
