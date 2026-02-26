function options = specifyOptions

%% SPECIFY Options
% function that includes paths, file names, participant IDs and other (hard coded) specifications

%% 

options.study.acronym = 'PCNS';
options.study.tasks = {'cFace','FF1','HBD'};

options.paths.workingDir = pwd;
options.paths.analysis   = ['/Users/yamaan/Projects/',options.study.acronym,filesep,options.study.tasks{1},'Analysis',filesep]; options.paths.data     = ['/Volumes/PCNS/Data/Data_raw/together'];
% options.paths.data       = ['/Users/yamaan/Projects/',options.study.acronym,filesep,'Data/RawData',filesep]; %temp local data path for testing
options.paths.DBExport   = ['/Users/yamaan/Projects/',options.study.acronym,filesep,'Data/REDCapExport',filesep];
% options.paths.DBExport   = ['/Users/yamaan/Projects/',options.study.acronym,filesep,'Data/REDCapExportTEST',filesep]; %temp DBExport file that only includes two participant, for testing
options.paths.HBDExport   = ['/Users/yamaan/Projects/',options.study.acronym,filesep,'Data/HBDAnalysis',filesep];
options.paths.EDFtoASC    = ['/Users/yamaan/Projects/',options.study.acronym,filesep,'Data/cFaceData/EDFtoASC',filesep]; % the location of converted EDF files (manually converted to ASC)
options.paths.eyeDataPlots   = [options.paths.analysis,'eyeDataPlots',filesep]; % the location of converted EDF files (first manually converted to ASC)
options.paths.plots      = [options.paths.analysis,'Plots',filesep];
options.paths.preprocessedEyeData = ['/Users/yamaan/Projects/',options.study.acronym,filesep,'Data/cFaceData/pupils_cface1',filesep]; % Anna B's preprocessed data
options.paths.eyeData = ['/Users/yamaan/Projects/',options.study.acronym,filesep,'Data/cFaceData/eyeData',filesep]; % save processed eye data here
options.paths.cFaceData = ['/Users/yamaan/Projects/',options.study.acronym,filesep,'Data/cFaceData',filesep]; % save processed cFace data here
options.paths.HRdata = ['/Users/yamaan/Projects/',options.study.acronym,filesep,'Data/cFaceData/HRdata',filesep]; % save processed cFace data here
options.paths.trialTimings = ['/Users/yamaan/Projects/',options.study.acronym,filesep,'Data/cFaceData/TrialTimings',filesep]; % save processed cFace trial time data here



options.data.pcnsDataTable   = ['/Users/yamaan/Projects/',options.study.acronym,filesep,'Data/REDCapExport/pcnsDataTable.csv'];
end
