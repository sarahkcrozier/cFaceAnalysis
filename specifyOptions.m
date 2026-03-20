function options = specifyOptions

%% SPECIFY Options
% function that includes paths, file names, and other (hard coded) specifications
%
% OUT:    options: struct, includes paths, file nams, etc.
%
%
%% 

options.study.acronym = 'PCNS';
options.study.DBName  = 'CogEmotPsych';
options.data.DBFileName = []; % will be filled if a file is found staring with options.study.DBName
options.study.tasks = {'cFace','FF1','HBD'};

%% SPECIFY data settings
options.data.baselineLength = 10; % in sec.
options.data.pupilTrials = 80;

options.paths.workingDir = pwd;
[~, user] = system('whoami');
user=strrep(user, sprintf('\n'), '');

%% SPECIFY PATHS
% add your own directory specifications here to run this code

if strcmp(user,'kwellste')
    options.paths.data    = ['/Volumes/Samsung_T5/SNG/projects/',options.study.acronym ,'/IncongruentFaces/physio_paper/data/'];
    options.paths.rawData = [options.paths.data,'raw/'];
    options.paths.plots   = [options.paths.data,'plots/'];
    options.paths.data2   = [];

elseif strcmp(user,'yamaan')
    options.paths.rawData = '/Volumes/PCNS/Data/Data_raw/together/';
    options.paths.data    = ['/Users/yamaan/Projects/',options.study.acronym ,'/Data/'];
    % options.paths.data       = ['/Users/yamaan/Projects/',options.study.acronym,filesep,'Data/RawData',filesep]; %temp local data path for testing
    options.paths.data2   = [options.paths.data,'cFaceData/']; % the logic of why there is this subfolder is not clear to me. I just added this here for now to make that work
    options.paths.plots   = ['/Users/yamaan/Projects/',options.study.acronym,'/cFaceAnalysis/'];

elseif strcmp(user,'sarah') % TO COMPLETE for MAXXI with user Sarah
    options.paths.rawData = '/Volumes/NEWYSNG/PCNS/Data/Data_raw/together';
    options.paths.data    = '/Volumes/Scratch/Sarah/PCNS/Data/';
    options.paths.data2   = [options.paths.data,'cFaceData/'];
    options.paths.plots   = '/Volumes/Scratch/Sarah/PCNS/cFaceAnalysis/';
end

% Paths to export of REDCap database file
options.paths.DBExport     = [options.paths.data,'REDCapExport',filesep];

if ~exist(options.paths.DBExport,'dir')==0
    d = dir(options.paths.DBExport);
    for i = 1:numel(d)
        if startsWith(d(i).name,options.study.DBName)
           redcapfound(i) = 1;
           options.data.DBFileName = [options.paths.DBExport,d(i).name];
        else
            redcapfound(i) = 0;
        end
    end
    if sum(redcapfound)== 0
        disp(['Database folder found but export file not found... . Please put file in ',options.paths.DBExport, 'and run this function again!']);
    end
    
else
    mkdir(options.paths.DBExport);
    disp(['Database folder newly created, check location of DBExport and move to ',options.paths.DBExport, '!']);
end

% Paths to export of REDCap database file
options.paths.HBDExport    = [options.paths.data,'HBDAnalysis',filesep];
if ~exist(options.paths.HBDExport,'dir')==0
    mkdir(options.paths.HBDExport)
end


if ~isempty(options.paths.data2) % only needed because data is organized differently
    options.paths.eyeASCFiles  = [options.paths.data2,'EDFtoASC',filesep];   % the location of converted EDF files (manually converted to ASC)
    options.paths.eyeData      = [options.paths.data2,'eyeData',filesep];      % save processed eye data here
    options.paths.HRdata       = [options.paths.data2,'HRdata',filesep];
    options.paths.trialTimings = [options.paths.data2,'TrialTimings',filesep];
else
    options.paths.eyeASCFiles  = [options.paths.rawData,'EDFtoASC',filesep];  % the location of converted EDF files (manually converted to ASC)
    options.paths.eyeData      = [options.paths.data,'eyeData',filesep];
    options.paths.HRdata       = [options.paths.data,'HRdata',filesep];
    options.paths.trialTimings = [options.paths.data,'TrialTimings',filesep];
end

options.paths.eyeDataPlots = [options.paths.plots,'eyeDataPlots',filesep]; % where to save eye data plots
options.paths.plots        = [options.paths.plots,'HRPlots',filesep];   

%% SPECIFY FILE NAMES

options.data.pcnsDataTable   = [options.paths.DBExport,'pcnsDataTable.csv'];   % whats that?
options.data.hbdOutcomesFile = [options.paths.HBDExport,'outcomes_myhrd.csv']; % whats that?
options.data.HRDiaryName     = 'output_HR_cface.txt';
options.data.pupilDiaryName  = 'output_pupils_cface.txt';
options.data.nPannsPItems = 7;
options.data.nPannsNItems = 7;
options.data.nPannsGItems = 16;

% Check if files needed for analyses are in the folder structure 
% ??????? Where or how is this file generated????????
HBDfile = dir(fullfile(options.data.hbdOutcomesFile));
assert(~isempty(HBDfile), 'HBD export not found in %s', options.paths.HBDExport);

file = dir(fullfile(options.data.DBFileName));
assert(~isempty(file), 'REDCap export not found in %s', options.paths.DBExport);

end

