function processCFaceData

%% SPECIFY Options
% function that includes paths, file names, participant IDs and other (hard coded) specifications
options = specifyOptions;


%% GET and ORGANIZE QUESTIONNAIRE data from REDCap export
getPrescreeningData
getOnlineQuestData


%% GET and ORGANIZE META DATA from REDCap export

% specify excluded and included datasets based on REDCap protocol entries

%% GET TASK BEHAVIOR data from mat files

% task responses

% task questionnaire

%% GET and ORGANIZE PHYSIO data

% get and collate PPU data


% get and collate pupil data


% get and collate EMG data


%% CREATE HYPOTHESES CSV filed


end