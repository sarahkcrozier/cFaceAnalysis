function [deletePeaks, addPeaks] = correctManuallyDetectedPeaks(participantID)

if strcmp(participantID,'052')
    deletePeaks = [9,11,13,15,17]';
    addPeaks    = 137;

elseif strcmp(participantID,'082')
    deletePeaks = [403:2:415,419]';
    addPeaks    = [419:2:425]';

elseif strcmp(participantID,'083')
    deletePeaks = [];
    addPeaks = [4,6,38,48,50,72,106,115,135]; % of NotGoodPeals

elseif strcmp(participantID,'109')
    deletePeaks = [];
    addPeaks = [1:9,11,12,13]; % of NotGoodPeals

elseif strcmp(participantID,'113')
    deletePeaks = [];
    addPeaks = [1:14,16:23]; % of NotGoodPeals

elseif strcmp(participantID,'117')
    deletePeaks = [];
    addPeaks = 1;

elseif strcmp(participantID,'120')
    deletePeaks = [13,170,195,247,341, 534, 537];
    addPeaks = []; % of NotGoodPeals

elseif strcmp(participantID,'128')
    deletePeaks = [1,19];
    addPeaks = []; % of NotGoodPeals

elseif strcmp(participantID,'130')
    deletePeaks = [];
    addPeaks = [1,3,4,5]; % of NotGoodPeals
elseif strcmp(participantID,'134')
    deletePeaks = 1;
    addPeaks = 29; % of NotGoodPeals
else
    deletePeaks = [];
    addPeaks = []; % of NotGoodPeals

end

