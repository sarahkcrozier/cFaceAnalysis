function convertPcnsDataTableToRepeatedMeasures

options = specifyOptions;

% Read original data table and keep only participants from group 2
originalData = readtable(options.data.pcnsDataTable);
originalData = originalData(originalData.group == 2, :);

% Calculate responseHRaverage
originalData.responseHRaverage = mean([originalData.congruentHRaverage, originalData.incongruentHRaverage], 2);

% Create long-format variables for Baseline and ResponseAverage
numParticipants = height(originalData);
Subject = repelem((1:numParticipants)', 2);
Condition = repmat({'1'; '2'}, numParticipants, 1);
HeartRate = reshape([originalData.baselineHR, originalData.responseHRaverage]', [], 1);
Medication = repelem(originalData.('ChlorpromazineEquivalents_mg_'), 2);
Group = repelem(originalData.group, 2);

% Create long-format table
longFormatTable = table(Subject, Condition, HeartRate, Medication, Group);

% Export the new table to DBExport directory
savePath = fullfile(options.paths.DBExport, 'pcnsData_LongFormat.csv');
writetable(longFormatTable, savePath);

% Display first few rows of the new table
disp(longFormatTable(1:12,:));