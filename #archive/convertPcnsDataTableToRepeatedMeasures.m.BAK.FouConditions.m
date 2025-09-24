function convertPcnsDataTableToRepeatedMeasures

options = specifyOptions;

% Read the original data from CSV
originalData = readtable(options.data.pcnsDataTable);

% Calculate responseHRaverage
originalData.responseHRaverage = mean([originalData.congruentHRaverage, originalData.incongruentHRaverage], 2);

% Preallocate new long-format table columns
numParticipants = height(originalData);
Subject = repelem((1:numParticipants)', 4);
Condition = repmat({'1'; '2'; '3'; '4'}, numParticipants, 1);
HeartRate = zeros(numParticipants*4, 1);
Medication = repelem(originalData.('ChlorpromazineEquivalents_mg_'), 4);
Group = repelem(originalData.group, 4);

% Fill HeartRate data
for i = 1:numParticipants
    idx = (i-1)*4 + 1;
    HeartRate(idx)   = originalData.baselineHR(i);
    HeartRate(idx+1) = originalData.incongruentHRaverage(i);
    HeartRate(idx+2) = originalData.congruentHRaverage(i);
    HeartRate(idx+3) = originalData.responseHRaverage(i);
end

% Create long-format table
longFormatTable = table(Subject, Condition, HeartRate, Medication, Group);


% Export the new table to DBExport directory
savePath = fullfile(options.paths.DBExport, 'pcnsData_LongFormat.csv');
writetable(longFormatTable, savePath);

% Display first few rows of the new table
disp(longFormatTable(1:12,:));