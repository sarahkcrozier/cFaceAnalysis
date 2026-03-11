function PCNSPupilDataExtract
% Build pupil-based dataset and write CSV, including shared REDCap/HBD/PANSS columns.

options = specifyOptions();  % must return .paths.DBExport and any others you use

% ----- Get shared participant rows (with inclusion criteria applied)
shared = getSharedPCNSData(options);    % table with ID and shared columns
N = height(shared);

% ----- Preallocate pupil vars
baselinePupil           = NaN(N,1);
averagePupil            = NaN(N,1);
incongruentPupilAverage = NaN(N,1);
congruentPupilAverage   = NaN(N,1);
eyeSide_cell            = repmat({''}, N, 1);   % store as 1-char strings ("R"/"L")

% ----- Fill pupil metrics
for i = 1:N
    currentID = shared.ID(i);
    % try
        % [~, cFacePupil] per your convention
        [~, cFacePupil] = cFacePupilArea(currentID);
        %% 

        if isfield(cFacePupil,'baseline'),             baselinePupil(i)            = cFacePupil.baseline;            end
        if isfield(cFacePupil,'peakAverage'),          averagePupil(i)             = cFacePupil.peakAverage;         end
        if isfield(cFacePupil,'incongruentAverage'),   incongruentPupilAverage(i)  = cFacePupil.incongruentAverage;  end
        if isfield(cFacePupil,'congruentAverage'),     congruentPupilAverage(i)    = cFacePupil.congruentAverage;    end
        if isfield(cFacePupil,'eyeSide')
            % coerce to 1-char string cell
            val = cFacePupil.eyeSide;
            if ischar(val) && ~isempty(val)
                eyeSide_cell{i} = val(1);
            elseif isstring(val) && strlength(val)>=1
                eyeSide_cell{i} = char(val(1));
            end
        end
    % catch ME
    %     warning('cFacePupilArea failed for ID %d: %s', currentID, ME.message);
    % end
end

% ----- Assemble pupil table (shared + pupil)
pupilTbl = shared;
pupilTbl.baselinePupil           = baselinePupil;
pupilTbl.eyeSide                 = eyeSide_cell;         % 1-char strings
pupilTbl.averagePupil            = averagePupil;
pupilTbl.incongruentPupilAverage = incongruentPupilAverage;
pupilTbl.congruentPupilAverage   = congruentPupilAverage;

% Optional: drop rows with missing ID (shouldn’t happen after shared filter)
pupilTbl = pupilTbl(~isnan(pupilTbl.ID), :);

% ----- Write CSV
outPath = fullfile(options.paths.eyeData, 'PCNS_PupilData.csv');
writetable(pupilTbl, outPath);
fprintf('Wrote pupil data: %s (%d rows)\n', outPath, height(pupilTbl));
end
