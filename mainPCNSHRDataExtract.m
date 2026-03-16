function mainPCNSHRDataExtract
% Build HR-based dataset and write CSV, including shared REDCap/HBD/PANSS columns,
% HR summary measures, PeakPPG metrics, and PPI summary metrics from cFaceHR.

options = specifyOptions();  % must return .paths.DBExport (and others you use)

% ----- Get shared participant rows (with inclusion criteria applied)
shared = getSharedPCNSData(options);    % table with ID and shared columns
N      = height(shared);

%% Initualize data rows
% ----- Preallocate HR vars
baselineHR            = NaN(N,1); % HR during baseline
postTaskHR            = NaN(N,1); % HR after task finished
incongruentHRaverage  = NaN(N,1); % HR averaged across all incongruent trials (used in paper)
congruentHRaverage    = NaN(N,1); % HR averaged across all congruent trials (used in paper)

% ----- PeakPPG vars 
baselinePeakPPG       = NaN(N,1); % maximum PPG value during baseline
incongruentPeakPPG    = NaN(N,1); % maximum PPG value during incongruent trials
congruentPeakPPG      = NaN(N,1); % maximum PPG value during congruent trials

% ----- New PPI summary metrics
IncongruentDiffPPI            = NaN(N,1);   % s average of 3 peaks post outcome start - average of 3 peaks before outcome start
CongruentDiffPPI              = NaN(N,1);   % s average of 3 peaks post outcome start - average of 3 peaks before outcome start
AllTrialsDiffPPI              = NaN(N,1);   % s average of 3 peaks post outcome start - average of 3 peaks before outcome start
IncongruentPercentageDiffPPI  = NaN(N,1);   % %  see above as percentage
CongruentPercentageDiffPPI    = NaN(N,1);   % %
AllTrialsPercentageDiffPPI    = NaN(N,1);   % %
IncongruentMaxChangePPI       = NaN(N,1);   % s minimum interval - average of 3 peaks before outcome start
CongruentMaxChangePPI         = NaN(N,1);   % s minimum interval - average of 3 peaks before outcome start
AllTrialsMaxChangePPI         = NaN(N,1);   % s minimum interval - average of 3 peaks before outcome start
baselinePPI                   = NaN(N,1);   % s 
incongruentBaselinePPI        = NaN(N,1);   % s
incongruentResponsePPI        = NaN(N,1);   % s

%% get data
% ----- Fill metrics per participant
for i = 1:N
    currentID = shared.ID(i);
    % try
        HR = cFaceHR(currentID);

        % HR measures
        baselineHR(i)           = HR.baseline;
        postTaskHR(i)           = HR.postTask;
        incongruentHRaverage(i) = HR.incongruentAverage;
        congruentHRaverage(i)   = HR.congruentAverage;

        % PeakPPG measures (support old/new field names)
        if     isfield(HR,'baselineHighpassedPeakPPG'),      baselinePeakPPG(i)    = HR.baselineHighpassedPeakPPG;
        elseif isfield(HR,'baselinePeakPPG'),                baselinePeakPPG(i)    = HR.baselinePeakPPG;
        end
        if     isfield(HR,'incongruentHighpassedPeakPPG'),   incongruentPeakPPG(i) = HR.incongruentHighpassedPeakPPG;
        elseif isfield(HR,'incongruentPeakPPG'),             incongruentPeakPPG(i) = HR.incongruentPeakPPG;
        end
        if     isfield(HR,'congruentHighpassedPeakPPG'),     congruentPeakPPG(i)   = HR.congruentHighpassedPeakPPG;
        elseif isfield(HR,'congruentPeakPPG'),               congruentPeakPPG(i)   = HR.congruentPeakPPG;
        end

        % New PPI summary metrics (flat fields created in cFaceHR)
        if isfield(HR,'IncongruentDiffPPI'),           IncongruentDiffPPI(i)           = HR.IncongruentDiffPPI;           end
        if isfield(HR,'CongruentDiffPPI'),             CongruentDiffPPI(i)             = HR.CongruentDiffPPI;             end
        if isfield(HR,'AllTrialsDiffPPI'),             AllTrialsDiffPPI(i)             = HR.AllTrialsDiffPPI;             end

        if isfield(HR,'IncongruentPercentageDiffPPI'), IncongruentPercentageDiffPPI(i) = HR.IncongruentPercentageDiffPPI; end
        if isfield(HR,'CongruentPercentageDiffPPI'),   CongruentPercentageDiffPPI(i)   = HR.CongruentPercentageDiffPPI;   end
        if isfield(HR,'AllTrialsPercentageDiffPPI'),   AllTrialsPercentageDiffPPI(i)   = HR.AllTrialsPercentageDiffPPI;   end

        if isfield(HR,'IncongruentMaxChangePPI'),      IncongruentMaxChangePPI(i)      = HR.IncongruentMaxChangePPI;      end
        if isfield(HR,'CongruentMaxChangePPI'),        CongruentMaxChangePPI(i)        = HR.CongruentMaxChangePPI;        end
        if isfield(HR,'AllTrialsMaxChangePPI'),        AllTrialsMaxChangePPI(i)        = HR.AllTrialsMaxChangePPI;        end

       
        % ---- Baseline/response PPI exports (seconds) ----
        % baselinePPI (mean PPI during baseline beats)
        if isfield(HR,'baselinePPI'), baselinePPI(i) = HR.baselinePPI; end

        % incongruent baseline/response PPI
        % (handle both spellings of the response field to be safe)
        if isfield(HR,'incongruentBaselinePPI')
            incongruentBaselinePPI(i) = HR.incongruentBaselinePPI;
        end
        if isfield(HR,'incongruentResponsePPI')
            incongruentResponsePPI(i) = HR.incongruentResponsePPI;
        elseif isfield(HR,'incongruentrespontPPI')  % fallback if earlier code used this name
            incongruentResponsePPI(i) = HR.incongruentrespontPPI;
        end


        
        % Optional concise per-participant print (no rounding enforced)
        fprintf(['ID %d | base=%.6f bpm, post=%.6f bpm | incHR=%.6f, congHR=%.6f | ' ...
                 'PeakPPG: base=%.6f, inc=%.6f, cong=%.6f | ' ...
                 'ΔPPI(s): inc=%.6f, cong=%.6f, all=%.6f | %%ΔPPI: inc=%.6f, cong=%.6f, all=%.6f | ' ...
                 'maxΔPPI(s): inc=%.6f, cong=%.6f, all=%.6f\n'], ...
            currentID, baselineHR(i), postTaskHR(i), ...
            incongruentHRaverage(i), congruentHRaverage(i), ...
            baselinePeakPPG(i), incongruentPeakPPG(i), congruentPeakPPG(i), ...
            IncongruentDiffPPI(i), CongruentDiffPPI(i), AllTrialsDiffPPI(i), ...
            IncongruentPercentageDiffPPI(i), CongruentPercentageDiffPPI(i), AllTrialsPercentageDiffPPI(i), ...
            IncongruentMaxChangePPI(i), CongruentMaxChangePPI(i), AllTrialsMaxChangePPI(i));
    % catch ME
    %     warning('cFaceHR failed for ID %d: %s', currentID, ME.message);
    % end
end

%% create table
% ----- Assemble output table (shared + HR + PeakPPG + PPI summaries)
hrTbl = shared;

% HR
hrTbl.baselineHR            = baselineHR;
hrTbl.postTaskHR            = postTaskHR;
hrTbl.incongruentHRaverage  = incongruentHRaverage;
hrTbl.congruentHRaverage    = congruentHRaverage;

% Peak PPG
hrTbl.baselinePeakPPG       = baselinePeakPPG;
hrTbl.incongruentPeakPPG    = incongruentPeakPPG;
hrTbl.congruentPeakPPG      = congruentPeakPPG;

% PPI summary metrics
hrTbl.IncongruentDiffPPI            = IncongruentDiffPPI;            % seconds
hrTbl.CongruentDiffPPI              = CongruentDiffPPI;              % seconds
hrTbl.AllTrialsDiffPPI              = AllTrialsDiffPPI;              % seconds
hrTbl.IncongruentPercentageDiffPPI  = IncongruentPercentageDiffPPI;  % percent
hrTbl.CongruentPercentageDiffPPI    = CongruentPercentageDiffPPI;    % percent
hrTbl.AllTrialsPercentageDiffPPI    = AllTrialsPercentageDiffPPI;    % percent
hrTbl.IncongruentMaxChangePPI       = IncongruentMaxChangePPI;       % seconds
hrTbl.CongruentMaxChangePPI         = CongruentMaxChangePPI;         % seconds
hrTbl.AllTrialsMaxChangePPI         = AllTrialsMaxChangePPI;         % seconds

hrTbl.baselinePPI            = baselinePPI;
hrTbl.incongruentBaselinePPI = incongruentBaselinePPI;
hrTbl.incongruentResponsePPI = incongruentResponsePPI;

% Optional: drop rows with missing ID (shouldn’t happen after shared filter)
hrTbl = hrTbl(~isnan(hrTbl.ID), :);

% ----- Write CSV
outPath = fullfile(options.paths.HRdata, 'PCNS_HRData.csv');
writetable(hrTbl, outPath);
fprintf('Wrote HR data: %s (%d rows)\n', outPath, height(hrTbl));
end
