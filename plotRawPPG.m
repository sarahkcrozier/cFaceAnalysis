function plotRawPPG(participantID)

options = specifyOptions;
dataDir  = options.paths.data;

    % 1. Format the Participant ID to 3 digits (e.g., 86 -> '086')
    if isnumeric(participantID)
        IDstring = sprintf('%03d', participantID);
    else
        IDstring = sprintf('%03d', str2double(participantID));
    end

   % 2. Define the path to the 'beh' folder
    % Structure: options.paths.data/PCNS_XXX_BL/beh/
    behFolder = fullfile(options.paths.data, ['PCNS_', IDstring, '_BL'], 'beh');
    
    % 3. Find the subfolder matching 'cface*MH*'
    subfolderPattern = fullfile(behFolder, 'cface*MH*');
    subfolderDir = dir(subfolderPattern);
    
    if isempty(subfolderDir)
        warning('No subfolder matching "cface*MH*" found in %s', behFolder);
        return;
    end
    
    % Use the first matching folder found
    actualSubfolder = fullfile(subfolderDir(1).folder, subfolderDir(1).name);

    % 4. Search for the PPG file ending in 'ppg.csv' inside that subfolder
    ppgFileDir = dir(fullfile(actualSubfolder, '*ppg.csv'));
    
    if isempty(ppgFileDir)
        warning('No PPG file found in %s', actualSubfolder);
        return;
    end
    
    filename = fullfile(ppgFileDir(1).folder, ppgFileDir(1).name);

    % 5. Read the data from the CSV file
    ppgOpts = detectImportOptions(filename);
    ppgOpts.SelectedVariableNames = {'time','PPG'}; 
    data = readtable(filename, ppgOpts);
    
    % 6. Extract the columns
    timeArr = data.time;
    ppgArr  = data.PPG;
    
    % 7. Create the plot
    figure('Color', 'w', 'Name', ['Raw PPG: Participant ' IDstring]);
    plot(timeArr, ppgArr, 'Color', [0 0.4470 0.7410], 'LineWidth', 0.5);
    
    % 8. Formatting
    xlabel('Time (seconds)');
    ylabel('PPG Amplitude (Raw)');
    title(['Raw PPG Signal - Participant ' IDstring]);
    grid on;
    axis tight;
    zoom on;
    
    fprintf('Successfully plotted raw PPG from: %s\n', filename);
end