% compute_gni.m
% Complete implementation of Global Noise Index calculation
% Based on Christianson et al. (2015)

% 
% 
% compute_gni('D:\AI_Noise_QA\Pediatric-CT-SEG-0C49C1B7\05-09-2009-NA-CT-51909\30144.000000-CT-03779', ...
%            'D:\AI_Noise_QA');

function compute_gni(root_path, output_path)
    if nargin < 1
        error('Please provide path to CT scan directory or root directory containing multiple scans');
    end
    
    % Create results directory
    [parent_dir, ~] = fileparts(output_path);
    results_dir = fullfile(parent_dir, 'GNI_Results');
    if ~exist(results_dir, 'dir')
        mkdir(results_dir);
    end
    
    % Set GNI parameters
    params = struct();
    params.kernelSize = 9;
    params.tissueRange = [-300, 300];
    params.histogramBinSize = 0.1;
    params.smoothingKernelSize = 5;
    
    % Find CT directories
    ct_dirs = findCTDirectories(root_path);
    
    % Process each CT directory
    fprintf('Found %d CT series to process\n', length(ct_dirs));
    for i = 1:length(ct_dirs)
        try
            processSingleCT(ct_dirs{i}, results_dir, params);
        catch ME
            warning('Failed to process %s: %s', ct_dirs{i}, ME.message);
            continue;
        end
    end
    
    % Generate summary
    generateSummaryReport(results_dir);
end

function processSingleCT(ct_path, results_dir, params)
    [~, study_id] = fileparts(ct_path);
    fprintf('\nProcessing study: %s\n', study_id);
    
    % Load DICOM series
    [ctScan, dicomInfo] = loadDICOMSeries(ct_path);
    
    % Calculate GNI
    [gni, noiseMap] = calculateGNI(ctScan, params);
    
    % Save results
    saveResults(gni, noiseMap, dicomInfo, study_id, results_dir);
end

function [gni, noiseMap] = calculateGNI(ctScan, params)
    % Initialize outputs
    numSlices = size(ctScan, 3);
    noiseMap = zeros(size(ctScan));
    sliceGNIs = zeros(numSlices, 1);
    
    % Create body mask
    fprintf('Creating body mask...\n');
    bodyMask = createBodyMask(ctScan);
    
    % Process each slice
    fprintf('Processing %d slices...\n', numSlices);
    for sliceIdx = 1:numSlices
        if mod(sliceIdx, 10) == 0
            fprintf('Processing slice %d of %d\n', sliceIdx, numSlices);
        end
        
        slice = double(ctScan(:,:,sliceIdx));
        mask = bodyMask(:,:,sliceIdx);
        
        % Calculate noise map
        noiseSlice = calculateNoiseMap(slice, mask, params);
        noiseMap(:,:,sliceIdx) = noiseSlice;
        
        % Calculate GNI for slice
        sliceGNIs(sliceIdx) = calculateSliceGNI(noiseSlice, slice, mask, params);
    end
    
    % Smooth GNI values
    gni = smoothGNIValues(sliceGNIs, params.smoothingKernelSize);
end

function [ctScan, dicomInfo] = loadDICOMSeries(dicom_path)
    files = dir(fullfile(dicom_path, '*.dcm'));
    if isempty(files)
        error('No DICOM files found in: %s', dicom_path);
    end
    
    info = dicominfo(fullfile(dicom_path, files(1).name));
    ctScan = zeros(info.Rows, info.Columns, length(files));
    
    positions = zeros(length(files), 1);
    headers = cell(length(files), 1);
    
    for i = 1:length(files)
        info = dicominfo(fullfile(dicom_path, files(i).name));
        headers{i} = info;
        positions(i) = info.ImagePositionPatient(3);
        
        img = double(dicomread(fullfile(dicom_path, files(i).name)));
        ctScan(:,:,i) = img * info.RescaleSlope + info.RescaleIntercept;
    end
    
    [positions, sortIdx] = sort(positions);
    ctScan = ctScan(:,:,sortIdx);
    headers = headers(sortIdx);
    
    dicomInfo = struct();
    dicomInfo.headers = headers;
    dicomInfo.positions = positions;
    dicomInfo.sliceSpacing = mean(diff(positions));
    dicomInfo.pixelSpacing = headers{1}.PixelSpacing;
    try
        dicomInfo.studyDescription = headers{1}.StudyDescription;
        dicomInfo.ctdiVol = headers{1}.CTDIvol;
    catch
        dicomInfo.studyDescription = 'Unknown';
        dicomInfo.ctdiVol = NaN;
    end
end

function bodyMask = createBodyMask(ctScan)
    bodyMask = false(size(ctScan));
    
    for i = 1:size(ctScan,3)
        slice = ctScan(:,:,i);
        normalizedSlice = mat2gray(slice, [-1024, 3000]);
        thresh = graythresh(normalizedSlice);
        mask = imbinarize(normalizedSlice, thresh);
        mask = bwareaopen(mask, 1000);
        mask = imfill(mask, 'holes');
        bodyMask(:,:,i) = mask;
    end
end

function noiseMap = calculateNoiseMap(slice, mask, params)
    kernel = ones(params.kernelSize);
    noiseMap = stdfilt(slice, kernel);
    
    tissueMask = slice >= params.tissueRange(1) & ...
                 slice <= params.tissueRange(2);
    
    validMask = tissueMask & mask;
    noiseMap(~validMask) = NaN;
    
    padding = floor(params.kernelSize/2);
    noiseMap(1:padding,:) = NaN;
    noiseMap(end-padding+1:end,:) = NaN;
    noiseMap(:,1:padding) = NaN;
    noiseMap(:,end-padding+1:end) = NaN;
end

function sliceGNI = calculateSliceGNI(noiseMap, ~, ~, params)
    validNoise = noiseMap(~isnan(noiseMap));
    
    if isempty(validNoise)
        sliceGNI = NaN;
        return;
    end
    
    edges = 0:params.histogramBinSize:max(validNoise(:));
    [counts, ~] = histcounts(validNoise, edges);
    counts = smoothdata(counts, 'gaussian', 5);
    [~, maxIdx] = max(counts);
    sliceGNI = edges(maxIdx);
end

function smoothedGNI = smoothGNIValues(sliceGNIs, kernelSize)
    smoothingKernel = ones(kernelSize,1) / kernelSize;
    validGNIs = sliceGNIs(~isnan(sliceGNIs));
    
    if isempty(validGNIs)
        smoothedGNI = NaN;
        return;
    end
    
    padding = floor(kernelSize/2);
    paddedGNIs = padarray(validGNIs, [padding 0], 'replicate');
    smoothedGNI = conv(paddedGNIs, smoothingKernel, 'valid');
end

function ct_dirs = findCTDirectories(root_path)
    ct_dirs = {};
    
    if containsDICOMs(root_path)
        ct_dirs = {root_path};
        return;
    end
    
    contents = dir(root_path);
    for i = 1:length(contents)
        if contents(i).isdir && ~strcmp(contents(i).name, '.') && ~strcmp(contents(i).name, '..')
            full_path = fullfile(root_path, contents(i).name);
            if containsDICOMs(full_path)
                ct_dirs{end+1} = full_path;
            else
                sub_dirs = findCTDirectories(full_path);
                ct_dirs = [ct_dirs sub_dirs];
            end
        end
    end
end

function has_dicoms = containsDICOMs(dir_path)
    files = dir(fullfile(dir_path, '*.dcm'));
    if isempty(files)
        has_dicoms = false;
        return;
    end
    
    try
        info = dicominfo(fullfile(dir_path, files(1).name));
        has_dicoms = isfield(info, 'Modality') && strcmp(info.Modality, 'CT');
    catch
        has_dicoms = false;
    end
end

function saveResults(gni, noiseMap, dicomInfo, study_id, results_dir)
    study_results_dir = fullfile(results_dir, study_id);
    if ~exist(study_results_dir, 'dir')
        mkdir(study_results_dir);
    end
    
    % Save numerical results
    save(fullfile(study_results_dir, 'gni_results.mat'), ...
        'gni', 'noiseMap', 'dicomInfo');
    
    % Create and save visualizations
    createVisualization(gni, noiseMap, dicomInfo, study_results_dir);
    
    % Save text summary
    saveSummaryText(gni, dicomInfo, study_results_dir);
end

function createVisualization(gni, noiseMap, dicomInfo, save_dir)
    fig = figure('Visible', 'off');
    
    subplot(2,2,1);
    midSlice = round(size(noiseMap,3)/2);
    imagesc(noiseMap(:,:,midSlice));
    colorbar;
    title('Central Slice Noise Map');
    axis equal tight;
    
    subplot(2,2,2);
    plot(dicomInfo.positions, gni);
    xlabel('Position (mm)');
    ylabel('GNI (HU)');
    title('GNI vs. Position');
    
    saveas(fig, fullfile(save_dir, 'noise_analysis.png'));
    close(fig);
end

function saveSummaryText(gni, dicomInfo, save_dir)
    fid = fopen(fullfile(save_dir, 'summary.txt'), 'w');
    fprintf(fid, 'GNI Analysis Summary\n');
    fprintf(fid, 'Date: %s\n', datestr(now));
    fprintf(fid, 'Study: %s\n', dicomInfo.studyDescription);
    fprintf(fid, 'Mean GNI: %.2f HU\n', mean(gni));
    fprintf(fid, 'Std Dev GNI: %.2f HU\n', std(gni));
    fprintf(fid, 'CTDIvol: %.2f mGy\n', dicomInfo.ctdiVol);
    fclose(fid);
end

function generateSummaryReport(results_dir)
    studies = dir(fullfile(results_dir, '*'));
    studies = studies([studies.isdir]);
    studies = studies(~ismember({studies.name}, {'.', '..'}));

    summary = table();
    for i = 1:length(studies)
        try
            load(fullfile(results_dir, studies(i).name, 'gni_results.mat'));
            row = table({studies(i).name}, mean(gni), std(gni), dicomInfo.ctdiVol, ...
                'VariableNames', {'StudyID', 'MeanGNI', 'StdGNI', 'CTDIvol'});
            summary = [summary; row];
        catch
            continue;
        end
    end

    writetable(summary, fullfile(results_dir, 'batch_summary.csv'));
end

function displayResults(results, study_id)
    fprintf('\n=================================\n');
    fprintf('Results for study: %s\n', study_id);
    fprintf('=================================\n');
    fprintf('Mean GNI: %.2f HU\n', mean(results.gni));
    fprintf('Std Dev GNI: %.2f HU\n', std(results.gni));
    fprintf('Min GNI: %.2f HU\n', min(results.gni));
    fprintf('Max GNI: %.2f HU\n', max(results.gni));
    
    if isfield(results.dicomInfo, 'ctdiVol') && ~isnan(results.dicomInfo.ctdiVol)
        fprintf('CTDIvol: %.2f mGy\n', results.dicomInfo.ctdiVol);
    end
    
    fprintf('Number of slices: %d\n', length(results.gni));
    fprintf('Results saved to: %s\n', results.save_path);
    fprintf('=================================\n\n');
end
