% GlobalNoiseIndex.m
%
% PURPOSE:
% Automated calculation of Global Noise Index (GNI) for CT image series,
% with optional organ-specific analysis.
%
% USAGE:
% [gni, noiseMap] = GlobalNoiseIndex(ctScan, params, organMask)
%
% INPUTS:
% ctScan - 3D matrix of CT values in Hounsfield Units (HU)
% params - Optional structure with fields:
%   .kernelSize - Size of ROI window (default: 9)
%   .tissueRange - HU range for tissue segmentation (default: [-300, 300] for soft tissue)
%   .histogramBinSize - Bin size for noise histogram (default: 0.1 HU)
%   .smoothingKernelSize - Size of kernel for slice smoothing (default: 5)
%   .contrastEnhanced - Boolean indicating contrast-enhanced scan (default: false)
% organMask - Optional 3D binary mask for organ-specific analysis
%
% OUTPUTS:
% gni - Global Noise Index value(s)
% noiseMap - 3D matrix of local noise values

function [gni, noiseMap] = GlobalNoiseIndex(ctScan, params, organMask)
    % Input validation and parameter setup
    validateInput(ctScan);
    params = setDefaultParams(params);
    
    % Check for organ mask
    hasOrganMask = exist('organMask', 'var') && ~isempty(organMask);
    if hasOrganMask
        validateOrganMask(organMask, size(ctScan));
    end
    
    % Initialize outputs
    numSlices = size(ctScan, 3);
    noiseMap = zeros(size(ctScan));
    sliceGNIs = zeros(numSlices, 1);
    
    % Create body mask
    fprintf('Creating body mask...\n');
    bodyMask = createBodyMask(ctScan);
    
    % Adjust tissue range if contrast-enhanced
    if params.contrastEnhanced
        params.tissueRange = adjustRangeForContrast(params.tissueRange);
    end
    
    % Process each slice
    fprintf('Processing %d slices...\n', numSlices);
    for sliceIdx = 1:numSlices
        if mod(sliceIdx, 10) == 0
            fprintf('Processing slice %d of %d\n', sliceIdx, numSlices);
        end
        
        slice = double(ctScan(:,:,sliceIdx));
        mask = bodyMask(:,:,sliceIdx);
        
        % Apply organ mask if provided
        if hasOrganMask
            mask = mask & organMask(:,:,sliceIdx);
            
            % Skip slice if insufficient organ voxels
            if sum(mask(:)) < params.kernelSize^2
                sliceGNIs(sliceIdx) = NaN;
                continue;
            end
        end
        
        % Calculate noise map
        noiseSlice = calculateNoiseMap(slice, mask, params);
        noiseMap(:,:,sliceIdx) = noiseSlice;
        
        % Calculate GNI for slice
        sliceGNIs(sliceIdx) = calculateSliceGNI(noiseSlice, slice, mask, params);
    end
    
    % Smooth GNI values across slices
    gni = smoothGNIValues(sliceGNIs, params.smoothingKernelSize);
end

function params = setDefaultParams(params)
    if nargin < 1 || isempty(params)
        params = struct();
    end
    
    defaultParams = struct(...
        'kernelSize', 9, ...            % 9x9 kernel
        'tissueRange', [-300, 300], ... % HU range for soft tissue
        'histogramBinSize', 0.1, ...    % HU bin size
        'smoothingKernelSize', 5, ...   % Smoothing kernel size
        'contrastEnhanced', false ...   % Contrast enhancement flag
    );
    
    fields = fieldnames(defaultParams);
    for i = 1:length(fields)
        if ~isfield(params, fields{i})
            params.(fields{i}) = defaultParams.(fields{i});
        end
    end
end

function validateOrganMask(organMask, expectedSize)
    if ~isequal(size(organMask), expectedSize)
        error('Organ mask size must match CT scan size');
    end
    if ~islogical(organMask)
        error('Organ mask must be binary');
    end
end

function range = adjustRangeForContrast(range)
    % Adjust tissue range for contrast-enhanced scans
    % Typically need to increase upper threshold
    range(2) = range(2) + 100;  % Adjust based on your specific needs
end

% [Previous helper functions remain the same]