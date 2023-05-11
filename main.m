clear
clc
%%
% Import data
dataset = readtable('train.csv');
% Working variable
data = dataset;
%%
% Removal of 'pool-related' non-informative rows
data(data.PoolArea > 0, :) = [];
%%
% Garage ...
hasGarage = zeros(height(data),1);
hasGarage(data.GarageArea > 0) = 1;

data = addvars( ...
    data, ...
    hasGarage, ...
    'Before', ...
    'GarageArea', ...
    'NewVariableNames', ...
    'hasGarage' ...
);

clear hasGarage
%%
% After data exploration low informative-content variables are removed
data = removevars( ...
    data, ...
    { ...
        'Id', 'Street', 'Alley', 'Utilities', 'Condition1', ...
        'Condition2', 'HouseStyle', 'Exterior1st', 'Exterior2nd', ...
        'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', ...
        'BsmtUnfSF', 'Electrical', 'LowQualFinSF', 'TotRmsAbvGrd', ...
        'Functional', 'FireplaceQu', 'GarageType', 'GarageYrBlt', ...
        'GarageFinish', 'GarageQual', 'WoodDeckSF', 'OpenPorchSF', ...
        'EnclosedPorch', 'x3SsnPorch', 'ScreenPorch', 'PoolArea', ...
        'PoolQC', 'Fence', 'MiscFeature', 'MiscVal', 'MoSold', ...
        'YrSold', 'SaleType', ...
    } ...
);
%%
% Missing values analysis
nanCount = sum(ismissing(data))
nanIndices = find(nanCount > 0)
%%
% Missing values management
for i = nanIndices
    data.(i) = fillmissing( ...
        data.(i), ...
        'constant', ...
        mean(i(~isnan(i))) ...
    );
end

clear nanCount nanIndices i
%%
numFeatures = (size(data, 2)) -1
%%
%%
columnsIndices = [19, 20, 22, 23, 27, 38, 43];
keys = {'NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'};
values = [0, 1, 2, 3, 4, 5];

for i = columnsIndices
    oldColumn = data.(i);
    newColumn = zeros(length(data.(i)), 1);

    for j = 1 : length(keys)
        newColumn(strcmp(oldColumn, keys(j))) = values(j);
    end
    data.(i) = newColumn;
end

clear columnsIndices keys values oldColumn newColumn i j
%%
for i = 1 : numFeatures
    if ~isnumeric(data.(i))
        data.(i) = grp2idx(data.(i));
    end
end
%% Outliers removal

outlierIndices = []

% Finds rows with outliers
for i = 1:(size(data, 2) - 1)
    col = data.(i);

    % Check if column has numeric values
    if isnumeric(col)
        Q1 = prctile(col, 25);
        Q3 = prctile(col, 75);

        IQR = Q3 - Q1;
        outlierStep = 1.5*IQR;

        outlierListCol = find(col > Q3 + outlierStep | col < Q1 - outlierStep );
        outlierIndices = cat(1, outlierIndices, outlierListCol);
    end
end

% Remove outliers with at least 2 outliers
[outlierIndicesCount, outlierIndices] = groupcounts(outlierIndices);

multipleOutliers = outlierIndices(outlierIndicesCount > 2);

data(multipleOutliers,: ) = [];

% Cleaning
clear i col Q1 Q2 Q3 IQR
clear outlierStep outlierListCol outlierIndices outlierIndicesCount
clear multipleOutliers
%%
% ...