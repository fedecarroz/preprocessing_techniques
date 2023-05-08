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

clear nanCount
clear nanIndices
clear i

%%

% Check outliers

outlier_indices = []

% Finds rows with outliers

for i = 1:(size(data, 2) - 1)
    col = data.(i);

    % Check if column has numeric values

    if isnumeric(col)

        Q1 = prctile(col, 25);
        Q3 = prctile(col, 75);

        IQR = Q3 - Q1;
        outlier_step = 1.5*IQR;

        outlier_list_col = find( col > Q3 + outlier_step | col < Q1 - outlier_step );
        outlier_indices = cat(1, outlier_indices, outlier_list_col);

    end
end

% Remove outliers with at least 2 outliers

[outlier_indices_count, outlier_indices] = groupcounts(outlier_indices);

multiple_outliers_index_position = find(outlier_indices_count > 2)
multiple_outliers = outlier_indices(multiple_outliers_index_position)

data(multiple_outliers,: ) = [];

% Cleaning
clear multiple_outliers_index_position
clear multiple_outliers
clear i col
clear Q1 Q2 IQR
clear outlier_step outlier_list_col outlier_indices
