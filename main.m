clear
clc
%%
%% 
% Import data
dataset = readtable('train.csv');
% Working variable
data = dataset;
%%
% Removal of 'pool-related' non-informative rows
data(data.PoolArea > 0, :) = [];
%%
% Normalization setup step
categorical_variables_indices = [];
normalization_type = 'minmax';
%%
% Garage ...
has_garage = zeros(height(data), 1);
has_garage(data.GarageArea > 0) = 1;

data = addvars( ...
    data, ...
    has_garage, ...
    'Before', ...
    'GarageArea', ...
    'NewVariableNames', ...
    'HasGarage' ...
);

clear has_garage
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

has_garage_index = find(strcmp(data.Properties.VariableNames, 'HasGarage'));
categorical_variables_indices = [categorical_variables_indices, has_garage_index];

clear has_garage_index
%%
% Missing values analysis
nan_count = sum(ismissing(data))
nan_indices = find(nan_count > 0)

% Missing values management
for i = nan_indices
    data.(i) = fillmissing( ...
        data.(i), ...
        'constant', ...
        mean(i(~isnan(i))) ...
    );
end

clear nan_count nan_indices i
%%
num_features = (size(data, 2)) -1
%%
%%
columns_indices = [19, 20, 22, 23, 27, 38, 43];
categorical_variables_indices = [categorical_variables_indices, columns_indices];

keys = {'NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'};
values = [0, 1, 2, 3, 4, 5];

for i = columns_indices
    old_column = data.(i);
    new_column = zeros(length(data.(i)), 1);

    for j = 1 : length(keys)
        new_column(strcmp(old_column, keys(j))) = values(j);
    end
    data.(i) = new_column;
end

clear columns_indices keys values old_column new_column i j
%%
for i = 1 : num_features
    if ~isnumeric(data.(i))
        if ~ismember(i, categorical_variables_indices)
            categorical_variables_indices = [categorical_variables_indices, i];
        end
        data.(i) = grp2idx(data.(i));
    end
end

clear i
%% Outliers removal

outlier_indices = [];

% Finds rows with outliers
for i = 1:(size(data, 2) - 1)
    column = data.(i);

    % Check if column has numeric values
    if isnumeric(column)
        Q1 = prctile(column, 25);
        Q3 = prctile(column, 75);

        IQR = Q3 - Q1;
        outlier_step = 1.5*IQR;

        outlier_list_col = find( ...
            column > Q3 + outlier_step | column < Q1 - outlier_step ...
        );
        outlier_indices = cat(1, outlier_indices, outlier_list_col);
    end
end

% Remove outliers with at least 2 outliers
[outlier_indices_count, outlier_indices] = groupcounts(outlier_indices);

multiple_outliers = outlier_indices(outlier_indices_count > 2);

data(multiple_outliers,: ) = [];

% Cleaning
clear i column Q1 Q2 Q3 IQR
clear outlier_step outlier_list_col outlier_indices outlier_indices_count
clear multiple_outliers
%% Data splitting

cv = cvpartition(height(data),'HoldOut',0.2);
training_data = data(training(cv),:);
test_data = data(test(cv),:);

X_train = table2array(removevars(training_data, {'SalePrice'}));
y_train = table2array(training_data(:, {'SalePrice'}));

X_test = table2array(removevars(test_data, {'SalePrice'}));
y_test = table2array(test_data(:, {'SalePrice'}));

clear cv training_data test_data
%% PCA

[~, ~, ~, ~, explained] = pca(X_train);
cum_sum = cumsum(explained)

threshold = 99.9

% Hyperparameter tuning (optimal number of components)
for i = 1: length(cum_sum)
    if cum_sum(i) >= threshold
        optimal_num_components = i
        exp_var = cum_sum(i)
        break
    end
end

% Data normalization
X_train_numerical = X_train;
X_train_numerical(:, categorical_variables_indices) = [];
X_train_categorical = X_train(:, categorical_variables_indices);

X_test_numerical = X_test;
X_test_numerical(:, categorical_variables_indices) = [];
X_test_categorical = X_test(:, categorical_variables_indices);

if strcmp(normalization_type, 'zscore')
    [X_train_numerical, mu, sigma] = zscore(X_train_numerical);
    X_test_numerical = standardizeCols(X_test_numerical, mu, sigma);
elseif strcmp(normalization_type, 'minmax')
    [X_train_numerical, train_settings] = normalize(X_train_numerical');
    X_test_numerical = mapminmax.apply(X_test_numerical', train_settings);
end

X_train = [X_train_numerical, X_train_categorical];
X_test = [X_test_numerical, X_test_categorical];

% PCA using optimal number of components
coeff = pca( ...
    X_train, ...
    'NumComponents', ...
    optimal_num_components ...
);

% ...
X_train = X_train * coeff;
X_test = X_test * coeff;

clear explained cum_sum threshold optimal_num_components exp_var
clear X_train_numerical X_train_categorical
clear X_test_numerical X_test_categorical
clear categorical_variables_indices
clear coeff i mu sigma