%% Preprocessing techniques project
%% Preliminary operations

clear
clc

rng(42) % For reproducibility
%% Import data
% The following dataset contains information regarding real estate, and the 
% goal is to train a model able to predict the sale price through a regression 
% task.

dataset = readtable('houses.csv');

% Working variable
data = dataset;
%% Normalization setup step
% In this section two variables are initialized. The first one will be useful 
% in distinguishing categorical from numerical variables, while the second one 
% defines the type of the normalization. Acceptable alternatives are 'zscore' 
% and 'minmax'.

categorical_variables_names = string([]);
normalization_type = 'zscore';
%% Data cleaning
% After a careful study of the dataset, several variables were eliminated, the 
% reasons are explained below along with an example provided for each.
% 
% The first variable 'Id' should be removed because it does not contain informative 
% data. In fact as the name suggest it contains only an enumeration of all the 
% rows of the table.

disp(data.Id);
%% 
% Variables with too many nan values should be removed as filling these values 
% could lead to values that are too inaccurate. In this dataset, in some cases, 
% nan values are represented by the string 'NA'.
% 
% For example, the 'Fence' column has been taken into account. In this case 
% the 'NA' value indicates a missing quality classification of the fence around 
% the house. 'NA' values could be informative sometimes, but in this case they 
% only represent meaningless missing values.

[str, ~, i] = unique(data.Fence);
count = histcounts(i, 1:numel(str)+1);

[max_count, max_i] = max(count);
max_str = str{max_i};

fprintf( ...
    "'%s' values are %d over %d", ...
    max_str, max_count, length(data.Fence) ...
)

clear str i count max_count max_i
%% 
% Columns such as 'Utilities' should be removed as they contain the same value 
% in each row.

disp(dataset.Utilities);
%% 
% The column 'TotRmsAbvGrd' (Total Rooms Above Grade) has been deleted as it 
% contains inconsistent data. More precisely, several rows present contradictory 
% data as the sum of the number of individual rooms differs from the total contained 
% in this column.

inconsistent_rooms_number = data( ...
    1, {'FullBath', 'BedroomAbvGr', 'HalfBath', 'KitchenAbvGr','TotRmsAbvGrd'} ...
);
disp(inconsistent_rooms_number);

clear inconsistent_rooms_number
%% 
% Another issue was found analyzing the 'PoolArea' column. In fact only few 
% values are informative. As a result it was decided to remove the rows referring 
% to houses with pools from the dataset, leaving just a column of zeros, which 
% can therefore be deleted.

houses_with_pool = sum(data.PoolArea > 0);
fprintf( ...
    "%d houses have the pool over %d", ...
    houses_with_pool, length(data.PoolArea) ...
)
data(data.PoolArea > 0, :) = [];

clear houses_with_pool
%% 
% Features engineering

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

categorical_variables_names = [categorical_variables_names, 'HasGarage'];

clear has_garage
%% 
% After data exploration low informative-content variables are removed.

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
        'PoolQC', 'Fence', 'MiscFeature', 'MoSold','YrSold', ...
    } ...
);
%% Missing values analysis

nan_count = sum(ismissing(data))
nan_indices = find(nan_count > 0)

% Missing values management
for i = nan_indices
    % Since all missing values are in numeric variables, the chosen way to
    % handle them is fill them with the average value of all the other
    % values in the same column.
    data.(i) = fillmissing( ...
        data.(i), ...
        'constant', ...
        mean(i(~isnan(i))) ...
    );
end

clear nan_count nan_indices i
%% Categorical features encoding

% 'Quality-related' features encoding
column_indices = [19, 20, 22, 23, 27, 38, 43];
column_names = string([ ...
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", ...
    "HeatingQC", "KitchenQual", "GarageCond", ...
])
categorical_variables_names = [categorical_variables_names, column_names];

keys = {'NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'};
values = [0, 1, 2, 3, 4, 5];

for i = column_indices
    old_column = data.(i);
    new_column = zeros(length(data.(i)), 1);

    for j = 1 : length(keys)
        new_column(strcmp(old_column, keys(j))) = values(j);
    end
    data.(i) = new_column;
end

clear column_indices column_names
clear keys values old_column new_column i j

% Other categorical features encoding
num_features = (size(data, 2)) -1

for i = 1 : num_features
    if ~isnumeric(data.(i))
        var_name = data.Properties.VariableNames{i};
        if ~ismember(var_name, categorical_variables_names)
            categorical_variables_names = [categorical_variables_names, var_name];
        end
        data.(i) = grp2idx(data.(i));
    end
end

clear i var_name num_features
%% Correlation
% scrivere qualcosa sulla correlazione

corr_mat = corrcoef(table2array(data))
column_names = (data.Properties.VariableNames)';
features_names = column_names(1:end-1,:);
target_corr = abs(corr_mat(1:end-1,end));
%% 
% Scrivere qualcosa sulle heatmap

h_full = heatmap(column_names, column_names, corr_mat)
h_sale_price = heatmap(('SalePrice'), features_names, target_corr)
%% 
% In the following code, the 10 highest and lowest values of correlation with 
% the target variable are shown.

[max_target_corr, I_max] = sort(target_corr, "descend");
max_features_names = features_names(I_max);
max_corr = table(max_features_names, max_target_corr);
disp("MAXIMUM CORRELATION VALUES WITH TARGET VARIABLE");
disp(max_corr(1:10, :));

[min_target_corr, I_min] = sort(target_corr, "ascend");
min_features_names = features_names(I_min);
min_corr = table(min_features_names, min_target_corr);
disp("MINIMUM CORRELATION VALUES WITH TARGET VARIABLE");
disp(min_corr(1:10, :));


% Removal of variables with low correlation with the target variable

target_corr_threshold = 0.5;

for i = 1 : length(min_target_corr)
    features_names = (data(:, 1:end-1).Properties.VariableNames)';
    var_name = min_features_names{i};
    if min_target_corr(i) < target_corr_threshold
        if ismember(var_name, categorical_variables_names)
            index = strcmp(categorical_variables_names, var_name);
            categorical_variables_names(index) = [];
        end
        data_index = find(strcmp(features_names, var_name));
        data.(data_index) = [];
    else
        break
    end
end

clear column_names features_names target_corr
clear min_target_corr I_min min_corr min_features_names
clear max_target_corr I_max max_corr max_features_names
clear i var_name index data_index

% Removal of variables with high correlation with other features
corr_mat = corrcoef(table2array(data));

features_corr_threshold = 0.7;
high_corr_features_num = 2;

high_corr_count = zeros(1,size(corr_mat, 2) - 1);

for i = 1:(size(corr_mat, 2) - 1)
    feature_corr = corr_mat(i,1:end - 1);

    %for each feature find feature which are high correlated except itself
    %(when correlation is equal to 1) 
    high_corr_found = length( ...
        find(feature_corr > features_corr_threshold & feature_corr < 1) ...
    )
    
    %add results to cumulative 
    high_corr_count(i) = high_corr_found;
end

multiple_high_corr_features = find(high_corr_count > high_corr_features_num)
data(:, multiple_high_corr_features) = [];

clear features_corr_threshold high_corr_features_num
clear feature_corr high_corr_found high_corr_indices
clear high_corr_count high_corr_features multiple_high_corr_features


%% 
% Showing the current number of features.

num_features = (size(data, 2)) -1;
fprintf('Number of remaining features: %s', num_features);
clear num_features
%% Outliers removal
% scrivere qualcosa di teoria sull'outlier removal

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

clear i column Q1 Q2 Q3 IQR
clear outlier_step outlier_list_col outlier_indices
clear outlier_indices_count multiple_outliers
%% Data splitting
% In this section a split is performed to separate training from test data.

cv = cvpartition(height(data),'HoldOut',0.2);
training_data = data(training(cv),:);
test_data = data(test(cv),:);

X_train = table2array(removevars(training_data, {'SalePrice'}));
y_train = table2array(training_data(:, {'SalePrice'}));

X_test = table2array(removevars(test_data, {'SalePrice'}));
y_test = table2array(test_data(:, {'SalePrice'}));

clear cv training_data test_data
%% Normalization
% In this section a z-score or minmax normalization is applied to continuous 
% (numerical) data only.

categorical_variables_indices = find( ...
    ismember(data.Properties.VariableNames, categorical_variables_names) ...
);
X_train_numerical = X_train;
X_train_numerical(:, categorical_variables_indices) = [];
X_train_categorical = X_train(:, categorical_variables_indices);

X_test_numerical = X_test;
X_test_numerical(:, categorical_variables_indices) = [];
X_test_categorical = X_test(:, categorical_variables_indices);

if strcmp(normalization_type, 'zscore')
    [X_train_numerical, mu, sigma] = zscore(X_train_numerical);
    X_test_numerical = (X_test_numerical - mu) ./ sigma;
elseif strcmp(normalization_type, 'minmax')
    [X_train_numerical, C, S] = normalize(X_train_numerical, 'range');
    X_test_numerical = (X_test_numerical - C) ./ S;
end

X_train = [X_train_numerical, X_train_categorical];
X_test = [X_test_numerical, X_test_categorical];
%% PCA
% In this section a PCA (Principal Component Analysis) is performed after the 
% tuning of the number of components.

[~, ~, ~, ~, explained] = pca(X_train);
cum_sum = cumsum(explained)

threshold = 90

% Hyperparameter tuning (optimal number of components)
for i = 1: length(cum_sum)
    if cum_sum(i) >= threshold
        optimal_num_components = i
        exp_var = cum_sum(i);
        fprintf( ...
            "The cumulative explained variance using %d number of components is %f", ...
            optimal_num_components, exp_var ...
        )
        break
    end
end

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
clear categorical_variables_names categorical_variables_indices
clear coeff i mu sigma
%% Multivariate regression
% Scrivere qualcosa di teorico

best_kfold_loss = Inf;
best_hyperparameters = [];

for reg_type = ["lasso", "ridge"]
    for lmd = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        for alpha = [1e-3, 1e-2, 1e-1]
            for batch_size = [1, 100, 200]
                model = fitrlinear( ...
                    X_train, ...
                    y_train, ...
                    'Solver', 'sgd', ...
                    'Learner', 'leastsquares', ...
                    'Regularization', reg_type, ...
                    'Lambda', lmd, ...
                    'LearnRate', alpha, ...
                    'BatchSize', batch_size, ...
                    'Verbose', 2, ...
                    'CrossVal', 'on', ...
                    'KFold', 10 ...
                );
                
                
                hyperparameters = [reg_type, lmd, alpha, batch_size]
                model_loss = kfoldLoss(model)
    
                if model_loss < best_kfold_loss
                    best_kfold_loss = model_loss;
                    best_hyperparameters = hyperparameters;
                end
            end
        end
    end
end

clear reg_type lmd alpha batch_size model model_loss hyperparameters

disp("After the tuning, the best hyperparameters are:");
hyperparameters_names = ["Regularization", "Lambda", "Learning rate", "Batch size"];
hyperparmas_table = table( ...
    hyperparameters_names', best_hyperparameters', ...
    'VariableNames', ["Hyperparameter", "Value"] ...
);
disp(hyperparmas_table);
disp("The resulting kfold loss of the model is:");
disp(best_kfold_loss);

clear hyperparameters_names hyperparmas_table best_kfold_loss

best_model = fitrlinear( ...
    X_train, ...
    y_train, ...
    'Solver', 'sgd', ...
    'Learner', 'leastsquares', ...
    'Regularization', best_hyperparameters(1), ...
    'Lambda', str2double(best_hyperparameters(2)), ...
    'LearnRate', str2double(best_hyperparameters(3)), ...
    'BatchSize', str2double(best_hyperparameters(4)) ...
);

y_pred = best_model.predict(X_test);

results = table( ...
    int32(y_test), int32(y_pred), ...
    'VariableNames',["y_test","y_pred"] ...
);
disp(results);

loss_type = best_model.FittedLoss
loss_value = best_model.loss(X_test, y_test)

mse = mean((y_pred - y_test).^2); % Mean Squared Error
rmse = sqrt(loss_value); % Root Mean Squared Error

SSR = sum((y_pred - y_test).^2); % Sum of Squares Regression 
SST = sum((y_test - mean(y_test)).^2); % Total Sum of Squares 
r_squared = 1 - SSR / SST; % Coefficient of determination

format short
metrics = table( ...
    double(loss_value), double(rmse), double(r_squared), ...
    'VariableNames',["MSE", "RMSE", "R_SQUARED"] ...
);
disp(metrics);

clear results SSR SST