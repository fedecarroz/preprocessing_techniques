%% Preprocessing techniques project
%% Preliminary operations

clear
clc
%% Import data

dataset = readtable('train.csv');

% Working variable
data = dataset;
%% Normalization setup step

categorical_variables_names = string([]);
normalization_type = 'zscore';
%% Preprocessing phase

% Dopo un attento studio del dataset sono state eliminate delle
% colonne. Di seguito vengono spiegate le motivazioni

disp(dataset.Fence)

% La colonnsa Fence contiene dati non informativi, ovvero non c'è una 
% variazione significativa di questi ultimi 
% quindi non contribuiscono alla comprensione del fenomeno

disp(dataset.Utilities)

% La colonna Utilities viene eliminata in quanto tutte le sue occorrenze
% sono uguali

disp(dataset.WoodDeckSF)

% La colonna WoodDeckSF contiene un'alta percentuale di valori
% nulli o zero, di conseguenza non è utile alla nostra analisi. 
% In più un elevato numero di dati nulli implica una ridotta varianza
% e una ridotta informazione.

disp(dataset.FireplaceQu)

% La colonna  FireplaceQu presenta diversi NaN values. Quindi da un'attenta analisi è stata eliminata in seguito in quanto
% una mancata gestione comporta errori o comportamenti imprevisti, 
% per esempio la produzione di risultati non validi

disp(dataset.TotRmsAbvGrd)

% La colonna TotRmsAbvGrd è stata eliminata in quanto presenta dati
% incosistenti, ovvero dati contradditori o nn confrmi che influiscono
% negativamente sul processo di addestramento e sui risultati

% Removal of 'pool-related' non-informative records
data(data.PoolArea > 0, :) = [];

% Addition of a 'garage-related' feature
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
column_names = [ ...
    'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', ...
    'HeatingQC', 'KitchenQual', 'GarageCond', ...
]
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
%% Correlazione

corr_mat = corrcoef(table2array(data))
features_names = (data(:, 1:end-1).Properties.VariableNames)'
target_corr = abs(corr_mat(1:end-1,end))

[max_target_corr, I_max] = sort(target_corr, "descend");
max_features_names = features_names(I_max);
max_corr = table(max_features_names, max_target_corr);
disp("MAXIMUM CORRELATION VALUES");
disp(max_corr(1:10, :));

[min_target_corr, I_min] = sort(target_corr, "ascend");
min_features_names = features_names(I_min);
min_corr = table(min_features_names, min_target_corr);
disp("MINIMUM CORRELATION VALUES");
disp(min_corr(1:10, :));

for i = 1 : length(min_target_corr)
    features_names = (data(:, 1:end-1).Properties.VariableNames)';
    var_name = min_features_names{i};
    if min_target_corr(i) < 0.2
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

% Dall'analisi della correlazione di ogni features con la correlazione
% della feature target, è stata eliminata la features relativa al SaleType
% Quest'ultima presenta una bassa correlazione con la variabile target e non
% presenta informazioni essenziali per la nostra analisi. Di conseguenza è
% rimossa anche la variabile MiscVal in quanto di riferimento ad una 
% caratteristica che è stata eliminata.

clear min_target_corr I_min min_corr min_features_names
clear max_target_corr I_max max_corr max_features_names
clear i corr_mat features_names target_corr var_name index data_index
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

clear i column Q1 Q2 Q3 IQR
clear outlier_step outlier_list_col outlier_indices
clear outlier_indices_count multiple_outliers
%% Data splitting

cv = cvpartition(height(data),'HoldOut',0.2);
training_data = data(training(cv),:);
test_data = data(test(cv),:);

X_train = table2array(removevars(training_data, {'SalePrice'}));
y_train = table2array(training_data(:, {'SalePrice'}));

X_test = table2array(removevars(test_data, {'SalePrice'}));
y_test = table2array(test_data(:, {'SalePrice'}));

clear cv training_data test_data
%% PCA and normalization

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

% Normalization of numerical (continuous) data only
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