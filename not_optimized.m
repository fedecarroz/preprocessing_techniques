%% Preprocessing techniques project (no preprocessing)
%% Preliminary operations

clear
clc

rng(42) % For reproducibility
%% Import data

dataset = readtable('train.csv');

% Working variable
data = dataset;
%% Missing values analysis

data = rmmissing(data);
%% Categorical features encoding

num_features = (size(data, 2)) -1

for i = 1 : num_features
    if ~isnumeric(data.(i))
        data.(i) = grp2idx(data.(i));
    end
end

clear i num_features
%% Data splitting

cv = cvpartition(height(data),'HoldOut',0.2);
training_data = data(training(cv),:);
test_data = data(test(cv),:);

X_train = table2array(removevars(training_data, {'SalePrice'}));
y_train = table2array(training_data(:, {'SalePrice'}));

X_test = table2array(removevars(test_data, {'SalePrice'}));
y_test = table2array(test_data(:, {'SalePrice'}));

clear cv training_data test_data
%% Linear regression

% TODO: inserire la regressione lineare
%%
y_test
y_pred = predict(model, X_test)
mse_value = mse(y_test, y_pred)
rmse_value = sqrt(mse_value)