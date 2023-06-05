%% Preprocessing techniques project (no preprocessing)
%% Preliminary operations

clear
clc

rng(42) % For reproducibility
%% Import data

dataset = readtable('houses.csv');

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
%% Multivariate regression

model = fitrlinear( ...
    X_train, ...
    y_train, ...
    'Solver', 'sgd', ...
    'Learner', 'leastsquares', ...
    'Verbose', 2 ...
);

y_pred = model.predict(X_test);

results = table( ...
    int32(y_test), int32(y_pred), ...
    'VariableNames',["y_test","y_pred"] ...
);
disp(results);

loss_type = model.FittedLoss
loss_value = model.loss(X_test, y_test)

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