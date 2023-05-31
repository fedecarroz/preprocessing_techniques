classdef FullBatchGD < handle
    properties
        n_features;
        epochs = 2000;
        learning_rate = 1e-3;
        lambda = 0;
        penalty = 'no_penalty'
        theta;
        cost_history;
        theta_history;
    end
    
    methods
        function obj = FullBatchGD(n_features, epochs, learning_rate, lambda, penalty)
            obj.n_features = n_features;
            obj.epochs = epochs;
            obj.learning_rate = learning_rate;
            obj.lambda = lambda;
            obj.penalty = penalty;
        end
        
        function fit(obj, X, y)
            m = size(X, 1); % Training samples

            if all(X(:,1) ~= 1)
                X = [ones(m, 1), X]; % Adding the bias
            end
            
            obj.theta = randn(obj.n_features + 1, 1); % Weights initialization
            obj.cost_history = zeros(obj.epochs, 1); % cost history initialization
            obj.theta_history = zeros(obj.epochs, obj.n_features+1); % Theta history initialization
            
            for step = 1 : obj.epochs
                y_pred = X * obj.theta; % Calculation of predictions
                error = y_pred - y; % Calculation of the error
                
                % TODO: controllare il calcolo dei regularization term
                if obj.penalty == "L1"
                    regularization = (obj.lambda/m) * sign(obj.theta); % L1 regularization term
                elseif obj.penalty == "L2"
                    regularization = (obj.lambda/m) * obj.theta; % L2 regularization term
                else
                    regularization = zeros(size(obj.theta)); % No regularization applied
                end
                
                gradient = ((X' * error) + regularization) / m; % Calculation of the gradient
                obj.theta = obj.theta - obj.learning_rate * gradient; % Updating weights
                
                obj.cost_history(step) = mean((y_pred - y).^2); % Adding the cost value to the cost history
                obj.theta_history(step, :) = obj.theta'; % Adding the theta value to the theta history
            end
        end

        function cost = compute_cost(obj, m, y_pred, y) % TODO: commentare il codice
            if obj.penalty == "L1"
                regularization_term = obj.lambda * sum(abs(obj.theta));
            elseif obj.penalty == "L2"
                regularization_term = obj.lambda * sum(obj.theta.^2);
            else
                error("Invalid penalty. Must be 'L1' or 'L2'.");
            end
            
            cost = ((1/(2*m)) * sum((y_pred - y).^2)) + regularization_term;
        end
        
        function y_pred = predict(obj, X)
            if all(X(:,1) ~= 1)
                m = size(X, 1); % Number of samples
                X = [ones(m, 1), X]; % Adding the bias
            end
            
            y_pred = X * obj.theta; % Prediction
        end
    end
end
