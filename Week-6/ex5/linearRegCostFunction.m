function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%Without Regularization

hx = X * theta;
error = hx - y;
now = error .^ 2;
now = sum(now);
now = (now / 2) / m;

% Regularization

thetaTmp = theta;
thetaTmp(1) = 0;
thetaTmp = thetaTmp .^ 2;
sumTheta2 = sum(thetaTmp);

sumTheta2 = ((sumTheta2 * lambda) / 2 ) / m;

% Now get the whole cost J

J = now + sumTheta2;


% Gradinent calculation

now = X' * error;
now = now / m;

% Regularized Sum

thetaTmp = theta;
thetaTmp(1) = 0;

thetaTmp = thetaTmp .* lambda;
thetaTmp = thetaTmp ./ m;

grad = now + thetaTmp;


% =========================================================================

grad = grad(:);

end
