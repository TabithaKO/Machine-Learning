function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 

grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%Calculate the hypothesis
hyp = sigmoid(X * theta);

%Calculate the cost function
cost =(-1/m)* sum((y .* log(hyp)) + (1 -y) .* (log(1-hyp)));

%Calculate the gradient
gradient = (1/m)* (X' * (hyp - y));


%Because we need to exclude the bias term we can set theta(1) to zero and
%proceed to multiply the vectors in the regularixation term without
%explicitly skipping theta(1)
theta(1)= 0;


%regularization term for the cost function
regCost = (lambda/(2*m)* sum(theta.^2)); 

%regularization term for gradient
regGrad = (lambda/m)*(theta);

%Use the regularization term in the cost function or gradient
J = cost + regCost;
grad = gradient + regGrad;


% =============================================================

end
