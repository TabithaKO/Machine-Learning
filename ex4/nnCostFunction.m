function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%=========Forward Propagation================

% Adding a column of ones to the input matrix X
X = [ones(m, 1) X];
% ------------------------------------------------
% Setting the input layer 
a1 = X;
% Computing the parameters
z2 = a1 * Theta1';
% Computing the activation on the second layer
a2 = sigmoid(z2);
% -------------------------------------------------
% Adding a bias input for the second layer
a2 = [ones(size(z2,1), 1) a2];
% Determining the activation on the output
z3 = a2 * Theta2';
% Computing the prediction from the output layer
a3 = (sigmoid(z3));

%The y values are in a matrix full of labels and so we're gonna unroll this
%matrix
%To begin with we'll create an identity matrix so that we have a 1s and 0s
%column for each label
eye_matrix = eye(num_labels);
%Then we'll unroll this identity matrix and form the vector of labels
% This is sort of like a classification
y_matrix = eye_matrix(y,:);

%=======================================
%Computing the cost function using a for loop I'll figure this out later

% for i = 1:m 
%    J = J + (log(a3)*labels(:,y(i)) + log(1-a3)*(1-labels(:,y(i))));
% end
%  J = J/(-m);

% =========================================
% Computing the cost using the vectorization method
J = (sum(sum(y_matrix .* log(a3) + (1-y_matrix) .* log(1-a3))))/(-m);

% -------------------------------------------------------------
% Computing the Regularized cost function
% Setting approprtiate theta matrices that exclude the bias units
newTheta1 = Theta1(:,2:end);
newTheta2 = Theta2(:,2:end);

% ============================================
% Computing the regularization terms separately
reg1 = (sum(sum(newTheta1 .^2)));
reg2 = (sum(sum(newTheta2 .^2)));
regfull = (lambda/(2*m)) * (reg1 + reg2);

% -------------Regularized Cost----------------
J = J + regfull;

% ===================================================
% -------------Back Propagation----------------------
% =============Computing the error===================
% Beginning from the back d3
d3 = a3 - y_matrix;
% We need to exclude the first bias column in Theta2 
d2 = (d3 *(newTheta2)) .* sigmoidGradient(z2);

% =============Computing the gradients===================
Delta1 = d2' * a1;
Delta2 = d3' * a2;

Theta1_grad = Delta1/m;
Theta2_grad = Delta2/m;

% =============Gradient Regularization====================

% Setting the bias units to 0
% The theta1 references the next hidden layer
regTheta1 = [zeros(hidden_layer_size , 1) Theta1(:,2:end)];
% The theta1 references the next hidden layer
regTheta2 = [zeros(num_labels, 1) Theta1(:,2:end)];

Theta1_grad = Theta1_grad +((regTheta1) *(lambda/m)) ;
Theta2_grad = Theta2_grad + ((regTheta2) * (lambda/m));
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
