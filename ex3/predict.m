function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%



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
% Adding a bias input for this layer
a2 = [ones(size(z2,1), 1) a2];
% Determining the activation on the output
z3 = a2 * Theta2';
% Computing the final prediction
pred = (sigmoid(z3));
%  Getting the max value
[val index] = max(pred, [], 2);
 p = index;







% =========================================================================


end
