function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions
    

% Calculating the F1 score using the fn, tp, fp values
%     These are the predictions made by the algorithm
%     Getting a binaty vector of 0s and 1s of the outlier predictions
    predictions = (pval < epsilon);

    
% yVal are the actual values of the data
% Calculating the true pos, false pos and false negs
    
    falsepos = sum((predictions == 1) & (yval == 0));
    truepos = sum((predictions == 1) & (yval == 1));
    falseneg = sum((predictions == 0) & (yval == 1));
    
%     Calculating the precision and recall
    precision = truepos/(truepos + falsepos);
    recall = truepos/(truepos + falseneg);
    
%     Calculating the F1 score
    F1= (2*(precision)*(recall))/(precision + recall);
    
% Comparing the F1 values
    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end









    % =============================================================

    
end

end
