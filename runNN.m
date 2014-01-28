function [trainpred,trainprob,theta1,theta2] = runNN(Z,y, p_hidden_layer,p_iterations,p_lambda,results_dir)





    ytrain = y(1:size(Z,1),2:4);
    [partsolutionprobability, partsolutionclass] = max(ytrain');

    solutionclass = partsolutionclass';

    [trainpred, trainprob, theta1,theta2] = mainNN(Z,solutionclass,size(Z,2),p_hidden_layer,3,p_iterations,p_lambda);


end