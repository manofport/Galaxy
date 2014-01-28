function [Z,U,k,mu,sigma] = runPCA(X,p_targetpctvariance,results_dir)

  
    %feature normalize
    [X_norm, mu, sigma] = featureNormalize(X);
    [U, S] = pca(X_norm);

    [pct, k] = determineK(S,p_targetpctvariance);
    Z = projectData(X_norm,U,k);
    
end

