function [p,solutionclass,h2,classoneprobabilities] = runPredictTest(Z,y,theta1,theta2)



    classoneprobabilities = y(1:size(Z,1),2:4);

    [solutionclassprobability, solutionclass] = max(classoneprobabilities');

    solutionclass = solutionclass';


    [p,h2] = predict(theta1,theta2,Z);
    
    
  

end