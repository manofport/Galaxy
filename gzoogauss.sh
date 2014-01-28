#! /usr/local/bin/octave -qf


addpath("/Users/eportman/Development/answerinformatics/gzoo");
addpath("/Users/eportman/Development/answerinformatics/functions/general");
addpath("/Users/eportman/Development/answerinformatics/functions/NN");
addpath("/Users/eportman/Development/answerinformatics/functions/PCA");
addpath("/Users/eportman/Development/answerinformatics/functions/IMG");





printf ("%s", program_name ());
arg_list = argv ();
for i = 1:nargin
    printf (" %s", arg_list{i});
endfor

printf ("\n");

save_p_restarttime = "";
save_p_step = "";
save_p_loadfile = "";
completion_file = strcat("/Users/eportman/Development/answerinformatics/gzoo/completenotice/",sprintf("%d",clock));
ytrain_file ="/Users/eportman/Development/answerinformatics/gzoo/newdata/training_solutions_rev1.csv";
ytest_file ="/Users/eportman/Development/answerinformatics/gzoo/y_test.csv";

    img_dir = "/Users/eportman/Development/answerinformatics/gzoo/newdata/images_training_rev1/*.jpg";
    testimg_dir = "/Users/eportman/Development/answerinformatics/gzoo/badimages_test/*.jpg";
    save_p_imagelist = glob(img_dir);
    save_p_testimagelist = glob(testimg_dir);
    imagewidth = 424;
    save_p_imgtrim = 125;
    imgsize = (imagewidth-2*save_p_imgtrim)^2;
    save_p_targetpctvariance = .98;
    save_p_hidden_layer = 100;
    save_p_iterations = 500;
    save_p_lambda = 1;
    save_p_shrink_factor = 2; %this must be greater than one.  The larger the nbr the less the shrinkage.
    m = size(save_p_imagelist,1);

    save_p_imin = 100000;
    save_p_images_to_process = min(save_p_imin,m);
    save_p_testimages_to_process = size(save_p_testimagelist,1);
    save_p_step = "processimage";


if (nargin == 0)
    timestring = sprintf("%d",clock);
    results_dir = "/Users/eportman/Development/answerinformatics/gzoo/piperesults/";
    results_dir = strcat(results_dir,timestring,"/");
    mkdir(results_dir);






    parmdatafile = strcat(results_dir,"parmdata",timestring,".mat");
    save(parmdatafile,"-7","save_p_*");
    fprintf("Saved parameters in %s\n",parmdatafile);
    fflush(stdout);

else


    if arg_list{1} == "-step"
        save_p_step = arg_list{2};
        save_p_loadfile = arg_list{3};
        save_p_restarttime = sprintf("%d",clock);
        results_dir = strcat("./",save_p_restarttime);
        mkdir(results_dir);
        load(save_p_loadfile);
        save_p_step = arg_list{2};
        whos
            save_p_iterations = 500;
        save_p_lambda = 1;

        save_p_testimages_to_process = size(save_p_testimagelist,1);
    else
        return;
    endif
endif




%Steps
%
% initial 3 output nn
% then go from there...
%
%
%
%

not_done = true;
done = "";
next_step = save_p_step;

while not_done
    switch next_step
        case "processimage"
            fprintf("Step1: Process Images\n\n");
            fflush(stdout);

            
            step1_save_X = processGaussianImages(img_dir,save_p_imagelist,imagewidth,save_p_shrink_factor,save_p_imgtrim,save_p_images_to_process,results_dir);
            result_file = strcat(results_dir,"process_image_result.mat");
            mkdir(results_dir);
            save(result_file,"-7","step1_save_*","save_*");
            done = strcat(done,"\n",next_step);
            save(strcat(completion_file,".image.txt"),"-text","done");


            next_step = "pca";
        case "pca"
            fprintf("Step2: Run PCA\n\n");
            fflush(stdout);
            [step2_save_Z, step2_save_U, step2_save_k,step2_save_mu,step2_save_sigma] = runPCA(step1_save_X,save_p_targetpctvariance,results_dir);


            result_file = strcat(results_dir,"pca_result.mat");
            mkdir(results_dir);
            save(result_file,"-7","step2_save_*","save_*");
            done = strcat(done,"\n",next_step);
            save(strcat(completion_file,".pca.txt"),"-text","done");
            next_step = "nnp";
        case "nn"
            fprintf("Step3: Run NN\n\n");
            save_p_iterations = 500;
            save_p_lambda = 1;

            step3_save_train_y = load(ytrain_file); %%%% replace this
            [step3_save_train_pred,step3_save_train_prob,step3_save_theta1,step3_save_theta2] = runNN(step2_save_Z,step3_save_train_y,save_p_hidden_layer,save_p_iterations,save_p_lambda,results_dir);
            result_file = strcat(results_dir,"nn_result.mat");
            mkdir(results_dir);
            save(result_file,"-7","step3_save_*","save_*","step2_save_*");
            done = strcat(done,"\n",next_step);


            temptrainy = step3_save_train_y(:,2:4)';
            human_pred = max(temptrainy)';
            done = strcat(done,sprintf('\nTraining Set Accuracy: %f\n', mean(double(step3_save_train_pred == human_pred)) * 100)); %not working


            save(strcat(completion_file,".nn.txt"),"-text","done");

            next_step = "nnp";
        case "nnp"
            save_p_iterations = 1500;
            save_p_lambda = 1;
            save_p_trainsubset = 50000;
            fprintf("Step3: Run NN Prob\nLambda is %d\n",save_p_lambda);


            step3p_save_train_y = load(ytrain_file); %%%% replace this
            step3p_save_train_ysub = step3_save_train_y(1:save_p_trainsubset,:);
            step2_save_Zsub = step2_save_z(1:50000,:);



            [step3p_save_train_pred,step3p_save_train_prob,step3p_save_theta1,step3p_save_theta2] = runNNProbabilities(step2_save_Zsub,step3p_save_train_ysub,save_p_hidden_layer,save_p_iterations,save_p_lambda,results_dir);

            [temptrainymax, temptrainymaxi] = max(step3p_save_train_ysub(:,2:4)');
            step3p_save_actualy = temptrainymaxi';

            whos
            accuracy = sum(step3p_save_train_pred == step3p_save_actualy)/size(step3p_save_actualy,1);

            fprintf("Accuracy against training set is: %d\n",accuracy*100);


            result_file = strcat(results_dir,"nnp_result.mat");
            mkdir(results_dir);
            save(result_file,"-7","step3*","save_*","step2_save_*");
            done = strcat(done,"\n",next_step);

            done = strcat(done,sprintf('\nTraining Set Accuracy: %f\n', accuracy);


            save(strcat(completion_file,".nnp.txt"),"-text","done");

            next_step = "crossvalidation";



        case "crossvalidation"
            fprintf("Step4: Cross Validation\n\n");
            

            step4_save_Z = projectData(testX_norm,step2_save_U,step2_save_k);

            [step4_save_p,step4_save_solutionclass,step4_save_h2,step4_save_solutionprob] = runPredictTest(step4_save_Z,step4_save_testy,step3_save_theta1,step3_save_theta2);
            result_file = strcat(results_dir,"predict_result.mat");
            mkdir(results_dir);
            save(result_file,"-7","step4_save_*","save_*");
            done = strcat(done,"\n",next_step);

            accuracy = 100*sum(step4_save_p == step4_save_solutionclass)/size(step4_save_p,1);


            done = strcat(done,sprintf("\nTest Class Accuracy is: %d",accuracy),"\n");
            fprintf("Test Class Accuracy is: %d\n",accuracy);
            save(strcat(completion_file,".predict.txt"),"-text","done");
        
            next_step = "done";


        case "predicttest"
            fprintf("Step4: Predict Test\n\n");
            

            step4_save_testy = load(ytest_file); %%%% replace this

            testX = processGaussianImages(testimg_dir,save_p_testimagelist,imagewidth,save_p_shrink_factor,save_p_imgtrim,save_p_testimages_to_process,results_dir);

            [testX_norm] = testNormalize(testX,step2_save_mu,step2_save_sigma);

            step4_save_Z = projectData(testX_norm,step2_save_U,step2_save_k);

            [step4_save_p,step4_save_solutionclass,step4_save_h2,step4_save_solutionprob] = runPredictTest(step4_save_Z,step4_save_testy,step3_save_theta1,step3_save_theta2);
            result_file = strcat(results_dir,"predict_result.mat");
            mkdir(results_dir);
            save(result_file,"-7","step4_save_*","save_*");
            done = strcat(done,"\n",next_step);

            accuracy = 100*sum(step4_save_p == step4_save_solutionclass)/size(step4_save_p,1);


            done = strcat(done,sprintf("\nTest Class Accuracy is: %d",accuracy),"\n");
            fprintf("Test Class Accuracy is: %d\n",accuracy);
            save(strcat(completion_file,".predict.txt"),"-text","done");
        
            next_step = "done";


        case "lr"
            fprintf("Step3: Run Linear Regression\n\n");
            

            step3_save_train_y = load(ytrain_file); %%%% replace this

            step3_save_class11 = step3_save_train_y(:,2);
            [m,n] = size(step2_save_Z)



            size(step3_save_class11)

            X = [ones(m,1) step2_save_Z];
            y = step3_save_class11;

            step3_save_theta = trainLinearReg(X,y, save_p_lambda);
            step3_save_prediction = X*step3_save_theta;
            size(step3_save_prediction)




            result_file = strcat(results_dir,"lr_result.mat");
            mkdir(results_dir);
            save(result_file,"-7","step3_save_*","save_*","step2_save_*");










            done = strcat(done,"\n",next_step);


            next_step = "done";




        otherwise
            not_done = false;

    endswitch

endwhile










