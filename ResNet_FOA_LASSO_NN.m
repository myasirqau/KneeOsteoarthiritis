%% FILE: ResNet_FOA_LASSO_NN.m
clc; clear; close all;

fprintf("=== Hybrid FOA + LASSO + Neural Network Classifier ===\n");

%% ---------------------- 1. Load Features ----------------------
load KneeFeatures.mat;   % must contain: KneeFeatures, labels
fprintf("Loaded features: %d samples × %d features\n", ...
        size(KneeFeatures,1), size(KneeFeatures,2));

%% ---------------------- 2. Train/Test Split -------------------
rng(1); % reproducibility
cv = cvpartition(labels, 'HoldOut', 0.3);

XTrain = KneeFeatures(training(cv), :);
YTrain = labels(training(cv));

XTest  = KneeFeatures(test(cv), :);
YTest  = labels(test(cv));

fprintf("Training samples: %d | Testing samples: %d\n", ...
        size(XTrain,1), size(XTest,1));

%% ---------------------- 3. FOA Feature Selection ---------------
fprintf("\nRunning Flea Optimization Algorithm (FOA)...\n");

populationSize = 20;
maxIter = 30;

[foaMask, foaFitness] = FOA_FeatureSelect(XTrain, YTrain, ...
                                         populationSize, maxIter);

XTrain_FOA = XTrain(:, foaMask);
XTest_FOA  = XTest(:, foaMask);

fprintf("FOA selected %d features.\n", sum(foaMask));

%% ---------------------- 4. LASSO Optimization ------------------
fprintf("\nRunning LASSO on FOA-selected features...\n");

lambdaVals = logspace(-4, -2, 50);   % 0.0001 → 0.01
YTrain_num = grp2idx(YTrain);

[B, FitInfo] = lasso(XTrain_FOA, YTrain_num, ...
                     'Lambda', lambdaVals, ...
                     'CV', 5);

lassoCoef = B(:, FitInfo.IndexMinMSE);
lassoMask = lassoCoef ~= 0;

XTrain_Final = XTrain_FOA(:, lassoMask);
XTest_Final  = XTest_FOA(:, lassoMask);

fprintf("LASSO reduced features to %d.\n", sum(lassoMask));

%% ---------------------- 5. Neural Network ----------------------
hiddenLayers = 50;   % Medium NN (fixed for reproducibility)

net = patternnet(hiddenLayers);

% ---- FORCE LONG TRAINING ----
net.trainParam.epochs   = 1000;
net.trainParam.max_fail = 100;    % disable early stopping
net.trainParam.min_grad = 1e-12;   % avoid gradient stop
net.trainParam.goal     = 0;       % disable performance goal

% ---- DISABLE VALIDATION ----
net.divideParam.trainRatio = 1.0;
net.divideParam.valRatio   = 0.0;
net.divideParam.testRatio  = 0.0;

net.trainParam.showWindow = true;


YTrain_dummy = dummyvar(YTrain_num)';

fprintf("\nTraining Neural Network...\n");
net = train(net, XTrain_Final', YTrain_dummy);

%% ---------------------- 6. Testing -----------------------------
YPred = net(XTest_Final');
[~, pred] = max(YPred, [], 1);
pred = pred';

YTest_num = grp2idx(YTest);
accuracy = mean(pred == YTest_num) * 100;

fprintf("\n================ RESULTS ================\n");
fprintf("FOA Fitness Value        : %.4f\n", foaFitness);
fprintf("Classification Accuracy  : %.2f %%\n", accuracy);
fprintf("========================================\n");

%% ---------------------- 7. Save Results ------------------------
save Hybrid_FOA_LASSO_NN_Results.mat ...
     accuracy foaFitness foaMask lassoMask

fprintf("\n✔ Results saved successfully.\n");
