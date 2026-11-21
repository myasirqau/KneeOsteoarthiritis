%% FILE: ResNet_FOA_LASSO_NN.m
%% This is first version of code and it has been changed fro ablation studies and to find the impact of different hyperparamters, use this code with proper adjustments
clc; clear; close all;



%% ---------------------- 7. Save Results ------------------------
save Hybrid_FOA_LASSO_NN_Results.mat ...
     accuracy foaFitness foaMask lassoMask

fprintf("\n✔ Results saved successfully.\n");
