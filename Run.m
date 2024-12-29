clear;
addpath('measure/');
addpath('my_tools/');
load('new_UCI.mat');
dataset_name = 'new_UCI.mat';

mu =2.5;
alpha=3.5;
anchor_rate= 0.017;               

G_res = EBTLR_AG(X, gt, alpha, mu,anchor_rate);