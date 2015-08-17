function [result] = LmnnFlowQuerry()
	fprintf('Here 1');
	rand('seed',1);
	
	%load Querry data
	xTe = csvread('fquerry_tmp.csv')';
	yTe = 0;

	%load saved L matrix from Lmnn Training
	cd 'LMNN/mlcircus-lmnn-608259407140/';
	setpaths;
	load 'flowQuerry.mat';
	%querry
	[errL,detail]=knncl(L,xTr, yTr,xTe,yTe,100);
	result.vid = detail.iTe;
	result.dist = detail.dist;
	cd '../..';
	%csvwrite('ans.csv',result);
	fprintf('Here 2');
	return