function [result] = LmnnQuerry()
	rand('seed',1);
	
	%load Querry data
	xTe = csvread('querry_tmp.csv')';
	yTe = 0;

	%load saved L matrix from Lmnn Training
	cd 'LMNN/mlcircus-lmnn-608259407140/';
	setpaths;
	load 'videoQuerry.mat';
	%querry
	[errL,detail]=knncl(L,xTr, yTr,xTe,yTe,300);
	%result = detail.lTe2;
	result.vid = detail.iTe;
	result.dist = detail.dist;
	cd '../..';
	%csvwrite('ans.csv',result);
	return

