function [result] = LmnnFlowSave()
	clear
	close all
	clc;
	rand('seed',1);

	xTr = csvread('Flowfeature.csv')';
	yTr = csvread('Flowclass.csv')';
	cd 'LMNN/mlcircus-lmnn-608259407140/';
	setpaths;
	fprintf('Use LMNN for dimensionality reduction\n');
	L0=pca(xTr)';
	[L,~] = lmnn2(xTr, yTr,3,L0,'maxiter',1000,'quiet',1,'outdim',3,'mu',0.5,'validation',0.2,'earlystopping',25,'subsample',0.3);

	save 'flowQuerry.mat';
	result = 'Lmnn flow calculated and matrix saved';
	cd '../..';
	return;