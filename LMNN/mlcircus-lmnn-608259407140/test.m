clear
close all
%install;
%setpaths;

clc;
rand('seed',1);

load lmnnvar.mat

[errRAW,classRaw]=knncl([],xTr, yTr,xTe,yTe,1);

L0=pca(xTr)';
[errPCA,classPCA]=knncl(L0(1:3,:),xTr, yTr,xTe,yTe,1);fprintf('\n');

[L,~] = lmnn2(xTr, yTr,3,L0,'maxiter',1000,'quiet',1,'outdim',3,'mu',0.5,'validation',0.2,'earlystopping',25,'subsample',0.3);
[errL,classLmnn]=knncl(L,xTr, yTr,xTe,yTe,1);fprintf('\n');


embed=gb_lmnn(xTr,yTr,3,L,'ntrees',200,'verbose',false,'XVAL',xVa,'YVAL',yVa);
[errGL,classGbLmnn]=knncl([],embed(xTr), yTr,embed(xTe),yTe,1);fprintf('\n');

fprintf('%d ',classRaw.lTe2);fprintf('\n');
fprintf('%d ',classPCA.lTe2);fprintf('\n');
fprintf('%d ',classLmnn.lTe2);fprintf('\n');
fprintf('%d ',classGbLmnn.lTe2);fprintf('\n');