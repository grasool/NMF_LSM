function M = NMF_LSM_without_preproc_BRATS_data(Img, No_region, auto, block_size)

%% This function implements the Non-negative Matrix Factorization-based Level 
% Set Method (NMF-LSM) segmentation approach.
% This code is intended for segmenting MRI Images in which we are interested in extracting different anatomical structures
% i.e. gray matter, white matter, cerebrospinal fluid (CSF), tumor, edema
% (swelling brain), necrosis (dead cells).
% We applied this code to measure the volumes in MRIs of multiple sclerosis (MS), Glioblastoma, and Alzheimer.

% We first applies the function
% [W, H] = Block_Division_and_NMF_Implementation(Img,no_region);
% that will devide the image into blocks, compute the histogram of each
% block, build the data matrix V, and apply NMF on the matrix V to obtain W
% and H matrices.
% The evolution function [u] = EVOLUTION(u,Img,W,H,no_region,g,alphaa,lambda,mu,epsilon,timestep,numIter);
% evolves the curve over time and make it converge to
% achive an accurate segmentation. In this function we estimate the level
% set function phi that represent the curve and achive the segmentation.
% At the end of this code we obtain the final the final segmentation results.

% Copyright (C) <2015>  <Dimah Dera>
% % Updated Aug 2016

% Reference:
% Dera, D., Bouaynaya, N., and Fathallah-Shaykh, H. M. (2015). "Level set segmentation using non-negative
% matrix factorization of brain mri images". In The IEEE International Conference on Bioinformatics and
% Biomedicine (BIBM).

% Dimah Dera, Nidhal Bouaynaya, Hassan M. Fathallah-Shaykh, “Automated Robust Image
% Segmentation: Level Set Method using Non-Negative Matrix Factorization with Application to
% Brain MRI” in the Bulletin of Mathematical Biology (BMB). (Accepted June 2016).

if (auto==1)
 no_region = auto_detection(Img,block_size);   
end

if(auto==0)
 no_region = No_region;  
end
%------------------------------------------------------------
% choosing the initial conditions (initial contours)
 c0=1;
[ny,nx] = size(Img);
Mask=(Img>0);
if (no_region==1 || no_region==2)
initialLSF = zeros(ny,nx,1);
initialLSF(:,:,1)=c0.* Mask;   
o=initialLSF(:,:,1);
o(o==0)=-c0;
initialLSF(:,:,1)=o; 
u=initialLSF;
end
if (no_region==3 || no_region==4)
initialLSF = zeros(ny,nx,2);
initialLSF(:,:,1)=c0.* Mask;   
o=initialLSF(:,:,1);
o(o==0)=-c0;
initialLSF(:,:,1)=o; 
initialLSF(:,:,2)=c0.* Mask;   
o=initialLSF(:,:,2);
o(o==0)=-c0;
initialLSF(:,:,2)=o;
u=initialLSF;
end
if(no_region==5 || no_region==6 || no_region==7 ||no_region==8)   
initialLSF = zeros(ny,nx,3);

initialLSF(:,:,1)=c0.* Mask;   
o=initialLSF(:,:,1);
o(o==0)=-c0;
initialLSF(:,:,1)=o; 
initialLSF(:,:,2)=c0.* Mask;   
o=initialLSF(:,:,2);
o(o==0)=-c0;
initialLSF(:,:,2)=o;
initialLSF(:,:,3)=c0.* Mask;   
o=initialLSF(:,:,3);
o(o==0)=-c0;
initialLSF(:,:,3)=o;
u=initialLSF; 
end

%%------------------------------------------------------------
A=255;
se=1;        %template radius for spatial filtering
sigma=2;
epsilon=1; %Dirac regulator
timestep=0.05;
mu=0.1/timestep;  
lambda=0.001*A^2;
alphaa1=1;
alphaa2=0.5;
Gs=fspecial('gaussian',se,4);
img_smooth=conv2(Img,Gs,'same');
[Ix,Iy]=gradient(img_smooth);
ff=Ix.^2+Iy.^2;
g=1./(1+ff);  % edge indicator function.
%------------------------------------------------------------
% Dividing the image into blocks, building the data matrix V and applying
% NMF to obtain W and H factors
[W, H] = Block_Division_and_NMF_Implementation(Img,no_region,block_size);
%------------------------------------------------------------
% Evolving the curve over time by estimating the level set function phi
[M] = EVOLUTION_BRATS_data(u,Img,W,H,no_region,g,alphaa1,alphaa2,lambda,mu,epsilon,sigma,timestep);
