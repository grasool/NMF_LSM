function [W, H] = Block_Division_and_NMF_Implementation(Img,no_region,block_size)
%% This function is intended for dividing the image into blocks, computing
% the histogram of each block, building the data matrix V, and applying NMF
% to ontain W and H factor matrices 
% Inputs: 
%        Img : the original image that we would like to segment
%        no_region : The number of regions
%        order: The parameter that saves the order of regins in the W and H
%        matrices
% Outputs: 
%     W and H matrices which are the output of the NMF

% % Copyright (C) <2015>  <Dimah Dera>

% Reference:
% Dera, D., Bouaynaya, N., and Fathallah-Shaykh, H. M. (2015). "Level set segmentation using non-negative
% matrix factorization of brain mri images". In The IEEE International Conference on Bioinformatics and
% Biomedicine (BIBM).

% Dimah Dera, Nidhal Bouaynaya, Hassan M. Fathallah-Shaykh, “Automated Robust Image
% Segmentation: Level Set Method using Non-Negative Matrix Factorization with Application to
% Brain MRI” in the Bulletin of Mathematical Biology (BMB). (Accepted June 2016).

[Ny,Nx] = size(Img);
bin = round(max(max(Img)))+1;
bin_size = 1;
F=zeros(Ny,Nx,bin);
P=zeros(Ny,Nx,bin);
V=zeros(bin,Nx*Ny);
BlockSize=block_size;
m=1;
for k=1+BlockSize:Ny+BlockSize
    for l=1+BlockSize:Nx+BlockSize
        Fi = zeros(Ny+2*BlockSize,Nx+2*BlockSize);
        for i=k-BlockSize:k+BlockSize
            for j=l-BlockSize:l+BlockSize
                Fi(i,j)=1;
            end
        end
        ndx = find(Fi(1+BlockSize:Ny+BlockSize,1+BlockSize:Nx+BlockSize));
        temp=reshape(Img(ndx),size(ndx));
        P1=hist(temp , 0:bin_size:bin);  
        F(k-BlockSize,l-BlockSize,:) = cumsum(P1(1:bin));
        P(k-BlockSize,l-BlockSize,:) = P1(1:bin)/F(k-BlockSize,l-BlockSize,bin);
        V(: ,m)= P1(1:bin)/( F(k-BlockSize,l-BlockSize,bin)+ (F(k-BlockSize,l-BlockSize,bin)==0)*eps);  
        F(k-BlockSize,l-BlockSize,:) = F(k-BlockSize,l-BlockSize,:)/F(k-BlockSize,l-BlockSize,bin);
        m=m+1;
    end
end
[W,H]=sparsenmfnnls(V,no_region);% NMF

for i=1:no_region
    for j=1:size(Img,1)*size(Img,2)
       H(i,j)=H(i,j)/sum(H(:,j));
    end
end
%---------------------------------------------------------


