function no_regions=auto_detection(Img,block_size)

%% This function is intended for detecting the number of regions automatically 
% using the Frobenius norm.

% We first divide the image into blocks, compute the histogram of each block,
% build the data matrix V, and apply NMF to ontain W and H factor matrices by using an initial 
% number of regions. We compute the Frobenius norm for W and H matrices. 
% Then, we set a threshold and increase the number of regions iteratively
% and re-apply NMF and compute the Frobenius norm until the difference
% between Frobenius norms is below the threshold.


% Inputs: 
%        Img : the original image that we would like to segment     
%        block_size: The size of the blocks  
% Outputs: 
%     no_regions : the number of regions 

% % Copyright (C) <2015>  <Dimah Dera>

% Reference:
% Dera, D., Bouaynaya, N., and Fathallah-Shaykh, H. M. (2015). "Level set segmentation using non-negative
% matrix factorization of brain mri images". In The IEEE International Conference on Bioinformatics and
% Biomedicine (BIBM).

% Dimah Dera, Nidhal Bouaynaya, Hassan M. Fathallah-Shaykh, “Automated Robust Image
% Segmentation: Level Set Method using Non-Negative Matrix Factorization with Application to
% Brain MRI” in the Bulletin of Mathematical Biology (BMB). (Accepted June 2016).


initial_no_region=1;
[Ny,Nx] = size(Img);
bin = 256;
bin_size = 256/bin;
F=zeros(Ny,Nx,bin);
P=zeros(Ny,Nx,bin);
V=zeros(256,Nx*Ny);
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
        V(: ,m)= P1(1:bin)/F(k-BlockSize,l-BlockSize,bin);
        F(k-BlockSize,l-BlockSize,:) = F(k-BlockSize,l-BlockSize,:)/F(k-BlockSize,l-BlockSize,bin);
        m=m+1;
    end
end
[W,H]=sparsenmfnnls(V,initial_no_region);% NMF

Frobenius_norm(1) = sqrt(sum(sum(abs(W.^2)))) + sqrt(sum(sum(abs(H.^2))));
thre=1; k=2;

while thre >=0.9    
initial_no_region=initial_no_region+1;   
[W,H]=sparsenmfnnls(V,initial_no_region);% NMF
Frobenius_norm(k) = sqrt(sum(sum(abs(W.^2)))) + sqrt(sum(sum(abs(H.^2))));
thre = Frobenius_norm(k)-Frobenius_norm(k-1);
k=k+1;
end
no_regions=initial_no_region;


