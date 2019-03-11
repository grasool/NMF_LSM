function [M] = EVOLUTION_BRATS_data(u,Img,W,H,no_region,g,alphaa1,alphaa2,lambda,mu,epsilon,sigma,timestep)
%% This function is the evolution function for evolving the curve over time 
% and make it converge at the end to the optimal solution (final segmentation).
% in this code we estimate the level set function u anf the bias field bias
% iteratively.
% Inputs:
%       u: the level set function (the initial contour.
%       Img: the image after removing the skull and non-brain structures.
%       W and H : the output matrices of the NMF.
%       no_region: the number of regions.
%       g: the edge indication function.
%       alphaa1,alphaa1,  lambda, and mu:  the weighting parameters
%       epsilon: the prameter of the Dirac delta function
%       timestep: the step size parameter for the gradient descent 
%       numIter: the number of iteration for the gradient descent.
% Outputs:
%       u: the final estimated level set function (final contour).
%       M: The segmented regions
% % Copyright (C) <2015>  <Dimah Dera>
% % Updated Aug 2016

% Reference:
% Dera, D., Bouaynaya, N., and Fathallah-Shaykh, H. M. (2015). "Level set segmentation using non-negative
% matrix factorization of brain mri images". In The IEEE International Conference on Bioinformatics and
% Biomedicine (BIBM).

% Dimah Dera, Nidhal Bouaynaya, Hassan M. Fathallah-Shaykh, “Automated Robust Image
% Segmentation: Level Set Method using Non-Negative Matrix Factorization with Application to
% Brain MRI” in the Bulletin of Mathematical Biology (BMB). (Accepted June 2016).

[ny,nx] = size(Img);
[vx,vy]=gradient(g);
H_term=zeros(ny,nx,no_region);

for i=1:no_region
H_term(:,:,i)=normalize01(reshape(H(i,:), [nx ny])'); 
end
if (no_region==1)
    W=[W W]; H_term1=zeros(ny,nx,no_region+1); H_term1(:,:,1)=H_term(:,:,1); H_term1(:,:,2) =H_term(:,:,1) ;        
end
bias=ones(size(Img));
if(no_region == 1 || no_region ==2)
for k=1:100
    u(:,:,1)=NeumannBoundCond(u(:,:,1));  
    [K1,Nx1,Ny1] = curvature_central(u(:,:,1)); 
    DiracU1=Dirac(u(:,:,1),epsilon);     
    H1 = Heaviside(u(:,:,1),epsilon);     
    M(:,:,1)=H1;     M(:,:,2)=1-H1; 
    if (no_region==1)
         bias = compute_b(Img,W,bias,M,sigma,no_region+1);
         e(:,:,:) = compute_e(Img,W,bias,M,sigma,no_region+1);
         DataF1 = alphaa1.*( e(:,:,1)-e(:,:,2) )+ alphaa2.*( M(:,:,1)- H_term1(:,:,1) - (M(:,:,2)- H_term1(:,:,2)) );
    elseif(no_region==2)
        bias = compute_b(Img,W,bias,M,sigma,no_region);
        e(:,:,:) = compute_e(Img,W,bias,M,sigma,no_region);
        DataF1 = alphaa1.*( e(:,:,1)-e(:,:,2) )+ alphaa2.*( M(:,:,1)- H_term(:,:,1) - (M(:,:,2)- H_term(:,:,2)) );
    end    
    ImageTerm1= - DiracU1.*DataF1; 
    weightedLengthTerm1=lambda*DiracU1.*(vx.*Nx1 + vy.*Ny1 + g.*K1);    
    penalizingTerm1=mu*(4*del2(u(:,:,1))-K1);     
    u(:,:,1)=u(:,:,1)+timestep.*( penalizingTerm1 + weightedLengthTerm1+ImageTerm1); %      
end
for i=1:no_region
    M(:,:,i)=M(:,:,i)>=0.9;          
end
end
%------------
if(no_region == 3 || no_region ==4)
t = multi_lsm(Img,no_region-1) ; 
U1 = imquantize(Img,t);
U=zeros(ny,nx,no_region);
for i=1:no_region
    U(:,:,i)=U1==i;
end
for k=1:100
    u(:,:,1)=NeumannBoundCond(u(:,:,1));  u(:,:,2)=NeumannBoundCond(u(:,:,2));
    [K1,Nx1,Ny1] = curvature_central(u(:,:,1)); [K2,Nx2,Ny2] = curvature_central(u(:,:,2));
    DiracU1=Dirac(u(:,:,1),epsilon);  DiracU2=Dirac(u(:,:,2),epsilon);   
    H1 = Heaviside(u(:,:,1),epsilon); H2 = Heaviside(u(:,:,2),epsilon);    
    if (no_region==4)
        M(:,:,1)=H1.*H2;     M(:,:,2)=H1.*(1-H2);     M(:,:,3)=(1-H1).*H2; M(:,:,4) = (1-H1).*(1-H2);
        bias = compute_b(Img,W,bias,M,sigma,no_region);
        e(:,:,:) = compute_e(Img,W,bias,M,sigma,no_region);
        DataF1 = alphaa1.*( (e(:,:,1)-e(:,:,2)-e(:,:,3)+e(:,:,4)).*H2+(e(:,:,2)-e(:,:,4)) )+ alphaa2.*(H2.*( (M(:,:,1)- H_term(:,:,1)) - (M(:,:,3)- H_term(:,:,3)) )+(1-H2).*((M(:,:,2)- H_term(:,:,2)) - (M(:,:,4)- H_term(:,:,4)) ));
        DataF2 = alphaa1.*( (e(:,:,1)-e(:,:,2)-e(:,:,3)+e(:,:,4)).*H1+(e(:,:,3)-e(:,:,4)) )+ alphaa2.*(H1.*( (M(:,:,1)- H_term(:,:,1)) - (M(:,:,2)- H_term(:,:,2)) )+(1-H1).*((M(:,:,3)- H_term(:,:,3)) - (M(:,:,4)- H_term(:,:,4)) ));
    elseif (no_region==3)
        M(:,:,1)=H1.*H2;     M(:,:,2)=H1.*(1-H2);     M(:,:,3)=(1-H1);    
        bias = compute_b(Img,W,bias,M,sigma,no_region);
        e(:,:,:) = compute_e(Img,W,bias,M,sigma,no_region);
        DataF1 =  alphaa1.*(e(:,:,1).*H2 + e(:,:,2).*(1-H2) - e(:,:,3)) + alphaa2.* (  H2.*(M(:,:,1)- H_term(:,:,1))+ (1-H2).*(M(:,:,2)-H_term(:,:,2)) - (M(:,:,3) -H_term(:,:,3)) );
        DataF2 =  H1.*(alphaa1.*(e(:,:,1)-e(:,:,2)) + alphaa2.*((M(:,:,1)- H_term(:,:,1)) - (M(:,:,2)-H_term(:,:,2))));
    end    
    ImageTerm1= - DiracU1.*DataF1; ImageTerm2= - DiracU2.*DataF2;  
    weightedLengthTerm1=lambda*DiracU1.*(vx.*Nx1 + vy.*Ny1 + g.*K1);     weightedLengthTerm2=lambda*DiracU2.*(vx.*Nx2 + vy.*Ny2 + g.*K2);
    penalizingTerm1=mu*(4*del2(u(:,:,1))-K1);     penalizingTerm2=mu*(4*del2(u(:,:,2))-K2);    
    u(:,:,1)=u(:,:,1)+timestep.*( penalizingTerm1 + weightedLengthTerm1+ImageTerm1); %
    u(:,:,2)=u(:,:,2)+timestep.*( penalizingTerm2 + weightedLengthTerm2+ImageTerm2);     
end
for i=1:no_region
    M(:,:,i)=bwareaopen(U(:,:,i),5)>=0.9;          
end
end
%---------------
if(no_region == 5 || no_region ==6 || no_region ==7 ||no_region ==8)
t = multi_lsm(Img,no_region-1) ; 
U1 = imquantize(Img,t);
U=zeros(ny,nx,no_region);
for i=1:no_region
    U(:,:,i)=U1==i;
end
for k=1:100
    u(:,:,1)=NeumannBoundCond(u(:,:,1));  u(:,:,2)=NeumannBoundCond(u(:,:,2));u(:,:,3)=NeumannBoundCond(u(:,:,3));
    [K1,Nx1,Ny1] = curvature_central(u(:,:,1)); [K2,Nx2,Ny2] = curvature_central(u(:,:,2)); [K3,Nx3,Ny3] = curvature_central(u(:,:,3));
    DiracU1=Dirac(u(:,:,1),epsilon); DiracU2=Dirac(u(:,:,2),epsilon); DiracU3=Dirac(u(:,:,3),epsilon); 
    H1 = Heaviside(u(:,:,1),epsilon); H2 = Heaviside(u(:,:,2),epsilon); H3 = Heaviside(u(:,:,3),epsilon);   
    if (no_region==5)
        M(:,:,1)=H1.*H2.*H3; M(:,:,2)=H1.*H2.*(1-H3); M(:,:,3)=H1.*(1-H2); M(:,:,4)=(1-H1).*H2; M(:,:,5)=(1-H1).*(1-H2);        
        bias = compute_b(Img,W,bias,M,sigma,no_region);
        e(:,:,:) = compute_e(Img,W,bias,M,sigma,no_region);
        DataF1 = alphaa1.*((e(:,:,1)-e(:,:,2)).*H2.*H3+(e(:,:,2)-e(:,:,3)-e(:,:,4)+e(:,:,5)).*H2+(e(:,:,3)-e(:,:,5)))+ alphaa2.*( H2.*H3.*(M(:,:,1)- H_term(:,:,1))+H2.*(1-H3).*(M(:,:,2)- H_term(:,:,2))+ (1-H2).*(M(:,:,3)- H_term(:,:,3))- H2.*(M(:,:,4)- H_term(:,:,4))-(1-H2).*(M(:,:,5)- H_term(:,:,5)));
        DataF2 = alphaa1.*((e(:,:,1)-e(:,:,2)).*H1.*H3+1.*(e(:,:,2)-e(:,:,3)-e(:,:,4)+e(:,:,5)).*H1+1.*(e(:,:,4)-e(:,:,5)))+ alphaa2.*( H1.*H3.*(M(:,:,1)- H_term(:,:,1))+H1.*(1-H3).*(M(:,:,2)- H_term(:,:,2))+ (1-H1).*(M(:,:,4)- H_term(:,:,4))- H1.*(M(:,:,3)- H_term(:,:,3))-(1-H1).*(M(:,:,5)- H_term(:,:,5)));
        DataF3 = (alphaa1.*(e(:,:,1)-e(:,:,2)) + alphaa2.*(M(:,:,1)- H_term(:,:,1) - (M(:,:,2)- H_term(:,:,2)))).*H1.*H2;
    elseif (no_region==6)
        M(:,:,1)=H1.*H2.*H3; M(:,:,2)=H1.*H2.*(1-H3); M(:,:,3)=H2.*H3.*(1-H1); M(:,:,4)=H1.*(1-H2).*(1-H3); M(:,:,5)=H3.*(1-H2);  M(:,:,6)=(1-H1).*(1-H3);     
        bias = compute_b(Img,W,bias,M,sigma,no_region);
        e(:,:,:) = compute_e(Img,W,bias,M,sigma,no_region);
        DataF1 = alphaa1.*((e(:,:,1)-e(:,:,2)-e(:,:,3)+e(:,:,4)).*H2.*H3+(e(:,:,2)-e(:,:,4)).*H2+(e(:,:,4)-e(:,:,6)).*(1-H3)  ) + alphaa2.*(H2.*H3.*(M(:,:,1)- H_term(:,:,1))+H2.*(1-H3).*(M(:,:,2)- H_term(:,:,2))-H2.*H3.*(M(:,:,3)- H_term(:,:,3))+(1-H2).*(1-H3).*(M(:,:,4)- H_term(:,:,4))-(1-H3).*(M(:,:,6)- H_term(:,:,6)) );
        DataF2 = alphaa1.*((e(:,:,1)-e(:,:,2)-e(:,:,3)+e(:,:,4)).*H1.*H3+(e(:,:,2)-e(:,:,4)).*H1+(e(:,:,3)-e(:,:,5)).*H3   )+ alphaa2.*(H1.*H3.*(M(:,:,1)- H_term(:,:,1))+H1.*(1-H3).*(M(:,:,2)- H_term(:,:,2))+ H3.*(1-H1).*(M(:,:,3)- H_term(:,:,3))- H2.*(1-H3).*(M(:,:,4)- H_term(:,:,4))- H3.*(M(:,:,5)- H_term(:,:,5)));
        DataF3 = alphaa1.*((e(:,:,1)-e(:,:,2)-e(:,:,3)+e(:,:,4)).*H1.*H2+(e(:,:,3)-e(:,:,5)).*H2-(e(:,:,4)-e(:,:,6)).*H1+(e(:,:,5)-e(:,:,6))) + alphaa2.*(H1.*H2.*(M(:,:,1)- H_term(:,:,1))+H1.*H2.*(M(:,:,2)- H_term(:,:,2))+H2.*(1-H1).*(M(:,:,3)- H_term(:,:,3))-H1.*(1-H2).*(M(:,:,4)- H_term(:,:,4))+(1-H2).*(M(:,:,5)- H_term(:,:,5)) -(1-H1).*(M(:,:,6)- H_term(:,:,6)));
    elseif (no_region==7)
        M(:,:,1)=H1.*H2.*H3; M(:,:,2)=H1.*H2.*(1-H3); M(:,:,3)=H1.*H3.*(1-H2); M(:,:,4)=H1.*(1-H2).*(1-H3); M(:,:,5)=H2.*H3.*(1-H1);  M(:,:,6)=(1-H1).*H2.*(1-H3);   M(:,:,7)=(1-H1).*(1-H2);  
        bias = compute_b(Img,W,bias,M,sigma,no_region);
        e(:,:,:) = compute_e(Img,W,bias,M,sigma,no_region);
        DataF1 = alphaa1.*(e(:,:,1).*H2.*H3+e(:,:,2).*H2.*(1-H3)+e(:,:,3).*H3.*(1-H2)+e(:,:,4).*(1-H2).*(1-H3)-e(:,:,5).*H2.*H3-e(:,:,6).*H2.*(1-H3)-e(:,:,7).*(1-H2))....
        + alphaa2.*((M(:,:,1)- H_term(:,:,1)).*H2.*H3+(M(:,:,2)- H_term(:,:,2)).*H2.*(1-H3)+(M(:,:,3)- H_term(:,:,3)).*H3.*(1-H2)+(M(:,:,4)- H_term(:,:,4)).*(1-H2).*(1-H3)-(M(:,:,5)- H_term(:,:,5)).*H2.*H3-(M(:,:,6)- H_term(:,:,6)).*H2.*(1-H3)-(M(:,:,7)- H_term(:,:,7)).*(1-H2)) ;
        DataF2 = alphaa1.*(e(:,:,1).*H1.*H3+e(:,:,2).*H1.*(1-H3)-e(:,:,3).*H1.*H3-e(:,:,4).*H1.*(1-H3)+e(:,:,5).*(1-H1).*H3+e(:,:,6).*(1-H1).*(1-H3)-e(:,:,7).*(1-H1)) +...
        alphaa2.*( (M(:,:,1)- H_term(:,:,1)).*H1.*H3+(M(:,:,2)- H_term(:,:,2)).*H1.*(1-H3)-(M(:,:,3)- H_term(:,:,3)).*H1.*H3-(M(:,:,4)- H_term(:,:,4)).*H1.*(1-H3)+(M(:,:,5)- H_term(:,:,5)).*(1-H1).*H3+(M(:,:,6)- H_term(:,:,6)).*(1-H1).*(1-H3)-(M(:,:,7)- H_term(:,:,7)).*(1-H1)) ;
        DataF3 =alphaa1.* ( (e(:,:,1)-e(:,:,2)-e(:,:,3)+e(:,:,4)-e(:,:,5)+e(:,:,6)).*H1.*H2+(e(:,:,3)-e(:,:,4)).*H1+(e(:,:,5)-e(:,:,6)).*H2) +...
         alphaa2.*( (M(:,:,1)- H_term(:,:,1)).*H1.*H2-(M(:,:,2)- H_term(:,:,2)).*H1.*H2+(M(:,:,3)- H_term(:,:,3)).*H1.*(1-H2)-(M(:,:,4)- H_term(:,:,4)).*H1.*(1-H2)+(M(:,:,5)- H_term(:,:,5)).*H2.*(1-H1)-(M(:,:,6)- H_term(:,:,6)).*H2.*(1-H1)) ;
    elseif (no_region==8)
        M(:,:,1)=H1.*H2.*H3; M(:,:,2)=H1.*H2.*(1-H3); M(:,:,3)=H1.*H3.*(1-H2); M(:,:,4)=H1.*(1-H2).*(1-H3); M(:,:,5)=(1-H1).*H2.*H3;  M(:,:,6)=(1-H1).*H2.*(1-H3);   M(:,:,7)=(1-H1).*(1-H2).*H3;  M(:,:,8)=(1-H1).*(1-H2).*(1-H3);  
        bias = compute_b(Img,W,bias,M,sigma,no_region);
        e(:,:,:) = compute_e(Img,W,bias,M,sigma,no_region);
        DataF1 = alphaa1.*((e(:,:,1)-e(:,:,2)-e(:,:,3)+e(:,:,4)-e(:,:,5)+e(:,:,6)+e(:,:,7)-e(:,:,8)).*H2.*H3+(e(:,:,2)-e(:,:,4)-e(:,:,6)+e(:,:,8)).*H2+(e(:,:,3)-e(:,:,4)-e(:,:,7)+e(:,:,8)).*H3+e(:,:,4)- e(:,:,8))+...
        alphaa2.*( (M(:,:,1)-H_term(:,:,1)).*H2.*H3+(M(:,:,2)-H_term(:,:,2)).*H2.*(1-H3)+(M(:,:,3)-H_term(:,:,3)).*H3.*(1-H2)+(M(:,:,4)-H_term(:,:,4)).*(1-H2).*(1-H3)-(M(:,:,5)-H_term(:,:,5)).*H2.*H3-(M(:,:,6)-H_term(:,:,6)).*H2.*(1-H3)-(M(:,:,7)-H_term(:,:,7)).*(1-H2).*H3-(M(:,:,8)-H_term(:,:,8)).*(1-H2).*(1-H3));
        DataF2 = alphaa1.*((e(:,:,1)-e(:,:,2)-e(:,:,3)+e(:,:,4)-e(:,:,5)+e(:,:,6)+e(:,:,7)-e(:,:,8)).*H1.*H3+(e(:,:,2)-e(:,:,4)-e(:,:,6)+e(:,:,8)).*H1+(e(:,:,5)-e(:,:,6)-e(:,:,7)+e(:,:,8)).*H3+e(:,:,6)-e(:,:,8))+...
        alphaa1.*((M(:,:,1)- H_term(:,:,1)).*H1.*H3+(M(:,:,2)- H_term(:,:,2)).*H1.*(1-H3)-(M(:,:,3)-H_term(:,:,3)).*H1.*H3-(M(:,:,4)- H_term(:,:,4)).*H1.*(1-H3)+(M(:,:,5)- H_term(:,:,5)).*(1-H1).*H3+(M(:,:,6)- H_term(:,:,6)).*(1-H1).*(1-H3)-(M(:,:,7)- H_term(:,:,7)).*(1-H1).*H3 -(M(:,:,8)- H_term(:,:,8)).*(1-H1).*(1-H3));
        DataF3 = alphaa1.*((e(:,:,1)-e(:,:,2)-e(:,:,3)+e(:,:,4)-e(:,:,5)+e(:,:,6)+e(:,:,7)-e(:,:,8)).*H1.*H2+(e(:,:,3)-e(:,:,4)-e(:,:,7)+e(:,:,8)).*H1+(e(:,:,5)-e(:,:,6)-e(:,:,7)+e(:,:,8)).*H2+e(:,:,7)-e(:,:,8))+...
        alphaa2.* ((M(:,:,1)- H_term(:,:,1)).*H1.*H2-(M(:,:,2)- H_term(:,:,2)).*H1.*H2+(M(:,:,3)-H_term(:,:,3)).*H1.*(1-H2)-(M(:,:,4)- H_term(:,:,4)).*H1.*(1-H2)+(M(:,:,5)- H_term(:,:,5)).*H2.*(1-H1)-(M(:,:,6)- H_term(:,:,6)).*H2.*(1-H1)+(M(:,:,7)- H_term(:,:,7)).*(1-H2).*(1-H1)-(M(:,:,8)- H_term(:,:,8)).*(1-H2).*(1-H1)) ;
    end    
    ImageTerm1= - DiracU1.*DataF1; ImageTerm2= - DiracU2.*DataF2;  ImageTerm3= - DiracU3.*DataF3;   
    weightedLengthTerm1=lambda*DiracU1.*(vx.*Nx1 + vy.*Ny1 + g.*K1);weightedLengthTerm2=lambda*DiracU2.*(vx.*Nx2 + vy.*Ny2 + g.*K2); weightedLengthTerm3=lambda*DiracU3.*(vx.*Nx3 + vy.*Ny3 + g.*K3);
    penalizingTerm1=mu*(4*del2(u(:,:,1))-K1);penalizingTerm2=mu*(4*del2(u(:,:,2))-K2); penalizingTerm3=mu*(4*del2(u(:,:,3))-K3);    
    u(:,:,1)=u(:,:,1)+timestep.*( penalizingTerm1 + weightedLengthTerm1+ImageTerm1); 
    u(:,:,2)=u(:,:,2)+timestep.*( penalizingTerm2 + weightedLengthTerm2+ImageTerm2);
    u(:,:,3)=u(:,:,3)+timestep.*( penalizingTerm3 + weightedLengthTerm3+ImageTerm3);      
       
end
for i=1:no_region
    M(:,:,i)=bwareaopen(U(:,:,i),5)>=0.9;          
end
end
 
%--------------------------------------------------------
%-------------------------------------------------------
%% The function of computing e(x) = (I(x) - mean )^2 / 2*variance
function e = compute_e(Img,W,b,M,sigma,no_region)
Ksigma=fspecial('gaussian',round(2*sigma)*2+1,sigma); % Gaussian kernel
KONE=conv2(ones(size(Img)),Ksigma,'same');
KONE_Img = Img.^2.*KONE;
KB1 = conv2(b,Ksigma,'same');
KB2 = conv2(b.^2,Ksigma,'same');
C = zeros(no_region,1);
e=zeros([size(Img),no_region]);
for i=1:no_region
    [~, C(i)]= max(W(:,i));
    N = KB1.*Img.*M(:,:,i);
    D = KB2.*M(:,:,i);
    C(i)=sum(N(:))/(sum(D(:))+(sum(D(:))==0));
    e(:,:,i) = KONE_Img - 2*Img.* (C(i)).*KB1 + (C(i))^2*KB2;
end
% -------------------------------------------------------
% The function of computing the direvative of the bias field
function  b = compute_b(Img,W,b,M,sigma,no_region)
Ksigma=fspecial('gaussian',round(2*sigma)*2+1,sigma); % Gaussian kernel
KB1 = conv2(b,Ksigma,'same');
KB2 = conv2(b.^2,Ksigma,'same');
num=zeros(size(Img));
de=num;
C=zeros(no_region,1);
for i=1:no_region
    [~, C(i)]= max(W(:,i));  
    N = KB1.*Img.*M(:,:,i);
    D = KB2.*M(:,:,i);
    C(i)=sum(N(:))/(sum(D(:))+(sum(D(:))==0));    
    num=num+(C(i))*M(:,:,i);
    de=de+(C(i))^2*M(:,:,i);       
end
numerator = conv2(num.*Img,Ksigma,'same');
denominator = conv2(de,Ksigma,'same');
b = numerator./ (denominator +(denominator==0)*eps);
%-------------------------------------------------
%-------------------------------------------------
% The function of computing the heaviside function
function h = Heaviside(x,epsilon)
h=0.5*sin(atan(x./epsilon))+0.5;
% -------------------------------------------------------
% The function of computing the dirac delta function
function f = Dirac(x, epsilon)
f=0.5*cos(atan(x./epsilon))*epsilon./(epsilon^2.+ x.^2);
% ---------------------------------------------------------
% The function of computing the curvatur
function [K,Nx1,Ny1] = curvature_central(u1)
[ux1,uy1]=gradient(u1); 
normDu1=sqrt(ux1.^2 + uy1.^2 + 1e-10); % the norm of the gradient plus a small possitive number 
Nx1=ux1./normDu1;
Ny1=uy1./normDu1;
[nxx,~]=gradient(Nx1);  
[~,nyy]=gradient(Ny1);
K=nxx+nyy;
% ---------------------------------------------------------
function g = NeumannBoundCond(f)
% Make a function satisfy Neumann boundary condition
[nrow,ncol] = size(f);
g = f;
g([1 nrow],[1 ncol]) = g([3 nrow-2],[3 ncol-2]);  
g([1 nrow],2:end-1) = g([3 nrow-2],2:end-1);          
g(2:end-1,[1 ncol]) = g(2:end-1,[3 ncol-2]);
% ---------------------------------------------------------
function t = multi_lsm(varargin)
narginchk(1,2);
[A, N] = parse_inputs(varargin{:});
if (isempty(A))
    warning(message('images:multithresh:degenerateInput',N))
    t = getDegenerateThresholds(A, N);    
    metric = 0.0;
    return;
end
num_bins = 256;
[p, minA, maxA] = getpdf(A, num_bins);
if (isempty(p))   
    warning(message('images:multithresh:degenerateInput',N))
    t = getThreshForNoPdf(minA, maxA, N);        
    metric = 0.0;
    return;
end
omega = cumsum(p);
mu = cumsum(p .* (1:num_bins)');
mu_t = mu(end);
if (N < 3)     
    sigma_b_squared = calcFullObjCriteriaMatrix(N, num_bins, omega, mu, mu_t);     
    % Find the location of the maximum value of sigma_b_squared.  
    maxval = max(sigma_b_squared(:));     
    isvalid_maxval = isfinite(maxval);
    
    if isvalid_maxval
        % Find the bin with maximum value. If the maximum extends over
        % several bins, average together the locations.
        switch N
            case 1
                idx = find(sigma_b_squared == maxval);
                % Find the intensity associated with the bin
                t = mean(idx) - 1;
            case 2
                [maxR, maxC] = find(sigma_b_squared == maxval);
                % Find the intensity associated with the bin
                t = mean([maxR maxC],1) - 1;                
        end        
    else
        [isDegenerate, uniqueVals] = checkForDegenerateInput(A, N);
        if isDegenerate
            warning(message('images:multithresh:degenerateInput',N));
        else
            warning(message('images:multithresh:noConvergence'));
        end
        t = getDegenerateThresholds(uniqueVals, N);
        metric = 0.0;        
    end    
else    
    initial_thresh = linspace(0, num_bins-1, N+2);
    initial_thresh = initial_thresh(2:end-1); % Retain N thresholds   
    % Set optimization parameters    
    options = optimset('TolX',1,'Display','off');    
    % Find optimum using fminsearch 
    [t, minval] = fminsearch1(@(t) objCriteriaND(t, ...
        num_bins, omega, mu, mu_t), initial_thresh, options);    
    maxval = -minval;    
    isvalid_maxval = ~(isinf(maxval) || isnan(maxval));
    if isvalid_maxval        
        t = round(t);  
    end            
end
% Prepare output values
if isvalid_maxval    
    % Map back to original scale as input A
    t = map2OriginalScale(t, minA, maxA);
    if nargout > 1    
        % Compute the effectiveness metric        
        metric = maxval/(sum(p.*(((1:num_bins)' - mu_t).^2)));        
    end    
else    
    [isDegenerate, uniqueVals] = checkForDegenerateInput(A, N);  
    if isDegenerate
        warning(message('images:multithresh:degenerateInput',N));
        t = getDegenerateThresholds(uniqueVals, N);
        metric = 0.0;
    else
        warning(message('images:multithresh:noConvergence'));
        % Return latest available solution
        t = map2OriginalScale(t, minA, maxA);
        if nargout > 1
            % Compute the effectiveness metric
            metric = maxval/(sum(p.*(((1:num_bins)' - mu_t).^2)));
        end
    end        
end
%--------------------------------------------------------------------------
function [A, N] = parse_inputs(varargin)
A = varargin{1};
validateattributes(A,{'uint8','uint16','int16','double','single'}, ...
    {'nonsparse', 'real'}, mfilename,'A',1);
if (nargin == 2)
    N = varargin{2};
    validateattributes(N,{'numeric'},{'integer','scalar','positive','<=',20}, ...
        mfilename,'N',2);
else
    N = 1; % Default N
end
%--------------------------------------------------------------------------
function [p, minA, maxA] = getpdf(A,num_bins)
A = A(:);
if isfloat(A)    
    % If A is an float images then scale the data to the range [0 1] while
    % taking care of special cases such as Infs and NaNs.    
    % Remove NaNs from consideration.
    % A cannot be empty here because we checked for it earlier.  
    A(isnan(A)) = [];   
    if isempty(A)
        % The case when A was full of only NaNs.
        minA = NaN;
        maxA = NaN;
        p = [];
        return;
    end    
    % Scale A to [0-1]
    idxFinite = isfinite(A);
    % If there are finite elements, then scale them between [0-1]. Maintain
    % Infs and -Infs as is so that they get included in the pdf.
    if any(idxFinite)
        minA = min(A(idxFinite));
        maxA = max(A(idxFinite));
        if(minA == maxA)
            p = [];
            return;
        end        
        % Call to BSXFUN below is equivalent to A = (A - minA)/(maxA - minA);
        A = bsxfun(@rdivide,bsxfun(@minus, A, minA),maxA - minA);        
    else
        % One of many possibilities: all Infs, all -Infs, mixture of Infs
        % and -Infs, mixture of Infs with NaNs.
        minA = min(A);
        maxA = max(A);
        p = [];
        return;
    end
else
    % If A is an integer image then no need to handle special cases for
    % Infs and NaNs.    
    minA = min(A);
    maxA = max(A);
    if(minA == maxA)
        p = [];
        return;
    else
        % Call to BSXFUN below is equivalent to A = single(A - minA)./single(maxA - minA);      
        A = bsxfun(@rdivide,single(bsxfun(@minus, A, minA)),single(maxA - minA));
    end    
end
counts = imhist(A,num_bins);
p = counts / sum(counts);
%--------------------------------------------------------------------------
function sigma_b_squared_val = objCriteriaND(thresh, num_bins, omega, mu, mu_t)
% 'thresh' has intensities [0-255], but 'boundaries' are the indices [1
% 256].
boundaries = round(thresh)+1; 
% Constrain 'boundaries' to:
% 1. be strictly increasing, 
% 2. have the lowest value > 1 (i.e. minimum 2), 
% 3. have highest value < num_bins (i.e. maximum num_bins-1).
if (~all(diff([1 boundaries num_bins]) > 0))
    sigma_b_squared_val = Inf;
    return;
end
boundaries = [boundaries num_bins]; 
sigma_b_squared_val = omega(boundaries(1)).*((mu(boundaries(1))./omega(boundaries(1)) - mu_t).^2);

for kk = 2:length(boundaries)
    omegaKK = omega(boundaries(kk)) - omega(boundaries(kk-1));
    muKK = (mu(boundaries(kk)) - mu(boundaries(kk-1)))/omegaKK;
    sigma_b_squared_val = sigma_b_squared_val + (omegaKK.*((muKK - mu_t).^2)); % Eqn. 14 in Otsu's paper
end
if (isfinite(sigma_b_squared_val))
    sigma_b_squared_val = -sigma_b_squared_val; % To do maximization using fminsearch.
else
    sigma_b_squared_val = Inf;
end
%--------------------------------------------------------------------------
function sigma_b_squared = calcFullObjCriteriaMatrix(N, num_bins, omega, mu, mu_t)
if (N == 1)    
    sigma_b_squared = (mu_t * omega - mu).^2 ./ (omega .* (1 - omega));    
elseif (N == 2)    
    % Rows represent thresh(1) (lower threshold) and columns represent
    % thresh(2) (higher threshold).
    omega0 = repmat(omega,1,num_bins);
    mu_0_t = repmat(bsxfun(@minus,mu_t,mu./omega),1,num_bins);
    omega1 = bsxfun(@minus, omega.', omega);
    mu_1_t = bsxfun(@minus,mu_t,(bsxfun(@minus, mu.', mu))./omega1);    
    % Set entries corresponding to non-viable solutions to NaN
    [allPixR, allPixC] = ndgrid(1:num_bins,1:num_bins); 
    pixNaN = allPixR >= allPixC; % Enforce thresh(1) < thresh(2)
    omega0(pixNaN) = NaN;
    omega1(pixNaN) = NaN;          
    term1 = omega0.*(mu_0_t.^2);    
    term2 = omega1.*(mu_1_t.^2);    
    omega2 = 1 - (omega0+omega1);
    omega2(omega2 <= 0) = NaN; % Avoid divide-by-zero Infs in term3    
    term3 = ((omega0.*mu_0_t + omega1.*mu_1_t ).^2)./omega2;    
    sigma_b_squared = term1 + term2 + term3;
end
%--------------------------------------------------------------------------
function sclThresh = map2OriginalScale(thresh, minA, maxA)
normFactor = 255;
sclThresh = double(minA) + thresh/normFactor*(double(maxA) - double(minA));
sclThresh = cast(sclThresh,'like',minA);
%--------------------------------------------------------------------------
function [isDegenerate, uniqueVals] = checkForDegenerateInput(A, N)
uniqueVals = unique(A(:))'; % Note: 'uniqueVals' is returned in sorted order. 
% Ignore NaNs because they are ignored in computation. Ignore Infs because
% Infs are mapped to extreme bins during histogram computation and are
% therefore not unique values.
uniqueVals(isinf(uniqueVals) | isnan(uniqueVals)) = []; 
isDegenerate = (numel(uniqueVals) <= N);
%--------------------------------------------------------------------------
function thresh = getThreshForNoPdf(minA, maxA, N)
if isnan(minA)
    % If minA = NaN => maxA = NaN. All NaN input condition.
    minA = 1; 
    maxA = 1;
end
if (N == 1)
    thresh = minA;
else
    if (minA == maxA)
        % Flat image, i.e. only one unique value (not counting Infs and
        % -Infs) exists
        thresh = getDegenerateThresholds(minA, N);
    else
        % Only scenario: A full of Infs and -Infs => minA = -Inf and maxA =
        % Inf
        thresh = getDegenerateThresholds([minA maxA], N);
    end
end
%--------------------------------------------------------------------------
function thresh = getDegenerateThresholds(uniqueVals, N)
% Notes:
% 1) 'uniqueVals' must be in sorted (ascending) order
% 2) For predictable behavior, 'uniqueVals' should not have NaNs 
% 3) For predictable behavior for all datatypes including uint8, N must be < 255
if isempty(uniqueVals)
    thresh = cast(1:N,'like', uniqueVals);
    return;
end
% 'thresh' will always have all the elements of 'uniqueVals' in it.
thresh = uniqueVals;
thNeeded1 = N - numel(thresh);
if (thNeeded1 > 0)
    
    % More values are needed to fill 'thresh'. Start filling 'thresh' from
    % the lower end starting with 1.
    
    if (uniqueVals(1) > 1)
        % If uniqueVals(1) > 1, we can directly fill some (or maybe all)
        % values starting from 1, without checking for uniqueness.        
        thresh = [cast(1:min(thNeeded1,ceil(uniqueVals(1))-1), 'like', uniqueVals)...
            thresh];        
    end    
    thNeeded2 = N - numel(thresh);
    if (thNeeded2  > 0)    
        lenThreshOrig = length(thresh);               
        thresh = [thresh zeros(1,thNeeded2)]; % Create empty entries, thresh datatype presevrved
        uniqueVals_d = double(uniqueVals); % Needed to convert to double for correct uniquness check     
        threshCandidate = max(floor(uniqueVals(1)),0); % Always non-negative, threshCandidate datatype presevrved    
        q = 1;
        while q <= thNeeded2
            threshCandidate = threshCandidate + 1;
            threshCandidate_d = double(threshCandidate); % Needed to convert to double for correct uniquness check
            if any(abs(uniqueVals_d - threshCandidate_d) ...
                    < eps(threshCandidate_d)) 
                % The candidate value already exists, so don't use it.               
                continue;
            else
                thresh(lenThreshOrig + q) = threshCandidate; % Append at the end
                q = q + 1;
            end
        end        
        thresh = sort(thresh);       
    end                             
end  
% ----------------------------------------------------------------------------

function [x,fval,exitflag,output] = fminsearch1(funfcn,x,options,varargin)
defaultopt = struct('Display','notify','MaxIter','200*numberOfVariables',...
    'MaxFunEvals','200*numberOfVariables','TolX',1e-4,'TolFun',1e-4, ...
    'FunValCheck','off','OutputFcn',[],'PlotFcns',[]);

% If just 'defaults' passed in, return the default options in X
if nargin==1 && nargout <= 1 && isequal(funfcn,'defaults')
    x = defaultopt;
    return
end
if nargin<3, options = []; end
% Detect problem structure input
if nargin == 1
    if isa(funfcn,'struct') 
        [funfcn,x,options] = separateOptimStruct(funfcn);
    else % Single input and non-structure
        error(message('MATLAB:fminsearch:InputArg'));
    end
end
if nargin == 0
    error(message('MATLAB:fminsearch:NotEnoughInputs'));
end
% Check for non-double inputs
if ~isa(x,'double')
  error(message('MATLAB:fminsearch:NonDoubleInput'))
end
n = numel(x);
numberOfVariables = n;
printtype = optimget(options,'Display',defaultopt,'fast');
tolx = optimget(options,'TolX',defaultopt,'fast');
tolf = optimget(options,'TolFun',defaultopt,'fast');
maxfun = optimget(options,'MaxFunEvals',defaultopt,'fast');
maxiter = optimget(options,'MaxIter',defaultopt,'fast');
funValCheck = strcmp(optimget(options,'FunValCheck',defaultopt,'fast'),'on');

% In case the defaults were gathered from calling: optimset('fminsearch'):
if ischar(maxfun)
    if isequal(lower(maxfun),'200*numberofvariables')
        maxfun = 200*numberOfVariables;
    else
        error(message('MATLAB:fminsearch:OptMaxFunEvalsNotInteger'))
    end
end
if ischar(maxiter)
    if isequal(lower(maxiter),'200*numberofvariables')
        maxiter = 200*numberOfVariables;
    else
        error(message('MATLAB:fminsearch:OptMaxIterNotInteger'))
    end
end
switch printtype
    case {'notify','notify-detailed'}
        prnt = 1;
    case {'none','off'}
        prnt = 0;
    case {'iter','iter-detailed'}
        prnt = 3;
    case {'final','final-detailed'}
        prnt = 2;
    case 'simplex'
        prnt = 4;
    otherwise
        prnt = 1;
end
% Handle the output
outputfcn = optimget(options,'OutputFcn',defaultopt,'fast');
if isempty(outputfcn)
    haveoutputfcn = false;
else
    haveoutputfcn = true;
    xOutputfcn = x; % Last x passed to outputfcn; has the input x's shape
    % Parse OutputFcn which is needed to support cell array syntax for OutputFcn.
    outputfcn = createCellArrayOfFunctions(outputfcn,'OutputFcn');
end

% Handle the plot
plotfcns = optimget(options,'PlotFcns',defaultopt,'fast');
if isempty(plotfcns)
    haveplotfcn = false;
else
    haveplotfcn = true;
    xOutputfcn = x; % Last x passed to plotfcns; has the input x's shape
    % Parse PlotFcns which is needed to support cell array syntax for PlotFcns.
    plotfcns = createCellArrayOfFunctions(plotfcns,'PlotFcns');
end
header = ' Iteration   Func-count     min f(x)         Procedure';
% Convert to function handle as needed.
funfcn = fcnchk(funfcn,length(varargin));
% Add a wrapper function to check for Inf/NaN/complex values
if funValCheck   
    varargin = {funfcn, varargin{:}};
    funfcn = @checkfun;
end
n = numel(x);
% Initialize parameters
rho = 1; chi = 2; psi = 0.5; sigma = 0.5;
onesn = ones(1,n);
two2np1 = 2:n+1;
one2n = 1:n;
xin = x(:); % Force xin to be a column vector
v = zeros(n,n+1); fv = zeros(1,n+1);
v(:,1) = xin;    % Place input guess in the simplex! (credit L.Pfeffer at Stanford)
x(:) = xin;    % Change x to the form expected by funfcn
fv(:,1) = funfcn(x,varargin{:});
func_evals = 1;
itercount = 0;
how = '';
% Initialize the output and plot functions.
if haveoutputfcn || haveplotfcn
    [xOutputfcn, optimValues, stop] = callOutputAndPlotFcns(outputfcn,plotfcns,v(:,1),xOutputfcn,'init',itercount, ...
        func_evals, how, fv(:,1),varargin{:});
    if stop
        [x,fval,exitflag,output] = cleanUpInterrupt(xOutputfcn,optimValues);
        if  prnt > 0
            disp(output.message)
        end
        return;
    end
end

% Print out initial f(x) as 0th iteration
if prnt == 3
    disp(' ')
    disp(header)
    fprintf(' %5.0f        %5.0f     %12.6g         %s\n', itercount, func_evals, fv(1), how);
elseif prnt == 4
    clc
    formatsave.format = get(0,'format');
    formatsave.formatspacing = get(0,'formatspacing');
    % reset format when done
    oc1 = onCleanup(@()set(0,'format',formatsave.format));
    oc2 = onCleanup(@()set(0,'formatspacing',formatsave.formatspacing));
    format compact
    format short e
    disp(' ')
    disp(how)
    
end
% OutputFcn and PlotFcns call
if haveoutputfcn || haveplotfcn
    [xOutputfcn, optimValues, stop] = callOutputAndPlotFcns(outputfcn,plotfcns,v(:,1),xOutputfcn,'iter',itercount, ...
        func_evals, how, fv(:,1),varargin{:});
    if stop  % Stop per user request.
        [x,fval,exitflag,output] = cleanUpInterrupt(xOutputfcn,optimValues);
        if  prnt > 0
            disp(output.message)
        end
        return;
    end
end
usual_delta = 0.05;             % 5 percent deltas for non-zero terms
zero_term_delta = 0.00025;      % Even smaller delta for zero elements of x
for j = 1:n
    y = xin;
    if y(j) ~= 0
        y(j) = (1 + usual_delta)*y(j);
    else
        y(j) = zero_term_delta;
    end
    v(:,j+1) = y;
    x(:) = y; f = funfcn(x,varargin{:});
    fv(1,j+1) = f;
end

% sort so v(1,:) has the lowest function value
[fv,j] = sort(fv);
v = v(:,j);

how = 'initial simplex';
itercount = itercount + 1;
func_evals = n+1;
if prnt == 3
    fprintf(' %5.0f        %5.0f     %12.6g         %s\n', itercount, func_evals, fv(1), how)
elseif prnt == 4
    disp(' ')
    disp(how)   
end
% OutputFcn and PlotFcns call
if haveoutputfcn || haveplotfcn
    [xOutputfcn, optimValues, stop] = callOutputAndPlotFcns(outputfcn,plotfcns,v(:,1),xOutputfcn,'iter',itercount, ...
        func_evals, how, fv(:,1),varargin{:});
    if stop  % Stop per user request.
        [x,fval,exitflag,output] = cleanUpInterrupt(xOutputfcn,optimValues);
        if  prnt > 0
            disp(output.message)
        end
        return;
    end
end
exitflag = 1;
while func_evals < maxfun && itercount < maxiter
    if max(abs(fv(1)-fv(two2np1))) <= max(tolf,10*eps(fv(1))) && ...
            max(max(abs(v(:,two2np1)-v(:,onesn)))) <= max(tolx,10*eps(max(v(:,1))))
        break
    end
    % xbar = average of the n (NOT n+1) best points
    xbar = sum(v(:,one2n), 2)/n;
    xr = (1 + rho)*xbar - rho*v(:,end);
    x(:) = xr; fxr = funfcn(x,varargin{:});
    func_evals = func_evals+1;
    
    if fxr < fv(:,1)
       
        xe = (1 + rho*chi)*xbar - rho*chi*v(:,end);
        x(:) = xe; fxe = funfcn(x,varargin{:});
        func_evals = func_evals+1;
        if fxe < fxr
            v(:,end) = xe;
            fv(:,end) = fxe;
            how = 'expand';
        else
            v(:,end) = xr;
            fv(:,end) = fxr;
            how = 'reflect';
        end
    else % fv(:,1) <= fxr
        if fxr < fv(:,n)
            v(:,end) = xr;
            fv(:,end) = fxr;
            how = 'reflect';
        else % fxr >= fv(:,n)
            % Perform contraction
            if fxr < fv(:,end)
                
                xc = (1 + psi*rho)*xbar - psi*rho*v(:,end);
                x(:) = xc; fxc = funfcn(x,varargin{:});
                func_evals = func_evals+1;
                
                if fxc <= fxr
                    v(:,end) = xc;
                    fv(:,end) = fxc;
                    how = 'contract outside';
                else
                    
                    how = 'shrink';
                end
            else
                
                xcc = (1-psi)*xbar + psi*v(:,end);
                x(:) = xcc; fxcc = funfcn(x,varargin{:});
                func_evals = func_evals+1;
                
                if fxcc < fv(:,end)
                    v(:,end) = xcc;
                    fv(:,end) = fxcc;
                    how = 'contract inside';
                else
                    
                    how = 'shrink';
                end
            end
            if strcmp(how,'shrink')
                for j=two2np1
                    v(:,j)=v(:,1)+sigma*(v(:,j) - v(:,1));
                    x(:) = v(:,j); fv(:,j) = funfcn(x,varargin{:});
                end
                func_evals = func_evals + n;
            end
        end
    end
    [fv,j] = sort(fv);
    v = v(:,j);
    itercount = itercount + 1;
    if prnt == 3
        fprintf(' %5.0f        %5.0f     %12.6g         %s\n', itercount, func_evals, fv(1), how)
    elseif prnt == 4
        disp(' ')
        disp(how)
        
    end
    % OutputFcn and PlotFcns call
    if haveoutputfcn || haveplotfcn
        [xOutputfcn, optimValues, stop] = callOutputAndPlotFcns(outputfcn,plotfcns,v(:,1),xOutputfcn,'iter',itercount, ...
            func_evals, how, fv(:,1),varargin{:});
        if stop  % Stop per user request.
            [x,fval,exitflag,output] = cleanUpInterrupt(xOutputfcn,optimValues);
            if  prnt > 0
                disp(output.message)
            end
            return;
        end
    end
end   % while

x(:) = v(:,1);
fval = fv(:,1);
output.iterations = itercount;
output.funcCount = func_evals;
output.algorithm = 'Nelder-Mead simplex direct search';
% OutputFcn and PlotFcns call
if haveoutputfcn || haveplotfcn
    callOutputAndPlotFcns(outputfcn,plotfcns,x,xOutputfcn,'done',itercount, func_evals, how, fval, varargin{:});
end
if func_evals >= maxfun
    msg = getString(message('MATLAB:fminsearch:ExitingMaxFunctionEvals', sprintf('%f',fval)));
    if prnt > 0
        disp(' ')
        disp(msg)
    end
    exitflag = 0;
elseif itercount >= maxiter
    msg = getString(message('MATLAB:fminsearch:ExitingMaxIterations', sprintf('%f',fval)));
    if prnt > 0
        disp(' ')
        disp(msg)
    end
    exitflag = 0;
else
    msg = ... 
      getString(message('MATLAB:optimfun:fminsearch:OptimizationTerminatedXSatisfiesCriteria', ...
               sprintf('%e',tolx), sprintf('%e',tolf)));
    if prnt > 1
        disp(' ')
        disp(msg)
    end
    exitflag = 1;
end
output.message = msg;
%--------------------------------------------------------------------------
function [xOutputfcn, optimValues, stop] = callOutputAndPlotFcns(outputfcn,plotfcns,x,xOutputfcn,state,iter,...
    numf,how,f,varargin)
% CALLOUTPUTANDPLOTFCNS assigns values to the struct OptimValues and then calls the
% outputfcn/plotfcns.
%
optimValues.iteration = iter;
optimValues.funccount = numf;
optimValues.fval = f;
optimValues.procedure = how;
xOutputfcn(:) = x;  % Set x to have user expected size
stop = false;
% Call output functions
if ~isempty(outputfcn)
    switch state
        case {'iter','init'}
            stop = callAllOptimOutputFcns(outputfcn,xOutputfcn,optimValues,state,varargin{:}) || stop;
        case 'done'
            callAllOptimOutputFcns(outputfcn,xOutputfcn,optimValues,state,varargin{:});
        otherwise
            error(message('MATLAB:fminsearch:InvalidState'))
    end
end
% Call plot functions
if ~isempty(plotfcns)
    switch state
        case {'iter','init'}
            stop = callAllOptimPlotFcns(plotfcns,xOutputfcn,optimValues,state,varargin{:}) || stop;
        case 'done'
            callAllOptimPlotFcns(plotfcns,xOutputfcn,optimValues,state,varargin{:});
        otherwise
            error(message('MATLAB:fminsearch:InvalidState'))
    end
end
%--------------------------------------------------------------------------
function [x,FVAL,EXITFLAG,OUTPUT] = cleanUpInterrupt(xOutputfcn,optimValues)
% CLEANUPINTERRUPT updates or sets all the output arguments of FMINBND when the optimization
% is interrupted.

callAllOptimPlotFcns('cleanuponstopsignal');
x = xOutputfcn;
FVAL = optimValues.fval;
EXITFLAG = -1;
OUTPUT.iterations = optimValues.iteration;
OUTPUT.funcCount = optimValues.funccount;
OUTPUT.algorithm = 'Nelder-Mead simplex direct search';
OUTPUT.message = getString(message('MATLAB:fminsearch:OptimizationTerminatedPrematurelyByUser'));
%--------------------------------------------------------------------------
function f = checkfun(x,userfcn,varargin)
% CHECKFUN checks for complex or NaN results from userfcn.
f = userfcn(x,varargin{:});
% Note: we do not check for Inf as FMINSEARCH handles it naturally.
if isnan(f)
    error(message('MATLAB:fminsearch:checkfun:NaNFval', localChar( userfcn )));  
elseif ~isreal(f)
    error(message('MATLAB:fminsearch:checkfun:ComplexFval', localChar( userfcn )));  
end
%--------------------------------------------------------------------------
function strfcn = localChar(fcn)
% Convert the fcn to a string for printing
if ischar(fcn)
    strfcn = fcn;
elseif isa(fcn,'inline')
    strfcn = char(fcn);
elseif isa(fcn,'function_handle')
    strfcn = func2str(fcn);
else
    try
        strfcn = char(fcn);
    catch
        strfcn = getString(message('MATLAB:fminsearch:NameNotPrintable'));
    end
end