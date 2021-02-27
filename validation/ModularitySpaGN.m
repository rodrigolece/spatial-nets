function [ModularitySpa,ModularityGN,DeterrenceFct]=ModularitySpaGN(Flow,Dist,N,binsize)

% Flow: Adjacency matrix
% Dist: Distance matrix between the nodes
% N: a measure of the importance of a node (by default its strength: Dist=sum(Flow,1); for example)
% binsize: size of the bins in the estimation of the deterrence function
% (has to be tuned according to the problem)

number=size(Flow,1); %number of nodes in the system

nbox=2000; %number of bins, might need to increase it depending on your system and of level of coarse-graining

% intialisation of different vectors and matrices
DeterrenceFct=zeros(nbox,1);
normaDeterrence=zeros(nbox,1);

matrixdistance=zeros(number,number);
nullmodelGN=zeros(number,number);
nullmodelSpa=zeros(number,number);

if ~issymmetric(Flow)
    Flow=Flow+Flow'; %symmetrised matrix (doesn't change the outcome of community detection (arXiv:0812.1770))
end
degree=sum(Flow); % degree/strength of the nodes
nullN=N*N'; % matrix of the importance of nodes
matrix=Flow./nullN; % normalised adjacency matrix

%We first calculate the correlations as a function of distance

for i=1:number
    for ii=1:number

    % convert distances in binsize's units
    dist=1+ceil(Dist(i,ii)/binsize); 
    matrixdistance(i,ii)=dist;
    
    % weighted average for the deterrence function
    num=matrix(i,ii);
    DeterrenceFct(dist,1)=DeterrenceFct(dist,1)+num*N(i,1)*N(ii,1);
    normaDeterrence(dist,1)=normaDeterrence(dist,1)+N(i,1)*N(ii,1);
    end 
end

% normalisation of the deterrence function
for i=1:nbox
    if(normaDeterrence(i,1)~=0)
        DeterrenceFct(i,1)=DeterrenceFct(i,1)/normaDeterrence(i,1);
    end
end

% copmutation of the randomised correlations (preserving space), spatial
% null-model
for i=1:number
    for ii=1:number
        nullmodelSpa(i,ii)=DeterrenceFct(matrixdistance(i,ii),1);
    end 
end

% the modularity matrix for the spatial null-model
ModularitySpa=Flow-nullN.*nullmodelSpa*sum(sum(Flow))/sum(sum(nullN.*nullmodelSpa));

% the modularity matrix for the GN null-model
nullmodelGN=degree'*degree/(sum(degree)); % Newman-Girvan null-model
ModularityGN=Flow-nullmodelGN;
