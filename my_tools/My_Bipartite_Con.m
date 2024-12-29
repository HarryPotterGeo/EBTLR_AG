function [centers, B] = My_Bipartite_Con(X,~,anchor_rate,opts,~,isGraph)
% Input:
%       - X: the data matrix of size nSmp * nFea * nView, where each row is a sample
%               point;
%       - c: the number of clusters;
%       - anchor_rate: the rate of sampling data points as anchors;
%       - opts: options for this algorithm
%           - style:
%               - '1': use VDA (default);
%               - '2': use randomly sampled points from the original data set;
%               - '3': use the nearest point of each cluster center generated by kmeans;
%               - '4': use centers of clusters generated by kmeans;
%           - toy:
%               - '0'; test real data (default);
%               - '1': test toy data;
% Output:
%       - centers: the selected anchors;
%       - B: Constructed Bipartite Graph for Each View
%
%   Written by Wei Xia (xd.weixia@gmail.com), written in 2020/11/20, revised in 2021/7/15
if (~exist('opts','var'))
    opts. style = 1;
    opts. toy = 0;
    opts. IterMax = 50;
end
if nargin < 6
    isGraph = 0;
end
if isGraph == 1
    B = X;
    n_view = length(X);
    [n,m] = size(X{1});
else
    k =10;
    if isfield(opts,'k')
        k = opts.k;
    end
    n_view = length(X);
    n = size(X{1},1);
    XX = [];
    for v = 1:length(X)
        XX = [X{v} XX];
    end
    m = fix(n*anchor_rate);
    B = cell(n_view,1);
    centers = cell(n_view,1);
    %%
    %%==============Anchor Selection=========%%
    if opts. style == 1 % VDA
        [~,ind,~] = VDA(XX,m);
        for v = 1:n_view
            centers{v} = X{v}(ind, :);
        end
    elseif opts. style == 2 % rand sample
        vec = randperm(n);
        ind = vec(1:m);
        for v = 1:n_view
            centers{v} = X{v}(ind, :);
        end
    elseif opts. style == 3 % KNP
        XX = [];
        for v = 1:n_view
            XX = [XX X{v}];
        end
        [~, ~, ~, ~, dis] = litekmeans(XX, m);
        [~,ind] = min(dis,[],1);
        ind = sort(ind,'ascend');
        for v = 1:n_view
            centers{v} = X{v}(ind, :);
        end
    elseif opts. style == 4 % kmeans sample
        XX = [];
        for v = 1:n_view
            XX = [XX X{v}];
            len(v) = size(X{v},2);
        end
        [~, Cen, ~, ~, ~] = litekmeans(XX, m);
        t1 = 1;
        for v=1:n_view
            t2 = t1+len(v)-1;
            centers{v} = Cen(:,t1:t2);
            t1 = t2+1;
        end
    end
    %%
    %%==========Bipartite Graph Inilization for Each View=========%%
    for v = 1:n_view
        D = L2_distance_1(X{v}', centers{v}');
        [~, idx] = sort(D, 2); % sort each row
        B{v} = zeros(n,m);
        for ii = 1:n
            id = idx(ii,1:k+1);
            di = D(ii, id);
            B{v}(ii,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);
        end
    end
end
end
