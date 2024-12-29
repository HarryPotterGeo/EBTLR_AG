function [G_res] = EBTLR_AG(X,gt,alpha,mu,anchor_rate)

V = length(X);                 
p=1;
C = length(unique(gt));       
N = size(X{1},1);               
weight_vector = ones(1,V)';    
M = fix( N * anchor_rate);      


%% construct Bipartite Graph
opt1. style = 1;
opt1. IterMax =50;
opt1. toy = 0;
[~, B] = My_Bipartite_Con(X,C,anchor_rate, opt1,10);


for i = 1:V
    J{i} = zeros(N,M);
    E{i} = zeros(N,M);
    Z{i} = zeros(N,M);
    Y1{i} = zeros(N,M);  %multiplier
    Y2{i} = zeros(N,M);  %multiplier

end

A = zeros(N,M);

E_tensor = cat(3, E{:});
Z_tensor = cat(3, Z{:});
B_tensor = cat(3, B{:});

kesai = -1;
iter = 1;
max_iter = 50;
% w = 1/V*ones(1,V);
sX = [N, M, V];
pho1 = 0.3;
pho2 = 0.1;
tol = 1e-6;
gamma = 1/V * ones(1, V);

while iter <= max_iter
    Zpre = Z_tensor;
    Epre = E_tensor;

        
    %% ——————更新 Z{i}——————

    for i = 1:V
        CC{i} = J{i} - Y2{i} / pho2;
        D{i} = B{i} - E{i} + Y1{i} / pho1;
    end
     
    for j = 1:V
        M1 = 2 * eye(N)* gamma(j)  + pho1 * eye(N) + pho2 * eye(N);
        M2 = 2 * A *gamma(j) + pho1 * CC{j} + pho2 * D{j};
        Z{j} = M1 \ M2;
    end
    Z_tensor = cat(3, Z{:});

    %% ——————Tensor J——————
    for v = 1:V
        QQ{v} = (Z{v} + Y2{v} / pho2);
    end
    Q_tensor = cat(3, QQ{:});
    Qg = Q_tensor(:);

    [z, ~] = wshrinkObj_weight_lp(Qg, weight_vector / pho2, sX, 0, 3, p);

    J_tensor = reshape(z, sX);

    for k = 1:V
        J{k} = J_tensor(:,:,k);
    end

    %% —————— Tensor E——————
    F = [];
    for k = 1:V
        F = [F; B{k} - Z{k} + Y1{k} / pho1];
    end
    [Econcat] = solve_l1l2(F, alpha / pho1);

    temp1 = {};
    temp1{1} = 1;
    temp1{2} = size(B{1},1) + 1;
    for k = 3:V
        temp1{k} = temp1{k-1} + size(B{k-1},1);
    end
    temp2 = {};
    temp2{1} = size(B{1},1);
    for k = 2:V
        temp2{k} = temp2{k-1} + size(B{k},1);
    end
    for k = 1:V
        E{k} = Econcat(temp1{k}:temp2{k}, :);
    end
    E_tensor = cat(3, E{:});


    %% ——————A——————
     
        Sum_M = cell(1, V);
        A_M2 = zeros(size(J{1})); %%
        A_M1 = 0;
        for j = 1:V
            Sum_M{j} = gamma(j) * J{j}; %%
            A_M1 = A_M1 + gamma(j);
            A_M2 = A_M2 + Sum_M{j};
        end
        A = (A_M1 * eye(N)) \ A_M2;

    %% ——————gamma——————
    for j = 1:V
        J_var{j} = norm(A - Z{j}, 'fro') ;
    end
   sum_gam = 0;
    for i = 1:V
        gamma(i) = (- J_var{i} / kesai)^(1 / (kesai - 1));
        sum_gam = sum_gam + gamma(i);
    end

    %% ——————Y1, Y2——————
    for i = 1:V
        Y1{i} = Y1{i} + pho1 * (B{i} - Z{i} - E{i});
        Y2{i} = Y2{i} + pho2 * (Z{i} - J{i});
    end

    %% —————pho1,2——————
    pho1 = pho1 * mu;
    pho2 = pho2 * mu;

    %% ————————————
    leq = B_tensor - Z_tensor - E_tensor;
    leqm = max(abs(leq(:)));
    difZ = max(abs(Z_tensor(:) - Zpre(:)));
    difE = max(abs(E_tensor(:) - Epre(:)));
    err = max([leqm, difZ, difE]);

    if err < tol
        break;
    end

    iter = iter + 1;
 end
toc

G_res = myNMIACCwithmean(A, gt, C)
fprintf("M: %.4f,rate:%.4f ,alpha: %.4f \n",M,anchor_rate,alpha);