function [WSR,U,V,W] = functionWMMSE(H,PK,d,K,IK)
%本函数实现了多个基站服务多个用户（每个基站均服务多个用户）的MIMO-IBC场景下的WMMSE波束成形
%输入：H  信道矩阵
%      PK 每个基站的最大发射功率，假设所有基站最大发射功率相同
%      d  原始输入信号的维度
%      K  基站数
%      IK 基站内用户数，假设所有基站服务的用户数相同
%输出：
%      WSR 和速率记录矩阵
%      U   接收波束成形矩阵
%      V   发射波束成形矩阵
%      W   权重矩阵
%

%迭代终止条件
epsilon = 1e-2; 
break_epsilon = epsilon;
%迭代次数记录
step = 0;

%矩阵初始化
V = zeros(T,d,K,IK); %所有用户的发射预编码矩阵
U = zeros(R,d,K,IK); %所有用户的接收预编码矩阵
W = zeros(d,d,K,IK); %所有用户的权重矩阵
Keep_W = zeros(d,d,K,IK); %记录当前所有用户的权重矩阵
WSR = [];%sum rate容器

% 通过信道矩阵获取信息
sz = size(H);
R = sz(1); %接收天线数
T = sz(2); %发射天线数

%初始化所有用户的发射预编码向量使其符合总功率约束
for k = 1:K
    for ik = 1:IK
        v_ik = randn(T,d)+1i*randn(T,d);
        V(:,:,k,ik) = sqrt(PK/IK)*(v_ik/norm(v_ik));
    end
end

%WMMSE核心
while break_epsilon >= epsilon && step <= 50
    step = step +1;
    %使用Keep_W记录当前的W
    Keep_W = W;
    %第一步：更新U
    for k = 1:K
        for ik = 1:IK
            % 计算J
            J = zeros(R,R,K,IK);
            for j = 1:K
                for mk = 1:IK
                   J(:,:,k,ik) = J(:,:,k,ik) + H(:,:,j,(k-1)*IK+ik)* V(:,:,j,mk)* V(:,:,j,mk)'* H(:,:,j,(k-1)*IK+ik)';
                end
            end
            J(:,:,k,ik) = J(:,:,k,ik) + eye(R);
            U(:,:,k,ik)=J(:,:,k,ik)\H(:,:,k,(k-1)*IK+ik)*V(:,:,k,ik);
        end
    end
    %第二步：更新W
    for k = 1:K
        for ik = 1:IK
            W(:,:,k,ik) = inv(eye(d)-U(:,:,k,ik)'*H(:,:,k,(k-1)*IK+ik)*V(:,:,k,ik));
        end
    end
    %第三步：寻找mu更新V
    M = zeros(T,T,K);
    for k = 1:K
        %存储迭代计算的功率
        p = 0; 
        for j = 1:K
            for mk = 1:IK
                M(:,:,k) = M(:,:,k) + H(:,:,k,(j-1)*IK+mk)'* U(:,:,j,mk)* W(:,:,j,mk)* U(:,:,j,mk)'*H(:,:,k,(j-1)*IK+mk);
            end
        end
        % 若M矩阵可逆
        if abs(det(M(:,:,k))) ~= 0
            for mk = 1:IK
                V(:,:,k,mk) = M(:,:,k)\ H(:,:,k,(k-1)*IK+mk)' * U(:,:,k,mk) * W(:,:,k,mk);
                p = p + trace(V(:,:,k,mk)*V(:,:,k,mk)');
            end
            if p <= PK
               mu = 0; 
               continue
            end
        end
        %看来不是0,继续使用bisection搜索
        p = 0;
        %计算Phi
        Phi = zeros(T,T);
        %特征分解
        [D,A] = eig(M(:,:,k));
        %计算N
        N = zeros(T,T);
        for mk = 1:IK
            N= N + H(:,:,k,(k-1)*IK+mk)'* U(:,:,k,mk)* W(:,:,k,mk)* W(:,:,k,mk)* U(:,:,k,mk)'*H(:,:,k,(k-1)*IK+mk);
        end
        Phi = D' * N * D;
        %计算功率
        mu_high = 100;
        mu_low = 0;
        while abs(p - PK)>=1e-6
            p = 0;
            mu = (mu_high + mu_low)/2;
            for t = 1:T
                p = p + Phi(t,t)/((A(t,t) + mu)^2);
            end
            %更新搜索空间
            if p > PK
                mu_low = mu;
            elseif p < PK
                mu_high = mu;
            end
        end
        for mk = 1:IK
            V(:,:,k,mk) = (M(:,:,k) + mu * eye(T))\ H(:,:,k,(k-1)*IK+mk)' * U(:,:,k,mk) * W(:,:,k,mk);
        end
    end
    
    %第四步：更新循环结束标志
    break_epsilon = 0;
    for k = 1:K
        for ik = 1:IK
            break_epsilon = break_epsilon + log(det(Keep_W(:,:,k,ik))) - log(det(W(:,:,k,ik)));
        end
    end
    break_epsilon = abs(break_epsilon);
    
    %第五步：计算和速率
    wsr = 0;%每个循环都需要初始化为0
    for k = 1:K
        for ik = 1:IK
            %计算F
            F = zeros(R,R);%对每个用户都需要初始化为0
            for j = 1:K
                for mk = 1:IK
                    if j~=k || mk~=ik
                        F = F + H(:,:,j,(k-1)*IK+ik)*V(:,:,j,mk)*V(:,:,j,mk)'*H(:,:,j,(k-1)*IK+ik)';
                    end
                end
            end
            F = F + eye(R);
            %计算WSR
            wsr = wsr + log2(det(eye(R)+H(:,:,k,(k-1)*IK+ik)*V(:,:,k,ik)*V(:,:,k,ik)'*H(:,:,k,(k-1)*IK+ik)'/F));
        end
    end
    WSR = [WSR,norm(wsr)];
end
end




