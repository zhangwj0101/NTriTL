function [Results, Gt, t1, t2] = TriTL(Train_Data,Test_Data,Parameter_Setting)

% function [Results, Gt] = TriTL(Train_Data,Test_Data,Parameter_Setting)

% The common program for TriTL, which can deal with multiple classes,
% multiple source domains and multiple target domains

%%%% Input:
% The parameter Train_data stores the file pathes of training data and the
% corresponding labels
% The parameter Test_data stores the file pathes of test data and the
% corresponding labels
% The parameter Parameterfile stores the parameter setting information,
% including the number of identical concepts k1, the number of alike concepts k2,
% the number of distinct concepts k3, the number of iterations

%%%% Output
% The variable Results is a matrix with size (numIteration + 1) x numTarget, where
% numIteration is the number of iterations, numTarget is the number of
% target domains. Results record the detailed prediction accuracies of all target domains in each iteration.

% The variable Gt is a matrix with size n x c, where n is the number of
% instances in all target domains, specifically, n = n_1 + ... + nt (n_t is
% the number of instances in t-th target domain), c is the number of
% classes
%
% Note that if you want to handle large data set, you should set larget
% memory for Matlab. You can set it in the file C:\boot.ini (This may not
% be true in your system), change '/fastdetect' to '/fastdetect /3GB'.
%
% Be good luck for your research, if you have any questions, you can
% contact the email: zhuangfz@ics.ict.ac.cn
%%
%read the parameters
fid=fopen(Parameter_Setting);
numK_1 = str2num(fgetl(fid));
numK_2 = str2num(fgetl(fid));
numK_3 = str2num(fgetl(fid));
numIteration = str2num(fgetl(fid));
numK = numK_1 + numK_2 + numK_3;
fclose(fid);

iscsvread = 0;
fid=fopen(Train_Data);
numSource = str2num(fgetl(fid));
fid1=fopen(fgetl(fid));
A = fgetl(fid1);
B = find(A == ',');
if length(B) == 2
    iscsvread = 1;
end
fclose(fid1);
fclose(fid);
labelset = [];
if iscsvread == 1
    % read source domain data
    fid=fopen(Train_Data);
    numSource = str2num(fgetl(fid));
    TrainX = [];
    TrainY = [];
    numTrain = [];
    for i = 1:numSource
        A = csvread(fgetl(fid));
        B = spconvert(A);
        TrainX = [TrainX B];
        C = textread(fgetl(fid));
        TrainY = [TrainY C'];
        numTrain(1,i) = length(C);
        labelset = union(labelset,C');
        clear A;
        clear B;
        clear C;
    end
    fclose(fid);
    
    % read target domain data
    fid=fopen(Test_Data);
    numTarget = str2num(fgetl(fid));
    TestX = [];
    TestY = [];
    numTest = [];
    for i = 1:numTarget
        A = csvread(fgetl(fid));
        B = spconvert(A);
        TestX = [TestX B];
        C = textread(fgetl(fid));
        TestY = [TestY C'];
        numTest(1,i) = length(C);
        labelset = union(labelset,C');
        clear A;
        clear B;
        clear C;
        clear E;
    end
    fclose(fid);
else
    % read source domain data
    fid=fopen(Train_Data);
    numSource = str2num(fgetl(fid));
    TrainX = [];
    TrainY = [];
    numTrain = [];
    for i = 1:numSource
        A = load(fgetl(fid));
        B = spconvert(A);
        TrainX = [TrainX B];
        C = textread(fgetl(fid));
        TrainY = [TrainY C'];
        numTrain(1,i) = length(C);
        labelset = union(labelset,C');
        clear A;
        clear B;
        clear C;
    end
    fclose(fid);
    
    % read target domain data
    fid=fopen(Test_Data);
    numTarget = str2num(fgetl(fid));
    TestX = [];
    TestY = [];
    numTest = [];
    for i = 1:numTarget
        A = load(fgetl(fid));
        B = spconvert(A);
        TestX = [TestX B];
        C = textread(fgetl(fid));
        TestY = [TestY C'];
        numTest(1,i) = length(C);
        labelset = union(labelset,C');
        clear A;
        clear B;
        clear C;
        clear E;
    end
    fclose(fid);
end

numC = length(labelset);
numFeature = size(TestX,1);
start = 1;
t1 = clock;
if start == 1
    DataSetX = [TrainX TestX];
    Learn.Verbosity = 1;
    Learn.Max_Iterations = 20;
    Learn.heldout = .1; % for tempered EM only, percentage of held out data
    Learn.Min_Likelihood_Change = 1;
    Learn.Folding_Iterations = 20; % for TEM only: number of fiolding in iterations
    Learn.TEM = 0; %tempered or not tempered
    [Pw_z,Pz_d,Pd,Li,perp,eta] = pLSA(DataSetX,[],numK_1+numK_2,Learn); %start PLSA
    %xlswrite(strcat('pwz_','common_selected','.xls'),Pw_z);
    %     csvwrite(strcat('pwz_common','.pwz'),Pw_z);
end
t2 = clock;

%% Following are Initializaitons
% pzw = csvread(strcat('pwz_common','.pwz'));
% pzw = ones(size(pzw,1),size(pzw,2));
pzw = Pw_z;
F1 = [];
F1 = pzw(:,1:numK_1);
Fs2 = [];
Fs3 = [];
for i = 1:numSource
    Fs2 = [Fs2, pzw(:,1+numK_1:numK_1+numK_2)];
end
Fs3 = ones(numFeature,numK_3*numSource)/numFeature;

Ft2 = [];
Ft3 = [];
for i = 1:numTarget
    Ft2 = [Ft2, pzw(:,1+numK_1:numK_1+numK_2)];
end
Ft3 = ones(numFeature,numK_3*numTarget)/numFeature;
clear pzw;

Gs = zeros(sum(numTrain),numC);
for i = 1:size(Gs,1)
    Gs(i,TrainY(i)) = 1;
end
% In our paper, Gt is assigned as the predicted results by supervised
% classifiers
% For simplicity, we randomly initialize Gt
% %% The initialization of the target-domain label
w_models = [];
for i = 1:numSource
    pos = 0;
    if i > 1
        for t = 1:i-1
            pos = pos + numTrain(t);
        end
    end
    if i == 1
        pos = 0;
    end
    TempTrainX = TrainX(:,pos+1:pos+numTrain(i));
    TempTrainY = TrainY(:,pos+1:pos+numTrain(i));
    for v = 1:length(TempTrainY)
        if TempTrainY(v) > 1
            TempTrainY(v) = -1;
        end
    end
    
    TempTrainXY = scale_cols(TempTrainX,TempTrainY);
    fprintf('.....................................\n');
    w00 = zeros(size(TempTrainXY,1),1);
    lambda = exp(linspace(-0.5,6,20));
    wbest = [];
    f1max = -inf;
    for j = 1:length(lambda)
        w_0 = train_cg(TempTrainXY,w00,lambda(j));
        f1 = logProb(TempTrainXY,w_0);
        if f1 > f1max
            f1max = f1;
            wbest = w_0;
            se_lambda = lambda(j);
        end
    end
    w_models = [w_models wbest];
    clear TempTrainX;
    clear TempTrainY;
    clear TempTrainXY;
end
% csvwrite(strcat('lg_models/','model_lg.model'),w_models);

TempGt = rand(sum(numTest),numC);
% for i = 1:size(TempGt,1)
%     if sum(TempGt(i,:)) > 0
%         TempGt(i,:) = TempGt(i,:)/sum(TempGt(i,:));
%     else
%         TempGt(i,:) = 1/numC;
%     end
% end

% w_models = csvread(strcat('lg_models/','model_lg.model'));
for i = 1:numTarget
    pos = 0;
    if i > 1
        for t = 1:i-1
            pos = pos + numTest(t);
        end
    end
    if i == 1
        pos = 0;
    end
    TempTestX = TestX(:,pos+1:pos+numTest(i));
    for j = 1:numSource
        wbest = w_models(:,j);
        ptemp = 1./(1 + exp(-wbest'*TempTestX));
    end
    TempGt(pos+1:pos+numTest(i),:) = TempGt(pos+1:pos+numTest(i),:) + [(ptemp'+0.5)/2 ((1-ptemp)'+0.5)/2];
    clear TempTestX;
end
TempGt = TempGt/numSource;
Gt = TempGt;

%%
S1 = ones(numK_1,numC)/numC;
S2 = ones(numK_2,numC)/numC;
Ss3 = ones(numK_3,numC*numSource)/numC;
St3 = ones(numK_3,numC*numTarget)/numC;

for i = 1:size(TrainX,2)
    TrainX(:,i) = TrainX(:,i)/sum(TrainX(:,i));
end
for i = 1:size(TestX,2)
    TestX(:,i) = TestX(:,i)/sum(TestX(:,i));
end

%% Caculate the objective value
stepLen = 1000;
%% the initial accuracy
iter_results = [];
for i = 1:numTarget
    pos = 0;
    if i > 1
        for t = 1:i-1
            pos = pos + numTest(t);
        end
    end
    if i == 1
        pos = 0;
    end
    pzd = Gt(pos+1:pos+numTest(i),:);
    nCorrect = 0;
    for j = 1:size(pzd,1)
        [va vi] = max(pzd(j,:));
        if labelset(vi) == TestY(pos+j)
            nCorrect = nCorrect + 1;
        end
    end
    iter_results(1,i+1) = nCorrect/(numTest(i));
    iter_results(1,1) = 0;
end
iter_results

VXGs = []; %%% temp variable
VGGs = []; %%% temp variable
for i = 1:numSource
    pos = 0;
    if i > 1
        for t = 1:i-1
            pos = pos + numTrain(t);
        end
    end
    if i == 1
        pos = 0;
    end
    VXGs = [VXGs TrainX(:,pos+1:pos+numTrain(i))*Gs(pos+1:pos+numTrain(i),:)];
    VGGs = [VGGs Gs(pos+1:pos+numTrain(i),:)'*Gs(pos+1:pos+numTrain(i),:)];
end
% [F,H]=NMF(TestX,20);
% [S,GG]=NMF(H,2);
%
% csvwrite(strcat('F.dat'),F);
% csvwrite(strcat('S.dat'),S);
% csvwrite(strcat('GG.dat'),GG');
%% start to iterate
for iterID = 1:numIteration
    index =  1;
    index = index + 1;
    
    VXGt = [];
    VGGt = [];
    for i = 1:numTarget
        pos = 0;
        if i > 1
            for t = 1:i-1
                pos = pos + numTest(t);
            end
        end
        if i == 1
            pos = 0;
        end
        VXGt = [VXGt TestX(:,pos+1:pos+numTest(i))*Gt(pos+1:pos+numTest(i),:)];
        VGGt = [VGGt Gt(pos+1:pos+numTest(i),:)'*Gt(pos+1:pos+numTest(i),:)];
    end
    
    A = zeros(numFeature,numK_1);
    B = zeros(numFeature,numK_1);
    for i = 1:numSource
        A = A + VXGs(:,(i-1)*numC+1:i*numC)*S1';
        B = B + F1*S1*VGGs(:,(i-1)*numC+1:i*numC)*S1' + Fs2(:,(i-1)*numK_2+1:i*numK_2)*S2*VGGs(:,(i-1)*numC+1:i*numC)*S1'+Fs3(:,(i-1)*numK_3+1:i*numK_3)*Ss3(:,(i-1)*numC+1:i*numC)*VGGs(:,(i-1)*numC+1:i*numC)*S1';
    end
    for i = 1:numTarget
        A = A + VXGt(:,(i-1)*numC+1:i*numC)*S1';
        B = B + F1*S1*VGGt(:,(i-1)*numC+1:i*numC)*S1' + Ft2(:,(i-1)*numK_2+1:i*numK_2)*S2*VGGt(:,(i-1)*numC+1:i*numC)*S1'+Ft3(:,(i-1)*numK_3+1:i*numK_3)*St3(:,(i-1)*numC+1:i*numC)*VGGt(:,(i-1)*numC+1:i*numC)*S1';
    end
    [xs ys] = find(B == 0);
    index = index + 1;
    if ~isempty(xs)
        for q = 1:size(xs,1)
            B(xs(q,1),ys(q,1)) = 1;
        end
    end
    temp_F1 = F1.*sqrt(A./B);
    
    temp_Fs2 = [];
    for i = 1:numSource
        A = VXGs(:,(i-1)*numC+1:i*numC)*S2';
        B = Fs2(:,(i-1)*numK_2+1:i*numK_2)*S2*VGGs(:,(i-1)*numC+1:i*numC)*S2'+F1*S1*VGGs(:,(i-1)*numC+1:i*numC)*S2' + Fs3(:,(i-1)*numK_3+1:i*numK_3)*Ss3(:,(i-1)*numC+1:i*numC)*VGGs(:,(i-1)*numC+1:i*numC)*S2';
        [xs ys] = find(B == 0);
        index = index + 1;
        if ~isempty(xs)
            for q = 1:size(xs,1)
                B(xs(q,1),ys(q,1)) = 1;
            end
        end
        temp_Fs2 = [temp_Fs2 Fs2(:,(i-1)*numK_2+1:i*numK_2).*sqrt(A./B)];
    end
    
    temp_Ft2 = [];
    for i = 1:numTarget
        A = VXGt(:,(i-1)*numC+1:i*numC)*S2';
        B = Ft2(:,(i-1)*numK_2+1:i*numK_2)*S2*VGGt(:,(i-1)*numC+1:i*numC)*S2'+F1*S1*VGGt(:,(i-1)*numC+1:i*numC)*S2' + Ft3(:,(i-1)*numK_3+1:i*numK_3)*St3(:,(i-1)*numC+1:i*numC)*VGGt(:,(i-1)*numC+1:i*numC)*S2';
        [xs ys] = find(B == 0);
        index = index + 1;
        if ~isempty(xs)
            for q = 1:size(xs,1)
                B(xs(q,1),ys(q,1)) = 1;
            end
        end
        temp_Ft2 = [temp_Ft2 Ft2(:,(i-1)*numK_2+1:i*numK_2).*sqrt(A./B)];
    end
    
    temp_Fs3 = [];
    for i = 1:numSource
        A = VXGs(:,(i-1)*numC+1:i*numC)*Ss3(:,(i-1)*numC+1:i*numC)';
        B = Fs3(:,(i-1)*numK_3+1:i*numK_3)*Ss3(:,(i-1)*numC+1:i*numC)*VGGs(:,(i-1)*numC+1:i*numC)*Ss3(:,(i-1)*numC+1:i*numC)'+F1*S1*VGGs(:,(i-1)*numC+1:i*numC)*Ss3(:,(i-1)*numC+1:i*numC)' + Fs2(:,(i-1)*numK_2+1:i*numK_2)*S2*VGGs(:,(i-1)*numC+1:i*numC)*Ss3(:,(i-1)*numC+1:i*numC)';
        [xs ys] = find(B == 0);
        index = index + 1;
        if ~isempty(xs)
            for q = 1:size(xs,1)
                B(xs(q,1),ys(q,1)) = 1;
            end
        end
        temp_Fs3 = [temp_Fs3 Fs3(:,(i-1)*numK_3+1:i*numK_3).*sqrt(A./B)];
    end
    
    temp_Ft3 = [];
    for i = 1:numTarget
        A = VXGt(:,(i-1)*numC+1:i*numC)*St3(:,(i-1)*numC+1:i*numC)';
        B = Ft2(:,(i-1)*numK_2+1:i*numK_2)*S2*VGGt(:,(i-1)*numC+1:i*numC)*St3(:,(i-1)*numC+1:i*numC)'+F1*S1*VGGt(:,(i-1)*numC+1:i*numC)*St3(:,(i-1)*numC+1:i*numC)' + Ft3(:,(i-1)*numK_3+1:i*numK_3)*St3(:,(i-1)*numC+1:i*numC)*VGGt(:,(i-1)*numC+1:i*numC)*St3(:,(i-1)*numC+1:i*numC)';
        [xs ys] = find(B == 0);
        index = index + 1;
        if ~isempty(xs)
            for q = 1:size(xs,1)
                B(xs(q,1),ys(q,1)) = 1;
            end
        end
        temp_Ft3 = [temp_Ft3 Ft3(:,(i-1)*numK_3+1:i*numK_3).*sqrt(A./B)];
    end
    
    A = zeros(numK_1,numC);
    B = zeros(numK_1,numC);
    for i = 1:numSource
        A = A + MatrixProduce(F1',VXGs(:,(i-1)*numC+1:i*numC),stepLen);
        B = B + MatrixProduce(F1',F1,stepLen)*S1*VGGs(:,(i-1)*numC+1:i*numC) + MatrixProduce(F1',Fs2(:,(i-1)*numK_2+1:i*numK_2),stepLen)*S2*VGGs(:,(i-1)*numC+1:i*numC) + MatrixProduce(F1',Fs3(:,(i-1)*numK_3+1:i*numK_3),stepLen)*Ss3(:,(i-1)*numC+1:i*numC)*VGGs(:,(i-1)*numC+1:i*numC);
    end
    for i = 1:numTarget
        A = A + MatrixProduce(F1',VXGt(:,(i-1)*numC+1:i*numC),stepLen);
        B = B + MatrixProduce(F1',F1,stepLen)*S1*VGGt(:,(i-1)*numC+1:i*numC) + MatrixProduce(F1',Ft2(:,(i-1)*numK_2+1:i*numK_2),stepLen)*S2*VGGt(:,(i-1)*numC+1:i*numC) + MatrixProduce(F1',Ft3(:,(i-1)*numK_3+1:i*numK_3),stepLen)*St3(:,(i-1)*numC+1:i*numC)*VGGt(:,(i-1)*numC+1:i*numC);
    end
    [xs ys] = find(B == 0);
    index = index + 1;
    if ~isempty(xs)
        for q = 1:size(xs,1)
            B(xs(q,1),ys(q,1)) = 1;
        end
    end
    temp_S1 = S1.*sqrt(A./B);
    
    A = zeros(numK_2,numC);
    B = zeros(numK_2,numC);
    for i = 1:numSource
        A = A + MatrixProduce(Fs2(:,(i-1)*numK_2+1:i*numK_2)',VXGs(:,(i-1)*numC+1:i*numC),stepLen);
        B = B + MatrixProduce(Fs2(:,(i-1)*numK_2+1:i*numK_2)',F1,stepLen)*S1*VGGs(:,(i-1)*numC+1:i*numC) + MatrixProduce(Fs2(:,(i-1)*numK_2+1:i*numK_2)',Fs2(:,(i-1)*numK_2+1:i*numK_2),stepLen)*S2*VGGs(:,(i-1)*numC+1:i*numC) + MatrixProduce(Fs2(:,(i-1)*numK_2+1:i*numK_2)',Fs3(:,(i-1)*numK_3+1:i*numK_3),stepLen)*Ss3(:,(i-1)*numC+1:i*numC)*VGGs(:,(i-1)*numC+1:i*numC);
    end
    for i = 1:numTarget
        A = A + MatrixProduce(Ft2(:,(i-1)*numK_2+1:i*numK_2)',VXGt(:,(i-1)*numC+1:i*numC),stepLen);
        B = B + MatrixProduce(Ft2(:,(i-1)*numK_2+1:i*numK_2)',F1,stepLen)*S1*VGGt(:,(i-1)*numC+1:i*numC) + MatrixProduce(Ft2(:,(i-1)*numK_2+1:i*numK_2)',Ft2(:,(i-1)*numK_2+1:i*numK_2),stepLen)*S2*VGGt(:,(i-1)*numC+1:i*numC) + MatrixProduce(Ft2(:,(i-1)*numK_2+1:i*numK_2)',Ft3(:,(i-1)*numK_3+1:i*numK_3),stepLen)*St3(:,(i-1)*numC+1:i*numC)*VGGt(:,(i-1)*numC+1:i*numC);
    end
    [xs ys] = find(B == 0);
    index = index + 1;
    if ~isempty(xs)
        for q = 1:size(xs,1)
            B(xs(q,1),ys(q,1)) = 1;
        end
    end
    temp_S2 = S2.*sqrt(A./B);
    
    temp_Ss3 = [];
    for i = 1:numSource
        A = MatrixProduce(Fs3(:,(i-1)*numK_3+1:i*numK_3)',VXGs(:,(i-1)*numC+1:i*numC),stepLen);
        B = MatrixProduce(Fs3(:,(i-1)*numK_3+1:i*numK_3)',F1,stepLen)*S1*VGGs(:,(i-1)*numC+1:i*numC) + MatrixProduce(Fs3(:,(i-1)*numK_3+1:i*numK_3)',Fs2(:,(i-1)*numK_2+1:i*numK_2),stepLen)*S2*VGGs(:,(i-1)*numC+1:i*numC) + MatrixProduce(Fs3(:,(i-1)*numK_3+1:i*numK_3)',Fs3(:,(i-1)*numK_3+1:i*numK_3),stepLen)*Ss3(:,(i-1)*numC+1:i*numC)*VGGs(:,(i-1)*numC+1:i*numC);
        [xs ys] = find(B == 0);
        index = index + 1;
        if ~isempty(xs)
            for q = 1:size(xs,1)
                B(xs(q,1),ys(q,1)) = 1;
            end
        end
        temp_Ss3 = [temp_Ss3 Ss3(:,(i-1)*numC+1:i*numC).*sqrt(A./B)];
    end
    
    temp_St3 = [];
    for i = 1:numTarget
        A = MatrixProduce(Ft3(:,(i-1)*numK_3+1:i*numK_3)',VXGt(:,(i-1)*numC+1:i*numC),stepLen);
        B = MatrixProduce(Ft3(:,(i-1)*numK_3+1:i*numK_3)',F1,stepLen)*S1*VGGt(:,(i-1)*numC+1:i*numC) + MatrixProduce(Ft3(:,(i-1)*numK_3+1:i*numK_3)',Ft2(:,(i-1)*numK_2+1:i*numK_2),stepLen)*S2*VGGt(:,(i-1)*numC+1:i*numC) + MatrixProduce(Ft3(:,(i-1)*numK_3+1:i*numK_3)',Ft3(:,(i-1)*numK_3+1:i*numK_3),stepLen)*St3(:,(i-1)*numC+1:i*numC)*VGGt(:,(i-1)*numC+1:i*numC);
        [xs ys] = find(B == 0);
        index = index + 1;
        if ~isempty(xs)
            for q = 1:size(xs,1)
                B(xs(q,1),ys(q,1)) = 1;
            end
        end
        temp_St3 = [temp_St3 St3(:,(i-1)*numC+1:i*numC).*sqrt(A./B)];
    end
    
    temp_Gt = [];
    for i = 1:numTarget
        pos = 0;
        if i > 1
            for t = 1:i-1
                pos = pos + numTest(t);
            end
        end
        if i == 1
            pos = 0;
        end
        A = MatrixProduce(TestX(:,pos+1:pos+numTest(i))',[F1 Ft2(:,(i-1)*numK_2+1:i*numK_2) Ft3(:,(i-1)*numK_3+1:i*numK_3)]*[S1;S2;St3(:,(i-1)*numC+1:i*numC)] ,stepLen);
        B = Gt(pos+1:pos+numTest(i),:)*[S1;S2;St3(:,(i-1)*numC+1:i*numC)]'*MatrixProduce([F1 Ft2(:,(i-1)*numK_2+1:i*numK_2) Ft3(:,(i-1)*numK_3+1:i*numK_3)]', [F1 Ft2(:,(i-1)*numK_2+1:i*numK_2) Ft3(:,(i-1)*numK_3+1:i*numK_3)],stepLen)*[S1;S2;St3(:,(i-1)*numC+1:i*numC)];
        [xs ys] = find(B == 0);
        index = index + 1;
        if ~isempty(xs)
            for q = 1:size(xs,1)
                B(xs(q,1),ys(q,1)) = 1;
            end
        end
        temp_Gt = [temp_Gt; Gt(pos+1:pos+numTest(i),:).*sqrt(A./B)];
    end
    
    %%%% Normalization
    for i = 1:numK_1
        temp_F1(:,i) = temp_F1(:,i)/sum(temp_F1(:,i));
    end
    
    for i = 1:numK_2*numSource
        temp_Fs2(:,i) = temp_Fs2(:,i)/sum(temp_Fs2(:,i));
    end
    
    for i = 1:numK_2*numTarget
        temp_Ft2(:,i) = temp_Ft2(:,i)/sum(temp_Ft2(:,i));
    end
    
    for i = 1:numK_3*numSource
        temp_Fs3(:,i) = temp_Fs3(:,i)/sum(temp_Fs3(:,i));
    end
    
    for i = 1:numK_3*numTarget
        temp_Ft3(:,i) = temp_Ft3(:,i)/sum(temp_Ft3(:,i));
    end
    
    for i = 1:size(temp_Gt,1)
        temp_Gt(i,:) = temp_Gt(i,:)/sum(temp_Gt(i,:));
    end
    
    %%%% update the value
    F1 = temp_F1;
    Fs2 = temp_Fs2;
    Ft2 = temp_Ft2;
    Fs3 = temp_Fs3;
    Ft3 = temp_Ft3;
    S1 = temp_S1;
    S2 = temp_S2;
    Ss3 = temp_Ss3;
    St3 = temp_St3;
    Gt = temp_Gt;
    
    %%% Caculate the accuracy
    for i = 1:numTarget
        pos = 0;
        if i > 1
            for t = 1:i-1
                pos = pos + numTest(t);
            end
        end
        if i == 1
            pos = 0;
        end
        pzd = Gt(pos+1:pos+numTest(i),:);
        nCorrect = 0;
        for j = 1:size(pzd,1)
            [va vi] = max(pzd(j,:));
            if labelset(vi) == TestY(pos+j)
                nCorrect = nCorrect + 1;
            end
        end
        iter_results(iterID+1,i+1) = nCorrect/(numTest(i));
        iter_results(iterID+1,1) = iterID;
    end
    %iter_results(iterID+1,:)
end
% csvwrite(strcat('F1.pwz'),F1);
% csvwrite(strcat('Fs2.pwz'),Fs2);
% csvwrite(strcat('Ft2.pwz'),Ft2);
% csvwrite(strcat('Fs3.pwz'),Fs3);
% csvwrite(strcat('Ft3.pwz'),Ft3);
% csvwrite(strcat('S1.pwz'),S1);
% csvwrite(strcat('S2.pwz'),S2);
% csvwrite(strcat('Ss3.pwz'),Ss3);
% csvwrite(strcat('St3.pwz'),St3);

% csvwrite(strcat('FS.pwz'),[F1,Fs2,Fs3]);
% csvwrite(strcat('FT.pwz'),[F1,Ft2,Ft3]);
% csvwrite(strcat('Ss.pwz'),[S1;S2;Ss3]);
% csvwrite(strcat('St.pwz'),[S1;S2;St3]);
% S = [S1;S2;St3];
% x = 0:1:size(S,1)-1;
% y1 = S(:,1);
% y2 = S(:,2);
% figure
% plot(x,y1,'r',x,y2,'b');
% grid on
% xlabel('x');
% ylabel('y1 & y2');
%% output
Results = iter_results;
