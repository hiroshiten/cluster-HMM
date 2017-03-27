%用留一法聚类、训练、测试

clear
addpath(genpath('D:/代码/HMMall'));
%% load data

%rawData = load('D:\dynamicNoLastSeg\emodb_dynamicNoLastSeg_88.csv');
rawData = load('D:\dynamicNoLastSeg\rawData.csv');
segmentCount = load('D:\dynamicNoLastSeg\segmentCount.txt');

% rawData = load('D:\fixedLength\rawData.csv');
% segmentCount = load('D:\fixedLength\segmentCount.txt');

[~,uttrTarget] = max(load('D:\tmp\target.txt'),[],2);
rawData = zscore(double(rawData));%归一化
targetNum = 7;
SampleNum = size(segmentCount,1);%样本（句子）数

beginNo = 1;
featureCell = {};
for i = 1:SampleNum
    endNo = beginNo + segmentCount(i)-1;
    featureCell = [featureCell;rawData(beginNo:endNo,:)];
    beginNo = endNo + 1;
end
clear rawData beginNo endNo i

%% parameters
k=10; % K-means
O = k;%维度
Q = k;
prior = normalise(rand(Q,1));
transmat = mk_stochastic(rand(Q,Q));
obsmat = mk_stochastic(rand(Q,O));
HMMpara = cell(1,3);

%% cross validation

LLSet=zeros(SampleNum,targetNum+1);
SampleID=1:SampleNum;
numfold = SampleNum;
CVP=cvpartition(SampleNum,'k',numfold);

for fold=1:numfold
    %预分配空间
    LL = zeros(1,targetNum);
    
    disp(['Cross-validation: ' num2str(fold) '/' num2str(numfold)]);
    trainingSampleID=SampleID(CVP.training(fold));
    testSampleID=SampleID(CVP.test(fold));    
    trainSamples = cell2mat(featureCell(trainingSampleID));    
    testSamples = cell2mat(featureCell(testSampleID));
    % 聚类
    [clusterIdx,centroid,~,~]=kmeans(trainSamples,k); 
    tmpIdx = kmeans([testSamples;trainSamples],k,'Start',centroid); %聚类样本不能少于k，所以带上训练样本一起聚类，然后保留测试样本的聚类结果
    tmpIdx = tmpIdx(1:size(testSamples,1));
    beginNo = sum(segmentCount(1:testSampleID-1));
    clusterIdx = [clusterIdx(1:beginNo);...
                  tmpIdx;...
                  clusterIdx(beginNo+1:end)];
    %将每段的聚类结果按句存储          
    clusterSeqCell = cell(SampleNum,1);
    beginNo = 1;
    for i = 1:SampleNum
        endNo = beginNo + segmentCount(i)-1;
        clusterSeqCell{i,1} = clusterIdx(beginNo:endNo,:)';
        beginNo = endNo + 1;
    end   
    %将句级聚类结果按情感类别存储
    SeqClassCell = cell(1,targetNum);
    for i=1:targetNum
        SeqClassCell{1,i} = clusterSeqCell(find(uttrTarget==i));    
    end
    
    clear tmpIdx clusterIdx beginNo endNo
    
    %准备HMM训练和测试样本   
    testSample = clusterSeqCell(testSampleID); %取出测试样本
    testSampleTarget = uttrTarget(testSampleID); %该测试样本的类别
    testSampleIDinClass = find(find(uttrTarget==testSampleTarget)==testSampleID);%该测试样本在此类别中的序号      
    SeqClassCell{1,testSampleTarget}(testSampleIDinClass,:)=[];%从该类别中删除测试样本
    clear clusterSeqCell
    %训练n个HMM 并测试
    for i=1:targetNum
        [~, HMMpara{1}, HMMpara{2}, HMMpara{3}] = dhmm_em(SeqClassCell{1,i}, prior, transmat, obsmat, 'max_iter', 100);  %训练
        % verbose = 0 不输出训练迭代结果
        LL(1,i) = dhmm_logprob(testSample, HMMpara{1}, HMMpara{2}, HMMpara{3}); %测试
    end;
    LLSet(fold,:)=[testSampleID,LL]; %似然值数组
end
LLSet=sortrows(LLSet,1);

[~,predicted] = max(LLSet(:,2:end),[],2);

confMat=zeros(targetNum,targetNum); 
for i=1:SampleNum
    confMat(uttrTarget(i),predicted(i))=confMat(uttrTarget(i),predicted(i))+1;
end

count=0;
for i=1:SampleNum
    count = count + confMat(1,1);
end
S = sum(confMat,2);
count = 0;
for i=1:targetNum
   accuracy(i) =  confMat(i,i)/S(i);
   count = count + confMat(i,i);
end

k
confMat
accuracy
avg = count/SampleNum
