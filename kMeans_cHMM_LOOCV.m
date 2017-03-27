clear
addpath(genpath('D:/����/HMMall'));
%% load data

rawData = load('D:\dynamicNoLastSeg\rawData.csv');
segmentCount = load('D:\dynamicNoLastSeg\segmentCount.txt');
segmentTarget = load('D:\dynamicNoLastSeg\target.txt');

% rawData = load('D:\fixedLength\rawData.csv');
% segmentCount = load('D:\fixedLength\segmentCount.txt');
% segmentTarget = load('D:\fixedLength\target.txt');

[~,uttrTarget] = max(load('D:\tmp\target.txt'),[],2);
rawData = zscore(double(rawData));%��һ��
targetNum = 7;
SampleNum = size(segmentCount,1);%���������ӣ���

%% k-means ��������������

k=10;  % K-means
[clusterIdx,centroid,~,D]=kmeans(rawData,k);  %D�Ǹ�������k�����ĵľ���
clusterSeqCell = cell(SampleNum,1);
beginNo = 1;
for i = 1:SampleNum
    endNo = beginNo + segmentCount(i)-1;
    clusterSeqCell{i,1} = D(beginNo:endNo,:)';
    beginNo = endNo + 1;
end
clear clusterIdx centroid D

%% GMM ��������������

%gm = gmdistribution.fit(rawData,3);

%% HMM parameters

O = k;%ά��
Q = 2;
prior = normalise(rand(Q,1));
transmat = mk_stochastic(rand(Q,Q));
obsmat = mk_stochastic(rand(Q,O));
HMMpara = cell(targetNum,3);
tmpHMMpara = cell(1,3);
SeqClassCell = cell(1,targetNum);
for i=1:targetNum
    SeqClassCell{1,i} = clusterSeqCell(find(uttrTarget==i));    
end

%% pretrain 7 HMMs
for i=1:targetNum
    [~, HMMpara{i,1}, HMMpara{i,2}, HMMpara{i,3}] = dhmm_em(SeqClassCell{1,i}, prior, transmat, obsmat, 'max_iter', 100);  
end

%% cross validation
LL = zeros(1,targetNum);
LLSet=zeros(SampleNum,targetNum+1);

SampleID=1:SampleNum;
numfold = SampleNum;
CVP=cvpartition(SampleNum,'k',numfold);
for fold=1:numfold
    disp(['Cross-validation: ' num2str(fold) '/' num2str(numfold)]);
    trainingSampleID=SampleID(CVP.training(fold));
    testSampleID=SampleID(CVP.test(fold));
    testSample = clusterSeqCell(testSampleID);
    testSampleTarget = uttrTarget(testSampleID); %�ò������������
    [testSampleIDinClass,~] = find(find(uttrTarget==i)==testSampleID);%�ò��������ڴ�����е����
    %���Ƹ���𣬲��Ӹø�����ɾ����������
    classWithoutTestSample = SeqClassCell{1,testSampleTarget};
    classWithoutTestSample(testSampleIDinClass,:)=[];
    %�ø���ѵ��һ��HMM
    [~, tmpHMMpara{1,1}, tmpHMMpara{1,2}, tmpHMMpara{1,3}] = dhmm_em(classWithoutTestSample, prior, transmat, obsmat, 'max_iter', 100);
    %����
    for i=1:targetNum
        if i == testSampleTarget
            LL(1,i) = dhmm_logprob(testSample, tmpHMMpara{1,1}, tmpHMMpara{1,2},tmpHMMpara{1,3});
        else        
            LL(1,i) = dhmm_logprob(testSample, HMMpara{i,1}, HMMpara{i,2},HMMpara{i,3});
        end
    end
    LLSet(fold,:)=[testSampleID,LL]; %��Ȼֵ����
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

confMat
accuracy
avg = count/SampleNum
