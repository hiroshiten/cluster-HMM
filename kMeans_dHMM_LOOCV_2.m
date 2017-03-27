%����һ�����ࡢѵ��������

clear
addpath(genpath('D:/����/HMMall'));
%% load data

%rawData = load('D:\dynamicNoLastSeg\emodb_dynamicNoLastSeg_88.csv');
rawData = load('D:\dynamicNoLastSeg\rawData.csv');
segmentCount = load('D:\dynamicNoLastSeg\segmentCount.txt');

% rawData = load('D:\fixedLength\rawData.csv');
% segmentCount = load('D:\fixedLength\segmentCount.txt');

[~,uttrTarget] = max(load('D:\tmp\target.txt'),[],2);
rawData = zscore(double(rawData));%��һ��
targetNum = 7;
SampleNum = size(segmentCount,1);%���������ӣ���

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
O = k;%ά��
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
    %Ԥ����ռ�
    LL = zeros(1,targetNum);
    
    disp(['Cross-validation: ' num2str(fold) '/' num2str(numfold)]);
    trainingSampleID=SampleID(CVP.training(fold));
    testSampleID=SampleID(CVP.test(fold));    
    trainSamples = cell2mat(featureCell(trainingSampleID));    
    testSamples = cell2mat(featureCell(testSampleID));
    % ����
    [clusterIdx,centroid,~,~]=kmeans(trainSamples,k); 
    tmpIdx = kmeans([testSamples;trainSamples],k,'Start',centroid); %����������������k�����Դ���ѵ������һ����࣬Ȼ�������������ľ�����
    tmpIdx = tmpIdx(1:size(testSamples,1));
    beginNo = sum(segmentCount(1:testSampleID-1));
    clusterIdx = [clusterIdx(1:beginNo);...
                  tmpIdx;...
                  clusterIdx(beginNo+1:end)];
    %��ÿ�εľ���������洢          
    clusterSeqCell = cell(SampleNum,1);
    beginNo = 1;
    for i = 1:SampleNum
        endNo = beginNo + segmentCount(i)-1;
        clusterSeqCell{i,1} = clusterIdx(beginNo:endNo,:)';
        beginNo = endNo + 1;
    end   
    %���伶��������������洢
    SeqClassCell = cell(1,targetNum);
    for i=1:targetNum
        SeqClassCell{1,i} = clusterSeqCell(find(uttrTarget==i));    
    end
    
    clear tmpIdx clusterIdx beginNo endNo
    
    %׼��HMMѵ���Ͳ�������   
    testSample = clusterSeqCell(testSampleID); %ȡ����������
    testSampleTarget = uttrTarget(testSampleID); %�ò������������
    testSampleIDinClass = find(find(uttrTarget==testSampleTarget)==testSampleID);%�ò��������ڴ�����е����      
    SeqClassCell{1,testSampleTarget}(testSampleIDinClass,:)=[];%�Ӹ������ɾ����������
    clear clusterSeqCell
    %ѵ��n��HMM ������
    for i=1:targetNum
        [~, HMMpara{1}, HMMpara{2}, HMMpara{3}] = dhmm_em(SeqClassCell{1,i}, prior, transmat, obsmat, 'max_iter', 100);  %ѵ��
        % verbose = 0 �����ѵ���������
        LL(1,i) = dhmm_logprob(testSample, HMMpara{1}, HMMpara{2}, HMMpara{3}); %����
    end;
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

k
confMat
accuracy
avg = count/SampleNum
