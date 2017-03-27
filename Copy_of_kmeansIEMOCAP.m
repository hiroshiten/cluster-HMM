clear
addpath(genpath('D:/代码/HMMall'));
%% load
load 'E:\iemocap_four_emotion\all\data\s1.feature';
load 'E:\iemocap_four_emotion\all\data\s2.feature';
load 'E:\iemocap_four_emotion\all\data\s3.feature';
load 'E:\iemocap_four_emotion\all\data\s4.feature';
load 'E:\iemocap_four_emotion\all\data\s5.feature';

load 'E:\iemocap_four_emotion\all\data\s1segCount_Target.txt';
load 'E:\iemocap_four_emotion\all\data\s2segCount_Target.txt';
load 'E:\iemocap_four_emotion\all\data\s3segCount_Target.txt';
load 'E:\iemocap_four_emotion\all\data\s4segCount_Target.txt';
load 'E:\iemocap_four_emotion\all\data\s5segCount_Target.txt';

s1z = zscore(s1);
s2z = zscore(s2);
s3z = zscore(s3);
s4z = zscore(s4);
s5z = zscore(s5);
targetNum = 4;
%% cluster using k-means

k=10;
[Idx,centroid,~,D]=kmeans([s2z;s3z;s4z;s5z],k,'MaxIter',500);%迭代次数默认是100
% score = 1./D;
% for i=1:size(D,1)
%    score(i,:) =  score(i,:)/sum(score(i,:)); 
% end

% for i=1:k
%    size(find(Idx==i),1)
% end

[Idx2,~,~,D2]=kmeans(s1z,k,'Start',centroid,'MaxIter',500);


% score2 = 1./D2;
% for i=1:size(D2,1)
%    score2(i,:) =  score2(i,:)/sum(score2(i,:)); 
% end
% 
% trainSampleNum = size(trainSegmentCount,1);%样本（句子）数
% beginNo = 1;
% a=[];
% save trainKmeansScore.txt a -ascii
% for i = 1:trainSampleNum
%     endNo = beginNo + trainSegmentCount(i)-1;
%     temp = score(beginNo:endNo,:);
%     temp = reshape(temp',[],1)';
%     save trainKmeansScore.txt temp -ascii -append
%     beginNo = endNo + 1;
% end
% 
% testSampleNum = size(testSegmentCount,1);%样本（句子）数
% beginNo = 1;
% a=[];
% save testKmeansScore.txt a -ascii
% for i = 1:testSampleNum
%     endNo = beginNo + testSegmentCount(i)-1;
%     temp = score2(beginNo:endNo,:);
%     temp = reshape(temp',[],1)';
%     save testKmeansScore.txt temp -ascii -append
%     beginNo = endNo + 1;
% end

%% save the cluster Index
trainSampleNum = size(trainSegmentCount,1);%样本（句子）数
beginNo = 1;
a=[];
%save trainKmeansCluster30.txt a -ascii
for i = 1:trainSampleNum
    endNo = beginNo + trainSegmentCount(i)-1;
    temp = Idx(beginNo:endNo,:)';
    %save trainKmeansCluster30.txt temp -ascii -append
    beginNo = endNo + 1;
end

testSampleNum = size(testSegmentCount,1);%样本（句子）数
beginNo = 1;
a=[];
%save testKmeansCluster30.txt a -ascii
for i = 1:testSampleNum
    endNo = beginNo + testSegmentCount(i)-1;
    temp = Idx2(beginNo:endNo,:)';
    %save testKmeansCluster30.txt temp -ascii -append
    beginNo = endNo + 1;
end

%% convert the cluster result into cells for HMM
beginNo = 1;
trainClusterCell = {};
for i = 1:size(trainSegmentCount,1)
    endNo = beginNo + trainSegmentCount(i)-1;
    trainClusterCell = [trainClusterCell;Idx(beginNo:endNo,:)'];
    beginNo = endNo + 1;
end

beginNo = 1;
testClusterCell = {};
for i = 1:size(testSegmentCount,1)
    endNo = beginNo + testSegmentCount(i)-1;
    testClusterCell = [testClusterCell;Idx2(beginNo:endNo,:)'];
    beginNo = endNo + 1;
end


for i=1:5
    testCell{i}=testClusterCell(find(testUttrTarget==i));
    trainCell{i}=trainClusterCell(find(trainUttrTarget==i));
end
%save 20clusterDataSet.mat testCell trainCell


%% HMM parameters

O = k;%维度
Q = k;
prior = normalise(rand(Q,1));
transmat = mk_stochastic(rand(Q,Q));
obsmat = mk_stochastic(rand(Q,O));
HMMpara = cell(targetNum,3);

%% train 5 HMMs
for i=1:targetNum
    disp(['HMM training: ' num2str(i) '/' num2str(targetNum)]);
    [~, HMMpara{i,1}, HMMpara{i,2}, HMMpara{i,3}] = dhmm_em(trainCell{1,i}, prior, transmat, obsmat, 'max_iter', 100);  
    % verbose = 0 不输出迭代结果
end

%% test
LLSet = [];
for i=1:targetNum %i is real target
    LL=[];
    for j=1:targetNum % j is index of HMMs
        for l = 1:size(testCell{1,i},1) %l is index of test samples
            LL(l,j) = dhmm_logprob(testCell{1,i}(l), HMMpara{j,1}, HMMpara{j,2}, HMMpara{j,3});
        end
        
    end
    LLSet = [LLSet;ones(size(LL,1),1)*i,LL];
end

%% result
[~,predicted] = max(LLSet(:,2:end),[],2); 
real = LLSet(:,1);
SampleNum = size(testClusterCell,1);
confMat=zeros(targetNum,targetNum); 
for i=1:SampleNum
    confMat(real(i),predicted(i))=confMat(real(i),predicted(i))+1;
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