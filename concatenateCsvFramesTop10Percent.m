%20:53:32
clear
FRAMES = 25; %ÿ�ΰ���֡��
ENERGY_TOP = 0.1; %������ߵ�ǰ10%��
FRAME_STEP = 1; %�����֮����֡
wrokpath = 'E:\iemocap_four_emotion\all\data\';
cd(wrokpath);                        % dos����cd���õ�ǰ·�����������ã����°���ȫ���������ļ�
filelist = dir('*.csv');   % dos����dir�г����е��ļ�����struct2cellת��ΪԪ������
filelist = struct2cell(filelist);
filelist = filelist(1,:)';
segCount = zeros(length(filelist),1);
parfor i=1:length(filelist)
    [~,filename,~] = fileparts(cell2mat(filelist(i,1)));
    csvData = load([wrokpath,filename,'.csv']);
    energyData = load([wrokpath,filename,'.energy']);
    energySum = zeros(size(energyData,1)-FRAMES+1,2);
    disp([datestr(now,13),' ',num2str(i),' ',filename]);
    %% ��������
    for frameIdx=1:FRAME_STEP:size(energySum,1)-FRAMES+1
        energySum(frameIdx,2)=sum(energyData(frameIdx:frameIdx+FRAMES-1,1));        
    end
    energySum(:,1) = 1:size(energySum,1); %���������
    energySum = sortrows(energySum,-2); %��������������
    frameCnt = round(size(energySum,1)*ENERGY_TOP);
    frameIdxFlag = energySum(1:frameCnt,1); %���������ߵĶ���ţ������=���е�һ֡��֡��ţ�
    frameIdxFlag = sort(frameIdxFlag);
    %% ��������
    newCsv = zeros(size(frameIdxFlag,1),size(csvData,2)*FRAMES);
    for idx = 1:size(frameIdxFlag)
       frameIdx = frameIdxFlag(idx);
       newCsv(idx,:) = reshape(csvData(frameIdx:frameIdx+FRAMES-1,:)',1,[]);
    end
    segCount(i,1)=frameCnt;
    %%�����ļ�
    %save([wrokpath,filename,'.feature'],'newCsv','-ascii');
    csvwrite([wrokpath,filename,'.feature'],newCsv);  %��save��,�����Բ���
end
csvwrite([wrokpath,'segmentCount.txt'],segCount);