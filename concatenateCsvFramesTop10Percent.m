%20:53:32
clear
FRAMES = 25; %每段包含帧数
ENERGY_TOP = 0.1; %能量最高的前10%段
FRAME_STEP = 1; %段与段之间相差几帧
wrokpath = 'E:\iemocap_four_emotion\all\data\';
cd(wrokpath);                        % dos命令cd重置当前路径，自行设置，其下包含全部待处理文件
filelist = dir('*.csv');   % dos命令dir列出所有的文件，用struct2cell转换为元胞数组
filelist = struct2cell(filelist);
filelist = filelist(1,:)';
segCount = zeros(length(filelist),1);
parfor i=1:length(filelist)
    [~,filename,~] = fileparts(cell2mat(filelist(i,1)));
    csvData = load([wrokpath,filename,'.csv']);
    energyData = load([wrokpath,filename,'.energy']);
    energySum = zeros(size(energyData,1)-FRAMES+1,2);
    disp([datestr(now,13),' ',num2str(i),' ',filename]);
    %% 整理能量
    for frameIdx=1:FRAME_STEP:size(energySum,1)-FRAMES+1
        energySum(frameIdx,2)=sum(energyData(frameIdx:frameIdx+FRAMES-1,1));        
    end
    energySum(:,1) = 1:size(energySum,1); %添加索引列
    energySum = sortrows(energySum,-2); %按能量倒序排列
    frameCnt = round(size(energySum,1)*ENERGY_TOP);
    frameIdxFlag = energySum(1:frameCnt,1); %保存能量高的段序号（段序号=段中第一帧的帧序号）
    frameIdxFlag = sort(frameIdxFlag);
    %% 整理特征
    newCsv = zeros(size(frameIdxFlag,1),size(csvData,2)*FRAMES);
    for idx = 1:size(frameIdxFlag)
       frameIdx = frameIdxFlag(idx);
       newCsv(idx,:) = reshape(csvData(frameIdx:frameIdx+FRAMES-1,:)',1,[]);
    end
    segCount(i,1)=frameCnt;
    %%保存文件
    %save([wrokpath,filename,'.feature'],'newCsv','-ascii');
    csvwrite([wrokpath,filename,'.feature'],newCsv);  %比save慢,但可以并行
end
csvwrite([wrokpath,'segmentCount.txt'],segCount);