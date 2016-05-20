
% This is an example to show how to use our software
% If you come across the problem 'out of memory ...', refer to the readme
% file again.
Train_Data = 'mydata/Train_Data.txt';
Test_Data = 'mydata/Test_Data.txt';
Parameter_Setting = 'mydata/Parameter_Setting.txt';
str = 'C:/mydata_cut/';
FileList=dir(str);
ff = 1;
for rr=1:length(FileList)
    if(FileList(rr).isdir==1&&~strcmp(FileList(rr).name,'.')&&~strcmp(FileList(rr).name,'..'))
        filedors{ff} = strcat(str,FileList(rr).name);
        ff= ff+1;
    end
end
xlswrite(strcat('dirs.xls'),filedors');
for rr=1:length(filedors)
    base = filedors{rr};
    trainPath = strcat(base,'/Train.data');
    trainLabelPath =strcat(base,'/Train.label');
    testPath = strcat(base,'/Test.data');
    testLabelPath = strcat(base,'/Test.label');
    fprintf('%s\n',base);
    fid=fopen(Train_Data,'w');
    fprintf(fid,'%d\r\n',1);
    fprintf(fid,'%s\r\n',trainPath);
    fprintf(fid,'%s\r\n',trainLabelPath);
    fclose(fid);
    fid=fopen(Test_Data,'w');
    fprintf(fid,'%d\r\n',1);
    fprintf(fid,'%s\r\n',testPath);
    fprintf(fid,'%s\r\n',testLabelPath);
    fclose(fid);
    [Results, Gt, t1, t2] = TriTL(Train_Data,Test_Data,Parameter_Setting)
    [res] = xlsread(strcat('Results.xls'));
    xlswrite(strcat('Results.xls'),[res;Results(:,2)']);
end



%  Train_Data = 'Train_Data.txt';
%  Test_Data = 'Test_Data.txt';
%  Parameter_Setting = 'Parameter_Setting.txt';


