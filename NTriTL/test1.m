
% This is an example to show how to use our software
% If you come across the problem 'out of memory ...', refer to the readme
% file again.

Train_Data = 'mydata/Train_Data.txt';
Test_Data = 'mydata/Test_Data.txt';
Parameter_Setting = 'mydata/Parameter_Setting.txt';
%  Train_Data = 'Train_Data.txt';
%  Test_Data = 'Test_Data.txt';
%  Parameter_Setting = 'Parameter_Setting.txt';

[Results, Gt, t1, t2] = TriTL(Train_Data,Test_Data,Parameter_Setting)
% csvwrite(strcat('mydata/testpro.dat'),Gt);
xlswrite(strcat('Results.xls'),Results(:,2)');
