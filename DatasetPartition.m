function [TrainData,TestData]=DatasetPartition(Dataset)
[rows,columns]=size(Dataset);
% randomly splitting the dataset into train and test data 
randIdx=randperm(rows);
trainIdx=randIdx(1,1:180);
testIdx=randIdx(1,181:270);

TrainData=Dataset(trainIdx,:);
TestData=Dataset(testIdx,:);
end
