fileID = fopen('heart.txt');
C = textscan(fileID,'%d %d %d %d %d %d %d %d %d %f %d %d %d %n');

for idx=1:length(C)
    Dataset(:,idx)=C{idx};
end
[TrainData,TestData]=DatasetPartition(Dataset);

train=TrainData(:,1:13);
trainclasses=TrainData(:,14);

trainclasses(trainclasses==2 )=-1; 

test=TestData(:,1:13);
testclasses=TestData(:,14);
testclasses(testclasses==2 )=-1; 

  [weight,bias, truePositive, falsePositive]=SeqMinOpt(double(test),double(testclasses),double(train),double(trainclasses), @(x1,x2) x1*x2,0.1) 
  
  