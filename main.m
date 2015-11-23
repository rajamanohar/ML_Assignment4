fileID = fopen('heart.txt');
C = textscan(fileID,'%d %d %d %d %d %d %d %d %d %f %d %d %d %n');
Dataset=double.empty;
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


% SMO performance using Polynomial kernal of degree 2
  [weight,bias, truePositive,falsePositive]=SeqMinOpt(test,...
      testclasses,train,trainclasses,@(x1,x2) (x1*x2 + 1)^2,0.1);
  
fprintf('\n***** SMO using polynomial Kernal function of degree 2*****\n');

fprintf('\n Sensitivity or true positive rate : %f\n',truePositive);
fprintf('\n False Positive Rate: %f\n',falsePositive);


% SMO performance using Gaussian radial basis function kernel
    [weight,bias,truePositive,falsePositive]=SeqMinOpt(test,...
      testclasses,train,trainclasses,@(x1,x2)  exp(-(2^-14*(abs(x1-x2)^2) )),0.1);
  
fprintf('\n***** SMO using Gaussian radial basis function kernel*****\n');
fprintf('\n Sensitivity or true positive rate : %f\n',truePositive);
fprintf('\n False Positive Rate: %f\n',falsePositive);

  