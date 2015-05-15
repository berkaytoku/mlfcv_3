trainImages = loadMNISTImages('../data/train-images.idx3-ubyte');
trainLabels = loadMNISTLabels('../data/train-labels.idx1-ubyte');

testImages = loadMNISTImages('../data/t10k-images.idx3-ubyte');
testLabels = loadMNISTLabels('../data/t10k-labels.idx1-ubyte');

ensemble = fitensemble(transpose(trainImages(:,1:60000)),trainLabels(1:60000,:),'AdaBoostM2',1000,'Tree');
error = loss(ensemble,transpose(testImages),testLabels,'mode','cumulative');
plot(error);