clear all, close all
write_fig = 0;
n1 = 200; n2 = 200;
S1 = eye(2); S2 = [1 0.95; 0.95 1];
m1 = [0.75; 0]; m2 = [-0.75; 0];

x1 = bsxfun(@plus, chol(S1)'*gpml_randn(0.2, 2, n1), m1);         
x2 = bsxfun(@plus, chol(S2)'*gpml_randn(0.3, 2, n2), m2);         
x = [x1 x2]'; y = [-ones(1,n1) ones(1,n2)]';

t1 = bsxfun(@plus, chol(S1)'*gpml_randn(0.2, 2, n1), m1);        
t2 = bsxfun(@plus, chol(S2)'*gpml_randn(0.3, 2, n2), m2);
tx = [t1 t2]'; ty = [-ones(1,n1) ones(1,n2)]';
n = length(tx);

meanfunc = @meanConst; hyp.mean = 0;
covfunc = @covSEard;   hyp.cov = log([1 1 1]);
likfunc = @likLogistic;
inffunc = @infEP;

hyp = minimize(hyp, @gp, -40, inffunc, meanfunc, covfunc, likfunc, x, y);
[a b c d lp] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, tx, ones(n,1));

% Classifier Evaluation
aclassified = (a > 0) - (a < 0);
classificationEval = sum((ty - aclassified).^2);
correctclassification = (aclassified == ty);
classificationRate = sum(correctclassification) / length(correctclassification)

% GP Uncertainty Estimation
p = exp(lp);
correctC = (aclassified == ty) .* p;
incorrectC = (aclassified ~= ty) .* p;
correctEntropy = -(correctC.*log(correctC) + (1 - correctC).*log(1-correctC));
incorrectEntropy = -(incorrectC.*log(incorrectC) + (1 - incorrectC).*log(1-incorrectC));

histogram(correctEntropy,10);
histogram(incorrectEntropy,10);