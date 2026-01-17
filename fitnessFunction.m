function fit = fitnessFunction(X, Y, mask)
% Fast NN-based fitness function

if sum(mask) < 5
    fit = 0;
    return;
end

Xsel = X(:, mask);

net = patternnet(10); % small NN for speed
net.trainParam.showWindow = false;
net.trainParam.epochs = 100;
net.divideParam.trainRatio = 1;
net.divideParam.valRatio = 0;
net.divideParam.testRatio = 0;

Y_dummy = dummyvar(Y)';
net = train(net, Xsel', Y_dummy);

YPred = net(Xsel');
[~, pred] = max(YPred, [], 1);
pred = pred';

fit = mean(pred == Y);
end
