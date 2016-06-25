function p = predict_single(Theta1, Theta2, X)
m = size(X, 1);

a1 = sigmoid([ones(m, 1) X] * Theta1');
p = sigmoid([ones(m, 1) a1] * Theta2');
end
