function g = relu_gradient(z)
g = zeros(size(z));
g=relu(z).*(1-relu(z));
end
