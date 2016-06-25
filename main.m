INPUT_LAYER_SIZE  = 16;     % 16 bit input
HIDDEN_LAYER_SIZE = 100;    % 100 unit hidden layer just because
OUTPUT_LAYER_SIZE = 4;      % 4 possible output results
LAMBDA=1;

% Matrix x values from 100 - 10000 because we dont want to use
values = randperm(10000) + 100;
X = binary_encode(values, INPUT_LAYER_SIZE);
y = fizz_buzz_encode(values);

initial_Theta1 = randomized_init(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE);
initial_Theta2 = randomized_init(HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE);

init_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

options = optimset('MaxIter', 10000);
costFunction = @(p) cost_function(p, INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE, X, y, LAMBDA);

fprintf('\nTraining Network...\n')
[nn_params, cost] = fmincg(costFunction, init_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:HIDDEN_LAYER_SIZE * (INPUT_LAYER_SIZE + 1)), ...
                 HIDDEN_LAYER_SIZE, (INPUT_LAYER_SIZE + 1));

Theta2 = reshape(nn_params((1 + (HIDDEN_LAYER_SIZE * (INPUT_LAYER_SIZE + 1))):end), ...
                 OUTPUT_LAYER_SIZE, (HIDDEN_LAYER_SIZE + 1));

Theta1
Theta2
%fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
fprintf('\nEnd of program. \n')
