%Converts an array of integers to an 8 bit binary array
function encodedValue = binary_encode(decArray, inputLayerSize)
    m = size(decArray, 2);
    encodedValue = zeros(m, inputLayerSize);
    for x=1:m
        encodedValue(x,:) = bitget(decArray(x), 1:inputLayerSize);
    end
end
