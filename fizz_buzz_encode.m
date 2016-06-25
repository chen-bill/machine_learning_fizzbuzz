%One-hot endcoding for 4 unit output
function result = fizz_buzz_encode(val)
    numRows = size(val,2);
    result = zeros(numRows, 4);
    for i=1:numRows
        if (mod(val(i), 15) == 0)
            result(i,:)= [0, 0, 0, 1];
        elseif (mod(val(i), 5) == 0)
            result(i,:) = [0, 0, 1, 0];
        elseif (mod(val(i),3) == 0)
            result(i,:) = [0, 1, 0, 0];
        else 
            result(i,:) = [1, 0, 0, 0];
        end
    end
end

