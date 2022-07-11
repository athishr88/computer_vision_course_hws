function[H] = computeH(points)
% Function to calculate the homography of transformation from left image to
% the right image based on the points provided by the user

% Function will input the points provided and calculate the homography
% based on Homogenouos linear least squares method 

    % Picking 4 random numbers 1 through 10 
    r = randperm(10, 4);
    
    % Selecting a random row in the points matrix for computing H
    p1 = points(r(1), :);
    p2 = points(r(2), :);
    p3 = points(r(3), :);
    p4 = points(r(4), :);

    % Implementing homogenous least squares 
    
    % Finding A
    A(1, :) = [p1(1) p1(2) 1 0 0 0 -p1(3)*p1(1) -p1(3)*p1(2) -p1(3)];
    A(2, :) = [0 0 0 p1(1) p1(2) 1 -p1(4)*p1(1) -p1(4)*p1(2) -p1(4)];

    A(3, :) = [p2(1) p2(2) 1 0 0 0 -p2(3)*p2(1) -p2(3)*p2(2) -p2(3)];
    A(4, :) = [0 0 0 p2(1) p2(2) 1 -p2(4)*p2(1) -p2(4)*p2(2) -p2(4)];

    A(5, :) = [p3(1) p3(2) 1 0 0 0 -p3(3)*p3(1) -p3(3)*p3(2) -p3(3)];
    A(6, :) = [0 0 0 p3(1) p3(2) 1 -p3(4)*p3(1) -p3(4)*p3(2) -p3(4)];

    A(7, :) = [p4(1) p4(2) 1 0 0 0 -p4(3)*p4(1) -p4(3)*p4(2) -p4(3)];
    A(8, :) = [0 0 0 p4(1) p4(2) 1 -p4(4)*p4(1) -p4(4)*p4(2) -p4(4)];

    % Singular value decomposition
    [U S V] = svd(A);
    
    % Selecting the last column of V vector in the SVD to get the best H
    % and reshaping to a 3 X 3 matrix
    H = reshape(V(:, 9), [3, 3])';

end