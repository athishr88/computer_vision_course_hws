function [error] = calcError(H)
% The function will calculate and return the error between left and right
% image reference points with the homography input H

    % get the points from .mat file
    load('points.mat');
    points;
    
    % the cordinates of the points in the left image will be extracted and
    % a last row of ones will be concatenated to the matrix and then the
    % matrix will be transposed for homography calculation
    cord1hom = cat(2, points(:, 1:2), ones(10,1));
    cord1homT = cord1hom';

    % Homography transformation of all selected points of the left image
    transformH = H*cord1homT;
    
    % normalizing the matrix
    for i=1:3
        transformH(i, :) = [transformH(i, :)./transformH(3, :)];
    end
    
    % Extracting the coordinates from the transformed matrix
    transform = transformH(1:2, :);
    
    % Extracting the coordinates of the points in the right image for error
    % calculation
    cord2 = points(:, 3:4)';
    
    % the normal distance between the transformed points and the original
    % points are calculated and cumulatively added for all 10 points to
    % get the total error 
    error = 0;
    for k=1:10
        error = error + norm(transform(:, k)'- cord2(:, k)');
    end
end

