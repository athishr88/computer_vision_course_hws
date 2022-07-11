function [outMatrix] = right2Left(inMatrix,invH)

% The function will return the value of a coordinate in the right image
% after it transforms into the left image through inverse homography
% transformation

% Homography transform
transformI = invH * inMatrix;

% Normalizing
for i=1:3
    transformI(i, :) = [transformI(i, :)./transformI(3, :)];
end

% Extracting the first two rows (The coordinates)
outMatrix = transformI(1:2, :);
end

