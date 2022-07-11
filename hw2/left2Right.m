function [outMatrix] = left2Right(inMatrix,H)

% The function will return the value of a coordinate in the left image
% after it transforms into the right image through homography
% transformation

% Homography transform
transformI = H * inMatrix;

% Normalizing
for i=1:3
    transformI(i, :) = [transformI(i, :)./transformI(3, :)];
end

% Extracting the first two rows (The coordinates)
outMatrix = transformI(1:2, :);
end

