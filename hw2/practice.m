coords1 = [1 2 3; 4 5 6; 7 8 9];

% forward mapping function 
% compute H * [x,y,z]'
% need to convert cartensian coords to homogenous.

q =  1 *[coords1; ones(1, size(coords1,2))];

p = q(3,:);

% normalize back to x,y coords
check = [q(1,:)./p; q(2,:)./p];