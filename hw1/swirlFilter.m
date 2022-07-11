function [outImg] = swirlFilter(inImg, factor, ox, oy)
% swirlFilter function adds a swirl at a location specified by the user
% with a radius of nearest distance to an edge

% distance from each edge is calculated and radius is found out  
distances = [ox size(inImg, 1)-ox-1 oy size(inImg, 2)-oy-1];
radius = min(distances)-1;
max_angle = factor*2*pi;

outImg = inImg;

% swirl function is going to go through a rectangular sample space aroud
% the swirl center and swirl is to be applied only to pixels whose normal
% distance is less than the radius calculated earlier
for i = ox-radius:ox+radius
    for j = oy-radius:oy+radius
        normal_distance = norm([i,j] - [ox,oy]); 
        if normal_distance ~= 0
            % angle of swirl is going to change with distance from the
            % center
            varying_angle = max_angle * normal_distance/radius;
        end
        if normal_distance <= radius 
            % with angle and distance from the center known, the new
            % location of the pixel is calculted retaining the pixel value
            % of the original location
            newi = (i-ox)*cos(varying_angle) - (j-oy)* sin(varying_angle) + ox;
            newj = (i-ox)*sin(varying_angle) - (j-oy)* cos(varying_angle) + oy;
            new_i = round(newi);
            new_j = round(newj);
            outImg(i, j,:) = inImg(new_i, new_j, :);
        end
    end
end

end

