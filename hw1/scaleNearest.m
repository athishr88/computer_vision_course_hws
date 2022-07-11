function [outImg] = scaleNearest(inImg,scale)
%scaleNearest function used nearest neighbour method to scale up or down an
%image


%   Initializing the resulting image with a resolution of scale factor times the
%   resolution of the original image
new_rows = round(scale * size(inImg, 1));
new_cols = round(scale * size(inImg, 2));
depth = size(inImg, 3);

outImg = zeros(new_rows, new_cols, size(inImg, 3), 'uint8');

for i = 1: new_rows
    for j = 1: new_cols
        for k = 1: depth
            % round function determines the nearest integer from a
            % fractional number round function used to fing the x and y
            % cordinate of pixel to be sampled from
            sample_row = round(i/scale);
            sample_col = round(j/scale);
            
            % check to find all pixels are in the resolution limits
            if sample_row < 1
                sample_row = 1;
            elseif sample_row > size(inImg, 1)
                sample_row = size(inImg, 1);    
            end 
            
            if sample_col < 1
                sample_col = 1;
            elseif sample_col > size(inImg, 2)
                sample_col = size(inImg, 2);
            end
            
            % allocating pixel values
            outImg(i,j, k) = inImg(sample_row, sample_col, k);
        end
    end
end

end

