function  [outimg]  = FrostyFilter(inimg,n,m)

outimg = zeros(size(inimg),'uint8');

for i = 1:size(inimg,1)
    for j = 1:size(inimg,2)
        
        if (i+n-1) > size(inimg,1)
            
            a = randi([i,size(inimg,1)],1);
         
        else
            
            a = randi([i,i+n-1],1);
          
        end
            
        
        if (j+m-1) > size(inimg,2)
            
            b = randi([j,size(inimg,2)],1);
         
        else
            
            b = randi([j,j+m-1],1);
          
        end          

        
        outimg(i,j,:) = inimg(a,b,:);
        
    end
    
end

subplot(1,2,1), imshow(inimg);
subplot(1,2,2), imshow(outimg);

end