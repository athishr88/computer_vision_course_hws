function  [outimg]  = meanFilter(inimg,n)

%taking the size of input
size_imag = size(inimg);

%making output same size as input
outimg = zeros(size(inimg),'uint8');

%iterating through each pixel and averaging the neighbours pixels to it
for i = 1:size(inimg,1)
    
    
    for j = 1:size(inimg,2)
        
        if (i+n-1) > size(inimg,1)
            
            a = size(inimg,1) ;
         
        else
            
            a = i+n-1 ;
          
        end
            
        
        if (j+n-1) > size(inimg,2)
            
            b =size(inimg,2) ;
         
        else
            
            b = j+n-1 ;
          
        end 
        
        a_1=inimg(i:a,j:b,1);
        b_1=inimg(i:a,j:b,2);
        c_1=inimg(i:a,j:b,3);
        
        outimg(i,j,1) = mean(a_1(:));
        outimg(i,j,2) = mean(b_1(:));
        outimg(i,j,3) = mean(c_1(:));
        
        
    end
    
    
    
end





end