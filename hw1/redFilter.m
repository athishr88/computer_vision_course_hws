function outimg = redFilter(inimg,n)

    size_img = size(inimg);

    outimg = zeros(size(inimg),'uint8');
    
    %middle portion is same as original
    outimg(:,floor((size_img(2)/3)):floor(2*(size_img(2)/3)),:) = inimg(:,floor((size_img(2)/3)):floor(2*(size_img(2)/3)),:);
    
    
  
    
    %gray scale version of the original image
    
     outimg_1 = 0.299*inimg( : ,1:floor((size_img(2)/3)) , 1);
     outimg_2 = 0.587*inimg( :,1:floor((size_img(2)/3)) , 2);
     outimg_3 = 0.114*inimg( :,1:floor((size_img(2)/3)) , 3);
     
     
     gray_scale_version = outimg_1+outimg_2+outimg_3;
     
     for j =1:3 
         
         outimg(:,1:floor((size_img(2)/3)),j) = gray_scale_version;
         
     end
     
     %red filter
     
     outimg_1 = n*inimg( : ,floor(2*(size_img(2)/3)):(size_img(2)) , 1);
     outimg_2 = ((1-n)/2)*inimg( :,floor(2*(size_img(2)/3)):(size_img(2)) , 2);
     outimg_3 =((1-n)/2) *inimg( :,floor(2*(size_img(2)/3)):(size_img(2)) , 3);
     
     
     red_scale_version = outimg_1+outimg_2+outimg_3;
     
     for j =1:3 
         
         outimg(:,floor(2*(size_img(2)/3)):(size_img(2)),j) = red_scale_version;
         
     end
     
        
     subplot(1,2,1), imshow(inimg);
     subplot(1,2,2), imshow(outimg); 
     
     
     
end