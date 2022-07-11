function[] = grad(img)
% The function takes an image as imput and casts the image onto a pre loade
% iphone screen white background

% Prompt for selecting four corners in the phone screen where the image has
% to be casted
A = imread('white_background.jpg');
imshow(A);
title('Click four points where image needs to be casted in the order LT-RT-RB-LB');
[x_1, y_1] = ginput(4);

size_b = size(img);

%four random points from input image
p_1= [x_1(1),y_1(1),1];
p_2= [x_1(2),y_1(2),1];
p_3= [x_1(3),y_1(3),1];
p_4= [x_1(4),y_1(4),1];

input = cat(1,p_1,p_2,p_3,p_4);


%four random points from output image
q_1= [1,1,1];
q_4= [1,size_b(2),1];
q_3= [size_b(1),size_b(2),1];
q_2= [size_b(1),1,1];


output = cat(1,q_1,q_2,q_3,q_4);

H_temp = zeros(8,9); %since there are four points

temp=1;


for i =1:4

    H_temp(temp,:) = [-output(i,1),-output(i,2),-1,0,0,0,output(i,1)*input(i,1),input(i,1)*output(i,2),input(i,1)]; 
    H_temp(temp+1,:) =[0,0,0,-output(i,1),-output(i,2),-1,output(i,1)*input(i,2),input(i,2)*output(i,2),input(i,2)];
    temp=temp+2;

end

%SVD step ,chosing the last sigma
[sig,u,v] = svd(H_temp);

%H matrix
H = reshape(v(:,8),[3,3])';

input = input';
output = output';

H =  input/output;

for b=1:size(img,1)
    for l=1:size(img,2)
        
        transfer_points=H*[b;l;1];
        
        x = round(transfer_points(2)/transfer_points(3)); 
        y = round(transfer_points(1)/transfer_points(3));
 
        A(x,y,:)= img(b,l,:);
       
    end
end

imshow(A);








