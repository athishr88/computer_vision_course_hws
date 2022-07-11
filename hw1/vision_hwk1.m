% This script creates a menu driven application

%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% your information
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;close all;clc;

% Display a menu and get a choice
choice = menu('Choose an option', 'Exit Program', 'Load Image');  % as you develop functions, add buttons for them here
 
% Choice 1 is to exit the program
while choice ~= 1
   switch choice
       case 0
           disp('Error - please choose one of the options.')
           % Display a menu and get a choice
           choice = menu('Choose an option', 'Exit Program', 'Load Image');  % as you develop functions, add buttons for them here
        case 2
           % Load an image
           image_choice = menu('Choose an image', 'lena1', 'mandril1', 'sully', 'yoda', 'shrek', 'wrench1','redBalloon', 'spiral');
           switch image_choice
               case 1
                   filename = 'lena1.jpg';
               case 2
                   filename = 'mandrill1.jpg';
               case 3
                   filename = 'sully.bmp';
               case 4
                   filename = 'yoda_small.bmp';
               case 5
                   filename = 'shrek.bmp';
               case 6
                   filename = 'wrench1.jpg';
               case 7
                   filename = 'redBaloon.jpg';
               case 8
                   filename = 'spiral.jpg';
             
           end
           current_img = imread(filename);
       case 3
           % Display image
           figure
           imagesc(current_img);
           if size(current_img,3) == 1
               colormap gray
           end
           
       case 4
           % Mean Filter
           
           prompt = 'Enter the kernel size(odd numbers only): '
           k_size = input(prompt);
           
           % 2. Call the appropriate function
           newImage = meanFilter(current_img, k_size); % create your own function for the mean filter
           
           % 3. Display the old and the new image using subplot
           subplot(1,2,1), imshow(current_img);
           subplot(1,2,2), imshow(newImage);
           
           
           
           % 4. Save the newImage to a file
           imwrite(newImage, 'Results\meanFilter.jpg');
           
              
       case 5
           % Adjusting brightness with loops
           prompt = 'Enter the brightness value (-255 to 255): '
           brightness = input(prompt);
           newImage = makeBright_L(current_img, brightness);
           subplot(1,2,1), imshow(current_img);
           subplot(1,2,2), imshow(newImage);
           imwrite(newImage, 'Results\makeBright_L.jpg');
           
       case 6
           % Adjsuting brightness without loops
           prompt = 'Enter the brightness value (-255 to 255):'
           brightness = input(prompt);
           newImage = makeBright_NL(current_img, brightness);
           subplot(1,2,1), imshow(current_img);
           subplot(1,2,2), imshow(newImage);
           imwrite(newImage, 'Results\makeBright_NL.jpg');
       case 7
           % inverting the image with loops
           newImage = invert_L(current_img);
           imwrite(newImage, 'Results\invert_L.jpg');
           subplot(1,2,1), imshow(current_img);
           subplot(1,2,2), imshow(newImage);
       case 8
           % inverting the image without using loops
           newImage = invert_NL(current_img);
           imwrite(newImage, 'Results\invert_NL.jpg');
           subplot(1,2,1), imshow(current_img);
           subplot(1,2,2), imshow(newImage);
       case 9
           % Adding random noise
           newImage = addRandomNoise_L(current_img);
           imwrite(newImage, 'Results\addRandomNoise.jpg');
           subplot(1,2,1), imshow(current_img);
           subplot(1,2,2), imshow(newImage);
       case 10
           % Converting coloured images to greyscale
           newImage = luminance_L(current_img);
           imwrite(newImage, 'Results\luminance_L.jpg');
           subplot(1,2,1), imshow(current_img);
           subplot(1,2,2), imshow(newImage);
       case 11
           % Red filter
           prompt = 'Enter the redVal value(0-1): '
           redVal = input(prompt);
           newImage = redFilter(current_img, redVal);
           imwrite(newImage, 'Results\redFilter.jpg');
           subplot(1,2,1), imshow(current_img);
           subplot(1,2,2), imshow(newImage);
       case 12
           % Binary masking based on mean image value
           newImage = binaryMask(current_img);
           imwrite(newImage, 'Results\binaryMask.jpg');
           subplot(1,2,1), imshow(current_img);
           subplot(1,2,2), imshow(newImage);
       case 13
           % Frosty filter
           n=  input('ENter the positive value for n within bounds:');
           m = input('ENter the positive value for m within bounds:');
           
           
           newImage = FrostyFilter(current_img,n,m);
           imwrite(newImage, 'Results\frostyFilter.jpg');
           subplot(1,2,1), imshow(current_img);
           subplot(1,2,2), imshow(newImage);
       case 14
           % Scale images based on nearest neighbour approach
           prompt = 'Enter the scaling factor: '
           scale = input(prompt);
           newImage = scaleNearest(current_img, scale);
           imwrite(newImage, 'Results\scaleNearest.jpg');
           subplot(1,2,1), imshow(current_img);
           subplot(1,2,2), imshow(newImage);
       case 15
           % Scale images based on bilinear approximation
           scale = input('Enter the scaling factor: ');
           newImage = scaleBilinear(current_img, scale);
           imwrite(newImage, 'Results\scaleBilinear.jpg');
           subplot(1,2,1), imshow(current_img);
           subplot(1,2,2), imshow(newImage);
       case 16
           % Crop and pasting faces on random images
           meLogo = imread('me_icon.jpg');
           newImage = famousMe(current_img, meLogo);
           imwrite(newImage, 'Results\famousMe.jpg');
           subplot(1,3,1), imshow(meLogo);
           subplot(1,3,2), imshow(current_img);
           subplot(1,3,3), imshow(newImage);
       case 17
           % Swirl filter
           factor = input('Enter the swirl factor: ');
           ox = input('Enter the x coordinate: ');
           oy = input('Enter the y coordinate: ');
           newImage = swirlFilter(current_img, factor, ox, oy);
           imwrite(newImage, 'Results\swirlFilter.jpg');
           subplot(1,2,1), imshow(current_img);
           subplot(1,2,2), imshow(newImage);
       %....
   end
   % Display menu again and get user's choice
   choice = menu('Choose an option', 'Exit Program', 'Load a different Image', ...
    'Display Image', 'Mean Filter', 'makeBright_L', 'makeBright_NL', 'invert_L', 'invert_NL', 'addRandomNoise',...
    'luminance_L', 'redFilter', 'binaryMask', 'FrostFilter', 'scaleNearest', 'scale_Bilinear',...
    'famousMe','Swirl_Filter');  % as you develop functions, add buttons for them here
end
