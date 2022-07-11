function[points] = getPoints(img1, img2)

% Function will take two input images and return the cordinates of the
% points seledcted by the user upon the prompt provided by the function in
% [left_image_x, left_image_y, right_image_x, right_image_y] format
    
    % Display the left image and prompt to select points of common features
    % in both left and right images
    imshow(img1);
    title('Click manually on common features');
    [y1, x1] = ginput(10);
    
    % Display the right image and prompt to select points of common features
    % in both left and right images
    imshow(img2);
    title('Click manually on same features you selected in the previous image in the same order');
    [y2, x2] = ginput(10);
    
    % Save the points in a 10 x 4 matrix and save it to a .mat file
    points = [x1, y1, x2, y2];
    save 'points.mat', points;
end