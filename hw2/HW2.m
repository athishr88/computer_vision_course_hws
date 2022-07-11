% panorama creator
clear all;close all;clc;

% Display menu to chose images or exit the program
choice = menu('Choose images', 'Exit', 'Load Images');

% Implementing code for each choice
while choice~=1
    switch choice
        % Error message if no option is selected
        case 0
            disp('Error - please choose one of the options.')
            choice = menu('Choose images', 'Exit', 'Load Images');
            
        % If Load images is selected, prompt to select the images of their
        % choice
        case 2
            mosaic_choices = menu('Choose the scene', 'Library', 'Neighborhood', 'Parking Lot');
            switch mosaic_choices
                % Read images and store them in variables
                case 1
                    img1 = imread('Square0.jpg');
                    img2 = imread('Square1.jpg');
                case 2
                    img1 = imread('Neig0.jpg');
                    img2 = imread('Neig1.jpg');
                case 3
                    img1 = imread('Apart0.jpg');
                    img2 = imread('Apart1.jpg'); 
            end
        case 3
            %Prompt to select points of the user choices that could have
            %common features 
            getPoints(img1, img2);
            
            % Load the .mat file and get the selected points cordinates
            load('points.mat');
            points;
            
            %Computing the H value and selecting the best H based on error 
            least_error = 1e+1000;
            for m=1:20
                H = computeH(points);
                error = calcError(H);
                if error < least_error
                    least_error = error;
                    best_H = H;
                end
            end

            least_error
            
            % Save the best H matrix in a .mat file for referencing in
            % other functions
            save 'best_H.mat', best_H;
            close all;
        case 4
            
            %Prompt to warp images to generate the panorama
            warp1(img1, img2, best_H);
        case 5
            % Prompt to crop the selected image onto a mobile phone screen
            grad(img1);
    end
    choice = menu('Choose an option', 'Exit', 'Load other Images', 'Choose feature points', 'Image warp', 'Task 4 grad');
    
end
clear all;

% img1 = imread('Square0.jpg');
% img2 = imread('Square1.jpg');


