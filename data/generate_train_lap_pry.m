clear;
close all;
folder = '/home/tiger/Graduate/datasets/LapSRN/trainingset';
savepath = '/home/tiger/Graduate/datasets/LapSRN/trainingset_pre';

%% Generate paths
LR_path = [savepath, '/LR'];
x2_path = [savepath, '/x2'];
x4_path = [savepath, '/x4'];

if(isdir(LR_path) == 0)
    mkdir(LR_path);
end
if(isdir(x2_path) == 0)
    mkdir(x2_path);
end
if(isdir(x4_path) == 0)
    mkdir(x4_path);
end

size_label = 128;
scale = 4;
size_input = size_label/scale;
size_x2 = size_label/2;
stride = 64;

%% downsizing
%downsizes = [1,0.7,0.5];
downsizes = [1];

data = zeros(size_input, size_input, 1, 1);
label_x2 = zeros(size_x2, size_x2, 1, 1);
label_x4 = zeros(size_label, size_label, 1, 1);

count = 0;
margain = 0;

%% generate data
filepaths = [];
filepaths = [filepaths; dir(fullfile(folder, '*.jpg'))];
filepaths = [filepaths; dir(fullfile(folder, '*.bmp'))];
filepaths = [filepaths; dir(fullfile(folder, '*.png'))];

length(filepaths)

for i = 1 : length(filepaths)
    fprintf(num2str(i));
    fprintf('\n');
    for flip = 1:3
    %for flip = 3:3
        %for degree = 1:4
        for degree = 1:1
            for downsize = 1 : length(downsizes)

                image = imread(fullfile(folder,filepaths(i).name));
                if flip == 1
                    image = flipdim(image, 1);
                end
                if flip == 2
                    image = flipdim(image, 2);
                end
                image = imrotate(image, 90*(degree - 1));
                image = imresize(image, downsizes(downsize), 'bicubic');

                if size(image,3)==3
                    image = rgb2ycbcr(image);
                    image = im2double(image(:, :, 1));
                    im_label = modcrop(image, scale);
                    [hei,wid] = size(im_label);

                    for x = 1 + margain : stride : hei-size_label+1 - margain
                        for y = 1 + margain :stride : wid-size_label+1 - margain
                            subim_label = im_label(x : x+size_label-1, y : y+size_label-1);
                            subim_label_x2 = imresize(subim_label, 1/scale*2, 'bicubic');
                            subim_input = imresize(subim_label, 1/scale, 'bicubic');

                            count = count+1;
                            
                            lr_name = [LR_path, '/', num2str(count), '.png'];
                            x2_name = [x2_path, '/', num2str(count), '.png'];
                            x4_name = [x4_path, '/', num2str(count), '.png'];
                            imwrite(subim_input, lr_name);
                            imwrite(subim_label_x2, x2_name);
                            imwrite(subim_label, x4_name);
                        end
                    end
                end
            end
        end
    end
end
