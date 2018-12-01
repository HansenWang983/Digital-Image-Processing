clear all;
% 读取图片到矩阵，矩阵元素为灰度值，下标为图片的几何坐标，注意从下标1开始，而坐标从0开始
f = imread('../river.JPG');
[height,width] = size(f)
% 查看图片信息
% whos f
% imfinfo ../river.JPG

% 直方图均衡化
g = my_histeq(f,256);

% histeq得到的图像
h = histeq(f,256);

%显示原图像、均衡化后，以及histeq返回的图像
%显示原图像、均衡化后，以及histeq返回的直方图
figure;
subplot(2,3,1),imshow(f);title('origin image');
subplot(2,3,2),imshow(g);title('hist equal image');
subplot(2,3,3),imshow(h);title('histeq test image');
subplot(2,3,4),imhist(f);title('origin hist');
subplot(2,3,5),imhist(g);title('hist equal hist');
subplot(2,3,6),imhist(h);title('histeq test image');