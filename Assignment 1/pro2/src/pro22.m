clear all;

lena_img = imread('../LENA.png');
eight_img = imread('../EightAM.png');

g = my_histmatch(eight_img,lena_img);

% for test
t = imhistmatch(eight_img,lena_img);
imwrite(t,'../result/test_histmatch.jpg')
% p = histeq(eight_img,imhist(lena_img));
% s = cumsum(imhist(t)) / numel(t);
% s = round(s*255);
% m = cumsum(imhist(g)) / numel(g);
% m = round(m*255);
% n = cumsum(imhist(p)) / numel(p);
% n = round(n*255);
% figure , plot(1:1:256,m,1:1:256,n)

%从左到右显示原图像、匹配图像，匹配后的以及imhistmatch返回的图像
%从左到右显示原图像、匹配图像，匹配后的以及imhistmatch返回的直方图
figure;
subplot(2,4,1),imshow(eight_img);title('origin image');
subplot(2,4,2),imshow(lena_img);title('reference image');
subplot(2,4,3),imshow(g);title('my match image');
subplot(2,4,4),imshow(t);title('test match image');
subplot(2,4,5),imhist(eight_img);title('origin hist');
subplot(2,4,6),imhist(lena_img);title('reference hist');
subplot(2,4,7),imhist(g);title('my match image');
subplot(2,4,8),imhist(t);title('test match image');