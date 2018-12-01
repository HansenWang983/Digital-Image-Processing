clear all;
f = imread('../river.JPG');
[height,width] = size(f)
% whos f
% K = imfinfo ../river.JPG

% figure,imshow(f)
% figure,imshow(f,[100,200])

% 大于等于最大值和小于等于最小值的将显示为黑色和白色，对于动态范围低的图片
% figure,imshow(f,[])

% 直方图
h = imhist(f,256);
% figure , imhist(f,256)
% ylim('auto')
% 绘制直方图为条形图，10和灰度级为一组
horz = 1:10:256;
h1 = h(1:10:256);
% bar(horz,h1)
% 绘制直方图为杆状图，10和灰度级为一组
% stem(horz,h1,'fill')
% 设置横纵坐标的最大最小值
% axis([0 255 0 1500])
% 设置刻度
% set(gca,'xtick',0:50:255)
% set(gca,'ytick',0:100:1500)

% 归一化
p = h / numel(f);
h2 = p(1:10:256);
% stem(horz,h2,'fill')

% 直方图均衡化
g = histeq(f,256);
figure , imshow(g)
figure , imhist(g,256)

% imwrite(g,'../result/histeq_img.tiff')
% 变换函数图像
cdf = cumsum(p);
x = linspace(0,1,256);
axis([0 1 0 1])
% plot(x,cdf)

