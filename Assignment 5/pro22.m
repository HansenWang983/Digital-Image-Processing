clear;

img2 = imread('blobz2.png');

img2 =double(img2);
[M,N] = size(img2);
[X,Y]=meshgrid(1:N,1:M);

mx=mean2(X);
my=mean2(Y);
mxy=mean2(X.*Y);
mx2=mean2(X.^2);
my2=mean2(Y.^2);
mx2y=mean2(X.^2.*Y);
mxy2=mean2(X.*Y.^2);
mx2y2=mean2(X.^2.*Y.^2);
C=[1,mx,my,mxy;mx,mx2,mxy,mx2y;my,mxy,my2,mxy2;mxy,mx2y,mxy2,mx2y2];
CI=inv(C);

mL=mean2(img2);
mLx=mean2(img2.*X);
mLy=mean2(img2.*Y);
mLxy=mean2(img2.*X.*Y);
v=[mL,mLx,mLy,mLxy]';
aL=CI*v;

% 光照函数
GL=aL(1)+aL(2).*X+aL(3).*Y+aL(4).*X.*Y;
% 去除光照影响
RL=img2-GL;

ymax=255;ymin=0;
xmax = max(max(RL)); %求得RL中的最大值
xmin = min(min(RL)); %求得RL中的最小值
OutImg = uint8(round((ymax-ymin)*(RL-xmin)/(xmax-xmin) + ymin)); %归一化并取整

% 全局阈值计算
count=0;
T=mean2(OutImg);
flag=false;
while ~flag
    count=count+1;
    g=OutImg>T;
    Tnext=0.5*(mean(OutImg(g))+mean(OutImg(~g)));
    flag=abs(T-Tnext)<0.5;
    T=Tnext;
end
img2 = uint8(img2);
% 转换为二值图像
g=im2bw(OutImg,150/255);
figure;subplot(3,2,1);imshow(img2);title('原图像');
subplot(3,2,2);imhist(img2);title('原直方图');
subplot(3,2,3);imshow(OutImg);title('增强后图像');
subplot(3,2,4);imhist(OutImg);title('增强后直方图');
subplot(3,2,5);imshow(g);title('切割后图像');

figure;
imshow(g);title('切割后图像');