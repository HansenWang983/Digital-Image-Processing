clear;

img1=imread('blobz1.png');

[m,n]=size(img1);

count=0;
T=mean2(img1);
flag=false;
while ~flag
    count = count+1;
    g = img1>T;
    T_next = 0.5*(mean(img1(g))+mean(img1(~g)));
    flag = abs(T-T_next)<0.5;
    T = T_next;
end

% 将灰度图像转换成二值图像
g=im2bw(img1,T/255);
figure;subplot(2,2,1);imshow(img1);title('原图像');
subplot(2,2,2);imhist(img1);title('直方图');
subplot(2,2,3);imshow(g);title('切割后图像');

