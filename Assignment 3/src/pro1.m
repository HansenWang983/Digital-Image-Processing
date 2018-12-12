clear 

barb_img = imread('../barb.png');

[M,N] = size(barb_img);

subplot(321),imshow(barb_img,[]),title('原图像f(x,y)')
% subplot(321),imshow(log(1+abs(fft2(barb_img))),[]),title('原频谱图像f(x,y)')

% 以(-1)^{(x+y)}乘以输入图像进行中心变换
[X,Y]=meshgrid(1:N,1:M);
% 类型转换
barb_img = double(barb_img);
barb_img = barb_img.*(-1).^(X+Y);

subplot(322),imshow(uint8(barb_img),[]),title('空域中心化调制图像')

[res_10,f_res10] = butterworth_filter(barb_img,1,10);
[res_20,f_res20] = butterworth_filter(barb_img,1,20);
[res_40,f_res40] = butterworth_filter(barb_img,1,40);
[res_80,f_res80] = butterworth_filter(barb_img,1,80);

subplot(323),imshow(res_10,[]),title('D0=10 butterworth filter低通滤波')
subplot(324),imshow(res_20,[]),title('D0=20 butterworth filter低通滤波')
subplot(325),imshow(res_40,[]),title('D0=40 butterworth filter低通滤波')
subplot(326),imshow(res_80,[]),title('D0=80 butterworth filter低通滤波')

% 显示频谱图像
% subplot(322),imshow(log(1+abs(fft2(barb_img))),[]),title('空域中心化调制后的频谱图像')
% subplot(323),imshow(f_res10,[]),title('D0=10 ')
% subplot(324),imshow(f_res20,[]),title('D0=20 ')
% subplot(325),imshow(f_res40,[]),title('D0=40 ')
% subplot(326),imshow(f_res80,[]),title('D0=80 ')
