clear

office_img = imread('../office.jpg');

office_img = rgb2gray(office_img);

whos office_img

[M,N] = size(office_img);

subplot(321),imshow(office_img,[]),title('原图像f(x,y)')

[X,Y]=meshgrid(1:N,1:M);
% 类型转换
office_img = double(office_img);
office_img = office_img.*(-1).^(X+Y);

subplot(322),imshow(uint8(office_img),[]),title('空域中心化调制图像')

[res1 , f_res1] = homomorphic_filter(office_img,1,2,0.25,1);
[res5 , f_res5] = homomorphic_filter(office_img,5,2,0.25,1);
[res10 , f_res10] = homomorphic_filter(office_img,10,2,0.25,1);
[res20 , f_res20] = homomorphic_filter(office_img,0.1,2,0.25,1);

subplot(323),imshow(res1,[]),title('D0=1 homomorphic filter低通滤波')
subplot(324),imshow(res5,[]),title('D0=5 butterworth filter低通滤波')
subplot(325),imshow(res10,[]),title('D0=10 butterworth filter低通滤波')
subplot(326),imshow(res20,[]),title('D0=20 butterworth filter低通滤波')

% range = max(res10(:)) - min(res10(:));
% minmat = ones(M,N).*min(res10(:));
% res10 = 255.*(res10-minmat)./range;
% subplot(324),imshow(uint8(res10)),title('D0=10 homomorphic filter低通滤波')
