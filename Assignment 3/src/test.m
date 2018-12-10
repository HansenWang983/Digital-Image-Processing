clear all

f = zeros(512,512);
f(246:266,230:276)=1;
subplot(221),imshow(f,[]),title('单狭缝图像')

% 对图像进行二维快速傅里叶变换
F = fft2(f);
S = abs(F);
% 显示幅度谱
subplot(222),imshow(S,[]),title('幅度谱（频谱坐标原点在左上角）')

% 把频谱坐标原点由左上角移至屏幕中央
Fc =fftshift(F);
Fd=abs(Fc);
subplot(223),imshow(Fd,[]),title('幅度谱（频谱坐标原点在屏幕中央）')

ratio=max(Fd(:))/min(Fd(:))
% ratio = 2.3306e+007,动态范围太大，显示器无法正常显示

% 取对数
S2=log(1+abs(Fc)); 
subplot(224),imshow(S2,[]),title('以对数方式显示频谱')


