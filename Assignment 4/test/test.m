clear all;
close all;
clc;

image = imread('../book_cover.jpg');
[m, n] = size(image);
% 参数如下：
% p,q为频率中心，a,b,T为运动模糊参数，
% k为维纳滤波参数， r为约束最小二乘方参数
% M,V 分别为高斯噪声的均值和方差
p = m / 2 + 1.0;
q = n / 2 + 1.0;
a = 0.1;
b = 0.1;
T = 1;
k = 0.05;
r = 0.5;
M = 0;
V = 500;

% 对原图像进行傅里叶变换以及中心化
fp = double(image);
F = fftshift(fft2(fp));

% 生成运动模型的傅里叶变换，中心在(p,q)
Mo = zeros(m, n);
for u = 1 : m
    for v = 1 : n
        temp = pi * ((u-p)*a + (v-q)*b);
        if (temp == 0)
            Mo(u,v) = T;
        else
            Mo(u, v) =  T * sin(temp) / temp * exp(-1i * (temp));
        end
    end
end

% 生成维纳滤波的傅里叶变换
Wiener = (abs(Mo).^2) ./ (abs(Mo).^2 + k) ./ Mo;

% 生成约束最小二乘方滤波的傅里叶变换
lp = [0, -1, 0; -1, 4, -1; 0, -1, 0];
Flp = fftshift(fft2(lp, m, n));
Hw = conj(Mo) ./ (abs(Mo).^2 + r * abs(Flp).^2);

% 生成高斯噪声的傅里叶变换
noise = 500^0.5 * randn([m, n]);
Fn = fftshift(fft2(noise));

% 运动模糊图像，并且加上高斯噪声
image1 = zeros(m, n, 'uint8');
G1 = F .* Mo + Fn;
gp1 = ifft2(fftshift(G1));
g1 = real(gp1);
% 归一化图像到 [0, 255];
mmax = max(g1(:));
mmin = min(g1(:));
range = mmax-mmin;
for i = 1 : m
    for j = 1 : n
        image1(i,j) = uint8(255 * (g1(i, j)-mmin) / range);
    end
end

% 为了接近真实情况，对归一化之后的加噪图像进行逆滤波
F2 = fftshift(fft2(image1));
% 直接逆滤波
G2 = F2 ./ Mo;
gp2 = ifftshift(G2);
g2 = real(gp2);D

% 维纳滤波
G3 = F2 .* Wiener;
gp3 = ifft2(fftshift(G3));
g3 = real(gp3);

% 约束最小二乘方滤波
G4 = F2 .* Hw;
gp4 = ifft2(fftshift(G4));
g4 = real(gp4);
subplot(2,3,1), imshow(image), title('原图');
subplot(2,3,2), imshow(image1), title('运动加噪图像');
subplot(2,3,3), imshow(g2, []), title('直接逆滤波');
subplot(2,3,4), imshow(g3, []), title('维纳滤波');
subplot(2,3,5), imshow(g4, []), title('约束最小二乘方滤波');