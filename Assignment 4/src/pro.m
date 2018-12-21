clear all;

img = imread('../book_cover.jpg');
[M, N] = size(img);

% 参数如下：
% p,q为频率中心，a,b,T为运动模糊参数，
% k为维纳滤波参数
% m,n 分别为高斯噪声的均值和方差
p = M / 2 + 1.0;
q = N / 2 + 1.0;
a = 0.1;
b = 0.1;
T = 1;
k = 0.01;
m = 0;
n = 500;

% 读取图像
img = double(img);

% 中心变换
[X,Y]=meshgrid(1:N,1:M);
img = img.*(-1).^(X+Y);

% 对原图像进行傅里叶变换
F = fft2(img);

% 生成运动模糊的傅里叶变换，即退化函数，频率中心在(p,q)
H = zeros(M, N);
for u = 1 : M
    for v = 1 : N
        d = pi * ((u-p)*a + (v-q)*b);
        if (d == 0)
            H(u,v) = T;
        else
            H(u,v) =  T * sin(d) / d * exp(-1i * (d));
        end
    end
end

% 生成均值为m和方差为n的高斯噪声的傅里叶变换
noise = m + sqrt(n) * randn([M, N]);
Fn = fftshift(fft2(noise));

% 生成维纳滤波的傅里叶变换
Wiener = (abs(H).^2) ./ (abs(H).^2 + k) ./ H;

% 生成运动模糊图像
MotionBlurred_f = F .* H;
MotionBlurred = real(ifft2(MotionBlurred_f));
% 反中心变换
[X,Y] = meshgrid(1:N,1:M);
MotionBlurred = MotionBlurred.*(-1).^(X+Y);
subplot(231),imshow(MotionBlurred,[]),title('运动模糊图像');

% 生成运动模糊图像，加上高斯噪声
BlurredNoisy_f = F .* H + Fn;
BlurredNoisy = real(ifft2(BlurredNoisy_f));
% 反中心变换
[X,Y] = meshgrid(1:N,1:M);
BlurredNoisy = BlurredNoisy.*(-1).^(X+Y);
subplot(232),imshow(BlurredNoisy,[]),title('运动模糊加噪图像');


% 获得模糊图像的频域信息
MotionBlurred_temp = MotionBlurred;
% 中心变换
[X,Y]=meshgrid(1:N,1:M);
MotionBlurred_temp = MotionBlurred_temp.*(-1).^(X+Y);
% 对运动模糊图像进行傅里叶变换
MotionBlurred_temp_f = fft2(MotionBlurred_temp);

% 对运动模糊图像进行逆滤波
MotionBlurred_Inverse_f =  MotionBlurred_temp_f ./ H;
MotionBlurred_Inverse = real(ifft2(MotionBlurred_Inverse_f));
% 反中心变换
[X,Y] = meshgrid(1:N,1:M);
MotionBlurred_Inverse = MotionBlurred_Inverse.*(-1).^(X+Y);
subplot(233),imshow(MotionBlurred_Inverse,[]),title('运动模糊图像进行逆滤波');

% 对运动模糊图像进行维纳滤波
MotionBlurred_Wiener_f =  MotionBlurred_temp_f .* Wiener;
MotionBlurred_Wiener = real(ifft2(MotionBlurred_Wiener_f));
% 反中心变换
[X,Y] = meshgrid(1:N,1:M);
MotionBlurred_Wiener = MotionBlurred_Wiener.*(-1).^(X+Y);
subplot(234),imshow(MotionBlurred_Wiener,[]),title('运动模糊图像进行维纳滤波');



% 获得模糊加噪图像的频域信息
BlurredNoisy_temp = BlurredNoisy;
% 中心变换
[X,Y]=meshgrid(1:N,1:M);
BlurredNoisy_temp = BlurredNoisy_temp.*(-1).^(X+Y);
% 对运动模糊图像进行傅里叶变换
BlurredNoisy_temp_f = fft2(BlurredNoisy_temp);

% 对模糊加噪声图像进行逆滤波
BlurredNoisy_Inverse_f = BlurredNoisy_temp_f ./ H;
BlurredNoisy_Inverse = real(ifft2(BlurredNoisy_Inverse_f));
% 反中心变换
[X,Y] = meshgrid(1:N,1:M);
BlurredNoisy_Inverse = BlurredNoisy_Inverse.*(-1).^(X+Y);
subplot(235),imshow(BlurredNoisy_Inverse,[]),title('模糊加噪图像进行逆滤波');

% 对模糊加噪图像进行维纳滤波
BlurredNoisy_Wiener_f = BlurredNoisy_temp_f .* Wiener;
BlurredNoisy_Wiener = real(ifft2(BlurredNoisy_Wiener_f));
% 反中心变换
[X,Y] = meshgrid(1:N,1:M);
BlurredNoisy_Wiener = BlurredNoisy_Wiener.*(-1).^(X+Y);
subplot(236),imshow(BlurredNoisy_Wiener,[]),title('模糊加噪图像进行维纳滤波');
