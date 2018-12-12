function [res,f_res] = butterworth_high_filter(img,n,D0)
    % 转换为灰度图像
    img =double(rgb2gray(img));
    % 中心变换
    [M,N] = size(img);
    [X,Y]=meshgrid(1:N,1:M);
    img = img.*(-1).^(X+Y);
    
    f_res = zeros(M,N);
    % 对图像进行二维快速傅里叶变换
    F = fft2(img);
    % 频谱图像大小与空域图像相同
    for u = 1:M
        for v = 1:N
            % butterworth低通滤波器
            if D0 == 0 
                h = 0;
            else
                h = 1 / (1 + (D0/sqrt(u^2+v^2))^(2*n));
            end
            % 与滤波函数相乘，等于空域卷积
            f_res(u,v) = F(u,v)*h;
        end
    end
    % DFT反变换取实部
    res = real(ifft2(f_res));
    % 频谱矩阵取对数
    f_res = log(1+abs(f_res));
    % 反中心变换
    [X,Y]=meshgrid(1:N,1:M);
    res = res.*(-1).^(X+Y);

    subplot(223),imshow(res,[]),title(['D0=',num2str(D0),' butterworth filter高通图像'])
    subplot(224),imshow(f_res,[]),title(['D0=',num2str(D0),' butterworth filter高通频谱'])
