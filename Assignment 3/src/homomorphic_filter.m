function [res,f_res] = homomorphic_filter(img,D0,H,L,C)
    [M,N] = size(img);
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
                h = 1 / (1 + (sqrt(u^2+v^2)/D0)^(2*n));
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
    res = uint8(res.*(-1).^(X+Y));
