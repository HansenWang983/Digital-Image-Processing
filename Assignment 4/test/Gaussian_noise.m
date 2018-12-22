function [res,f_res] = Gaussian_noise(img,u,v,a,b,T)
    [M,N] = size(img);
    P = M / 2;
    Q = N / 2;
    % 中心变换
%     [X,Y]=meshgrid(1:N,1:M);
%     img = img.*(-1).^(X+Y);
    % 频谱矩阵
    f_res = zeros(M,N);
    % 傅立叶变换
%     F = fft2(img);
    F = fftshift(fft2(img));

    % 生成高斯噪声的傅里叶变换
    noise = 500^0.5 * randn([M, N]);
    Fn = fftshift(fft2(noise));
    % 运动模糊滤波
    for u = 1:M
        for v = 1:N
            d = pi*((u-P)*a+(v-Q)*b);
            if d == 0
                h = T;
            else
                h = T*sin(d)*exp(-j*d)/(d);
            end
            f_res(u,v) = h*F(u,v)+Fn(u,v);
        end
    end
    % 反傅立叶变换
    res = real(ifft2(fftshift(f_res)));
    % % 反中心变换
%     [X,Y] = meshgrid(1:N,1:M);
%     res = res.*(-1).^(X+Y);
end