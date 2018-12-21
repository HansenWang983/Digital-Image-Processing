function [res,f_res] = motion_filter(img,a,b,T)
    [M,N] = size(img);
    P = M / 2;
    Q = N / 2;
    % 中心变换
    [X,Y]=meshgrid(1:N,1:M);
    img = img.*(-1).^(X+Y);
    % 频谱矩阵
    f_res = zeros(M,N);
    % 傅立叶变换
    F = fft2(img);

    noise = 500^0.5 * randn([M, N]);
    Fn = fftshift(fft2(noise));
    % 运动模糊滤波
    for u = 1:M
        for v = 1:N
            d = pi*((u-P)*a+(v-Q)*b);
            if d == 0
                h(u,v) = T;
            else
                h(u,v) = T*sin(d)*exp(-j*d)/(d);
            end
        end
    end
    f_res = h .* F;
    % 反傅立叶变换
    res = real(ifft2(f_res));
    % % 反中心变换
    [X,Y] = meshgrid(1:N,1:M);
    res = res.*(-1).^(X+Y);
end