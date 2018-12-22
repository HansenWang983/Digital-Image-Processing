function [res,f_res] = inverse_filter(img,a,b,T)
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
    % 运动模糊滤波
    for u = 1:M
        for v = 1:N
            d = pi*((u-P)*a+(v-Q)*b);
            if d == 0
                h = T;
            else
                h = T*sin(d)*exp(-j*d)/(d);
            end
            f_res(u,v) = F(u,v)/h;
        end
    end
    % 反傅立叶变换
    res = real(ifft2(f_res));
    % % 反中心变换
    [X,Y] = meshgrid(1:N,1:M);
    res = res.*(-1).^(X+Y);
end