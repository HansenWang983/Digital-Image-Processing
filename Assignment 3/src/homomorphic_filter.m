function [res,f_res] = homomorphic_filter(img,D0,H,L,C)
    % 转换为灰度图像
    img =double(rgb2gray(img));
    % 取对数
    img = log(1+img);

    [M,N] = size(img);

    % 中心变换
    [X,Y]=meshgrid(1:N,1:M);
    img = img.*(-1).^(X+Y);
    % 频谱矩阵
    f_res = zeros(M,N);
    % 傅立叶变换
    F = fft2(img);
    % 高通同态滤波
    for u = 1:M
        for v = 1:N
            d = u^2+v^2;
            h = (H-L).*(1-exp(-C.*(d./D0^2)))+L;
            f_res(u,v) = h*F(u,v);
        end
    end
    % 反傅立叶变换
    res = real(ifft2(f_res));
    % 反中心变换
    [X,Y] = meshgrid(1:N,1:M);
    res = res.*(-1).^(X+Y);
    % 取指数
    res = exp(res)-1;
    % 频谱矩阵
    f_res = log(1+abs(f_res));
 
    subplot(221),imshow(res,[]),title(['D0=',num2str(D0),' homomorphic filter高通图像'])
    subplot(222),imshow(f_res,[]),title(['D0=',num2str(D0),' homomorphic filter高通频谱'])

