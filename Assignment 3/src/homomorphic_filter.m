function [res,f_res] = homomorphic_filter(img,D0,H,L,C)
    [M,N] = size(img);
    % 取对数
    % img = log(1+img);
    f_res = zeros(M,N);
    F = fft2(img);

    for u = 1:M
        for v = 1:N
            d = u^2+v^2;
            h = (H-L).*(1-exp(-C.*(d./D0^2)))+L;
            f_res(u,v) = h*F(u,v);
        end
    end
    res = real(ifft2(f_res));
    [X,Y] = meshgrid(1:N,1:M);
    % 取指数
    % res = exp(res)-1;
    res = uint8(res.*(-1).^(X+Y));
    f_res = log(1+abs(f_res));
end