function  h = my_histmatch(f,g)
    [height_f,width_f] = size(f)
    [height_g,width_g] = size(g)

    hist_f = imhist(f);
    hist_g = imhist(g);

    cdf_f = cumsum(hist_f) / numel(f); 
    cdf_g = cumsum(hist_g) / numel(g);

    % 建立r->z映射
    M   = zeros(1,256);
    for i = 1 : 256
        % 对每一个灰度值，寻找与通过G(z)得到的s值差最小的那一项，其下标减1即为对应的z灰度值。
        [tmp,ind] = min(abs(cdf_f(i) - cdf_g));
        % 将坐标减1变为灰度值
        M(i) = ind-1;
    end
    h = f;
    for i=1:height_f
        for j=1:width_f
            h(i,j) = M(f(i,j)+1);
        end
    end
    imwrite(h,'../result/my_histmatch.jpg')
end