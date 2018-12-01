function f_equal = my_histeq(f,L)
    [height,width] = size(f);
    hr = imhist(f,L);
    pr = hr / numel(f);
    s = round((L-1)*cumsum(pr));
    
    f_equal = f;
    for i=1:height
        for j=1:width
            f_equal(i,j) = s(f(i,j));
        end
    end
    imwrite(f_equal,'../result/myhisteq_img.jpg')
end