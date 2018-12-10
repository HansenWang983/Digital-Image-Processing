clear all

barb_img = imread('../barb.png');

[M,N] = size(barb_img);


subplot(221),imshow(barb_img,[]),title('ԭͼ��f(x,y)')

% ��(-1)^{(x+y)}��������ͼ��������ı任
% [Y,X]=meshgrid(1:M,1:N);
% barb_img(x,y) = barb_img.*(-1).^(X+Y);
for x = 1:M
    for y = 1:N
        barb_img(x,y) = barb_img(x,y).*(-1).^(x+y);
    end
end 

subplot(222),imshow(barb_img,[]),title('�������Ļ�����ͼ��')

F = fft2(barb_img);

subplot(223),imshow(log(1+abs(F)),[]),title('����ҶƵ��')
