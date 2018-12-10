clear all

barb_img = imread('../barb.png');

[M,N] = size(barb_img);


subplot(221),imshow(barb_img,[]),title('原图像f(x,y)')

% 以(-1)^{(x+y)}乘以输入图像进行中心变换
% [Y,X]=meshgrid(1:M,1:N);
% barb_img(x,y) = barb_img.*(-1).^(X+Y);
for x = 1:M
    for y = 1:N
        barb_img(x,y) = barb_img(x,y).*(-1).^(x+y);
    end
end 

subplot(222),imshow(barb_img,[]),title('空域中心化调制图像')

F = fft2(barb_img);

subplot(223),imshow(log(1+abs(F)),[]),title('傅里叶频谱')


% 频谱图像尺寸
[N1,N2]=size(F);                 
n=2;                                    
d0=80;                              
% 数据圆整      
n1=fix(N1/2);                      
n2=fix(N2/2);                            
for i=1:N1                               
    for j=1:N2
            d=sqrt((i-n1)^2+(j-n2)^2);
            if d==0
                h=0;                     
            else
                % Butterworth低通的幅频响应
                h=1/(1+(d/d0)^(2*n));     
            end
            % 图像矩阵计算处理
            result(i,j)=h*F(i,j);          
    end
end

% 对傅立叶变换结果取绝对值，然后取对数
% F2 = log(1+abs(result));        
% 将计算后的矩阵用图像表示
% subplot(224),imshow(F2),title('Butterworth滤波后的频谱图像')

result=ifftshift(result);               % 傅立叶变换平移
X2=ifft2(result);                       % 图像傅立叶逆变换
X3=uint8(real(X2));                     % 数据类型转换
subplot(224),imshow(X3)               % 显示处理后的图像
xlabel('(d) Butterworth低通滤波图像');