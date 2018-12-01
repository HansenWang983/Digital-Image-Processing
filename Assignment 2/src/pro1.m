clear

car_img = imread('../car.png');

wheel_img = imread('../wheel.png');

[height,width] = size(car_img)
[m,n] = size(wheel_img)

% cor = filter2(wheel_img,car_img);
% con = conv2(car_img,wheel_img);
cpad = floor(n/2)
rpad = floor(m/2)
pad_img = padarray(car_img,[rpad cpad],0,'both');

% correlation_map的大小与原图像相同
G = zeros(height,width);
% uint8 -> double 
pattern = double(wheel_img(:));  
% 矩阵二范数
norm_sub = norm(pattern);  

for i=1:height
    for j=1:width
        subMatr=pad_img(i:i+m-1,j:j+n-1);  
        windows=double(subMatr(:));  
        norm_windows = norm(windows);
        % G(i,j)=sum(sum(windows.*pattern)) / (norm_sub*norm_windows);
        G(i,j) = sum(sum(corrcoef(windows,pattern)));

        % rot = rot90(subMatr,2);
        % rot = double(rot(:));
        % norm_rot = norm(rot);
        % G(i,j)=sum(sum(windows.*pattern)) / (norm_rot.*norm_rot); 
        % numerator = power(sum(sum(windows.*pattern)),2);
        % denominator = power(sum(sum(windows)),2).*power(sum(sum(pattern)),2);
        % G(i,j) = numerator / denominator;
    end  
end  
% [x,y] = find(G>=0.95)
% 转换成行向量
array = reshape(G,[1,height*width]);
% 排序
[e,I] = sort(array,'descend');
% 得到矩阵坐标
for i = 1:6
  x(i) = mod(I(i),height)+1;
  y(i) = ceil(I(i) / height);
end

figure,  
subplot(1,3,1) , imshow(wheel_img),title('pattern'),set(gca,'FontSize',20);
subplot(1,3,2) , imshow(car_img),title('detected image'),set(gca,'FontSize',20);
% 对每个坐标为中心，按照pattern的大小画出矩形框
for i = 1:length(x)
	hold on  
	plot([y(i)-rpad,y(i)+n-1-rpad],[x(i)-cpad,x(i)-cpad],'-r','LineWidth',3);  
	plot([y(i)+n-1-rpad,y(i)+n-1-rpad],[x(i)-cpad,x(i)+m-1-cpad],'-r','LineWidth',3);  
	plot([y(i)-rpad,y(i)+n-1-rpad],[x(i)+m-1-cpad,x(i)+m-1-cpad],'-r','LineWidth',3);  
  plot([y(i)-rpad,y(i)-rpad],[x(i)-cpad,x(i)+m-1-cpad],'-r','LineWidth',3);  
  plot(y(i),x(i),'r.')
  G(x(i),y(i))
end
% 显示相关值矩阵，注意转换成uint8
% 单位化
for i=1:height
    m = max(G(i,:));
    A(i,:) = G(i,:) / m;
end
subplot(1,3,3) , imshow(uint8(255*A)),title('correlation map'),set(gca,'FontSize',20);
