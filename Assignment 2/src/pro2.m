clear;

car_img = imread('../sport_car.pgm');
[height,width] = size(car_img)

for i = 1:height
	for j = 1:width
		r1 = randi([0,255]);
		r2 = randi([0,255]);
		if r1 < r2
			t = r1;
			r1 = r2;
			r2 = t;
		end
		t1(i,j) = r1;
		t2(i,j) = r2;
	end
end

for i = 1:height
	for j = 1:width
		if car_img(i,j)>t1(i,j)
			salt_img(i,j) = 255;
		elseif car_img(i,j)<t2(i,j)
			salt_img(i,j) = 0;
		else
			salt_img(i,j) = car_img(i,j);
		end
	end
end

% salt_img = imnoise(car_img,'salt & pepper',0.4);

% 窗口大小
n = 3;
% unit8 -> double
s1 = double(salt_img);
mf_img = s1;

for i = 1:height-n+1
	for j = 1:width-n+1
		% 获得区域图像
		subMatr = s1(i:i+(n-1),j:j+(n-1));
		% 转换成行向量
		array = reshape(subMatr.',1,9);
		% 得到中值
		med = median(array);
		% 将中心像素的值做替换
		mf_img(i+(n-1)/2,j+(n-1)/2) = med;
		% mf_img(i,j) = med;
	end
end

mf_img = uint8(mf_img);

test_img = medfilt2(salt_img);

subplot(2,2,1) , imshow(car_img),title('origin image'),set(gca,'FontSize',20);
subplot(2,2,2) , imshow(salt_img),title('noise image'),set(gca,'FontSize',20);
subplot(2,2,3) , imshow(mf_img),title('image with my implementation '),set(gca,'FontSize',20);
subplot(2,2,4) , imshow(test_img),title('medfilt2 image'),set(gca,'FontSize',20);


