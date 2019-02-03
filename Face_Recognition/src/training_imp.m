% training_imp.m
clear;
load random_gene.mat;

Img_Mat = [];
row = 112;
col = 92;
d = row*col;
k = 30;

tic;
for x = 1:40 
     % temp_set d*7
     temp_set = [];
     for y = training_set(x,:)
          temp_mat = imread(['../att_faces/s',num2str(x),'/',num2str(y),'.pgm']);   
          temp_mat = reshape(temp_mat,[d,1]); %将图片转化为一个列向量
          temp_set = [temp_set temp_mat];
     end
     % Img_Mat d*40
     Img_Mat = [Img_Mat mean(temp_set,2)];
end

% display the mean image
% space = 2; %间距的大小
% subplot_row = 5; %子图行数
% subplot_col = 8; %子图列数
% immat = zeros(space*(subplot_row+1)+row*subplot_row,space*(subplot_col+1)+col*subplot_col);
% immat = uint8(immat);
% for ii = 1:subplot_row
%      for kk = 1:subplot_col
%           index = (ii-1)*subplot_col+kk;
%           temp_mat = Img_Mat(:,index);
%           temp_mat = reshape(temp_mat,[row col]);
%           temp_max = max(max(temp_mat));
%           temp_min = min(min(temp_mat));
%           temp_range = temp_max - temp_min;
%           temp_mat = round(255*(temp_mat - temp_min)/temp_range);
%           immat((ii-1)*row+1+ii*space:ii*(row+space),(kk-1)*col+kk*space+1:kk*(col+space)) = temp_mat;
%      end
% end
% figure
% imshow(immat)

% differ_mat d*N
differ_mat = [];
img_mean = mean(Img_Mat,2);
% 40张平均图像
num_img = size(Img_Mat,2);

for i = 1:num_img
     temp_mat = double(Img_Mat(:,i)) - img_mean;
     differ_mat = [differ_mat temp_mat];
end

% C_mat N*N
C_mat = differ_mat' * differ_mat;
[eiv eic] = eig(C_mat);   %求取特征向量eiv以及特征值eic

% 降序排列特征值
[dd,ind] = sort(diag(eic),'descend');
eic_sort = eic(ind,ind);
eiv_sort = eiv(:,ind);
% Wk_mat N*k
Wk_mat = eiv_sort(:,1:k);

% Vk_mat d*k
Vk_mat = differ_mat * Wk_mat;

% normalize columns of Vk_mat
Vk_mat = normc(Vk_mat);

% Ei_Face k*N
Ei_Face = Vk_mat' * differ_mat ;     %得到投影子空间的坐标

t1 = toc;
disp(['k: ',num2str(k)]);
disp(['训练用时(s): ',num2str(t1)]);
save training.mat img_mean Vk_mat Ei_Face d
save eigenvector_sort.mat eiv_sort differ_mat img_mean