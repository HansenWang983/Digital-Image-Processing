% eigenface_display.m
clear;
load eigenvector_sort.mat;

k = 39;
row = 112;
col = 92;

% Vk_mat d*k
Vk_mat = eiv_sort(:,1:k);

% Ei_Face k*N
Ei_Face = Vk_mat' * differ_mat ;     %得到投影子空间的坐标

% 将 k 个特征脸组成全新的矩阵显示
% space = 2; %间距的大小
% subplot_row = 8; %子图行数
% subplot_col = 10; %子图列数
% immat = zeros(space*(subplot_row+1)+row*subplot_row,space*(subplot_col+1)+col*subplot_col);
% immat = uint8(immat);
% for ii = 1:subplot_row
%      for kk = 1:subplot_col
%           index = (ii-1)*subplot_col+kk;
%           temp_mat = Vk_mat(:,index);
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


% 重构40张平均图像
% project_sample d*N
project_sample = [];
project_sample = Vk_mat * Ei_Face;
% 计算重构误差，二范数表示
disp(['k: ',num2str(k)]);
err=norm(project_sample-differ_mat,2);
disp(['重构误差（欧式距离）: ',num2str(err)]);
project_sample = project_sample + img_mean;
% 显示重构的人脸平均图像
% space = 2; %间距的大小
% subplot_row = 5; %子图行数
% subplot_col = 8; %子图列数
% immat = zeros(space*(subplot_row+1)+row*subplot_row,space*(subplot_col+1)+col*subplot_col);
% immat = uint8(immat);
% for ii = 1:subplot_row
%      for kk = 1:subplot_col
%           index = (ii-1)*subplot_col+kk;
%           temp_mat = project_sample(:,index);
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
