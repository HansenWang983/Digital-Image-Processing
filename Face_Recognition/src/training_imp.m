% training_imp.m
clear;

Img_Mat = [];
row = 112;
col = 92;
d = row*col;
k = 25;

tic;
for x = 1:40 
     % 每个目录随机选取7个作为训练样本，剩余3个作为测试样本
     idx = randperm(10);
     training_set(x,:) = idx(1:7);
     testing_set(x,:) = idx(8:10);
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
for x = 1:5
    for y = 1:8
        temp_mat = Img_Mat(:,(x-1)*8+y);
        temp_mat = reshape(temp_mat,[row col]);
        subplot(5,8,(x-1)*8+y),imshow(temp_mat,[]);
    end
end

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
Ei_Face = Vk_mat' * differ_mat ;     %得到协方差矩阵的特征向量组成的投影子空间

% display the Eigenface
figure
for x = 1:k
    temp_mat = Vk_mat(:,x);
    temp_mat = reshape(temp_mat,[row col]);
    subplot(5,5,x),imshow(temp_mat,[]);
end

% project_sample d*N
% project_sample = [];
% project_sample = Vk_mat * Ei_Face;
t1 = toc;
disp(['训练用时(s): ',num2str(t1)]);
save training.mat img_mean Vk_mat Ei_Face testing_set d