% testing.m
clear;
load training.mat;

tic;
% 记录识别正确数
correct_num = 0;
for x = 1:40
    for y = testing_set(x,:)
        temp_mat = imread(['../att_faces/s',num2str(x),'/',num2str(y),'.pgm']);   
        % 显示测试图像
        % figure,
        % subplot(1,2,1),imshow(temp_mat);
        % title('Test Image');

        %************投影降维度测试图片***************************
        temp_mat = reshape(temp_mat,d,1);
        temp_mat = double(temp_mat) - img_mean;
        project_test = [];
        % project_test k*1
        project_test = Vk_mat' * temp_mat;

        %****************计算二范数*****************************
        com_dist = [];
        % Ei_Face k*40
        % i = 1:40
        for i = 1:size(Ei_Face,2)
            vec_dist = norm(project_test - Ei_Face(:,i),2);
            com_dist = [com_dist vec_dist];
        end
        %************筛选出距离最小的样本图片*********************
        [match_min,match_index] = min(com_dist);
        if match_index == x
            correct_num = correct_num+1;
        end

        % 显示识别图像，用于全局训练
        % directories = ceil(match_index / 10);
        % subject = mod(match_index,10);
        % if subject == 0
        %     subject = 10;
        % end
        % recognize_img = imread(['../att_faces/s',num2str(directories),'/',num2str(subject),'.pgm']);   
        % subplot(1,2,2),imshow(recognize_img);
        % title('Recognized Image');
    end
end
t1 = toc;
disp(['识别正确的图像数: ',num2str(correct_num),'/120']);
disp(['识别系统的正确率: ',num2str(correct_num/120)]);
disp(['测试用时(s): ',num2str(t1)]);