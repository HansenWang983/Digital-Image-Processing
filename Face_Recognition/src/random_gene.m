% random_gene.m
clear;

for x = 1:40 
    % 每个目录随机选取7个作为训练样本，剩余3个作为测试样本
    idx = randperm(10);
    training_set(x,:) = idx(1:7);
    testing_set(x,:) = idx(8:10);
end

save random_gene.mat training_set testing_set