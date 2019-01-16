clear;

binary_array = [
    0,0,0,0,0,0,0;
    0,0,1,1,0,0,0;
    0,0,0,1,0,0,0;
    0,0,0,1,1,0,0;
    0,0,1,1,1,1,0;
    0,0,1,1,1,0,0;
    0,1,0,1,0,1,0;
    0,0,0,0,0,0,0;
];

Structure_Element1 = [1,1,1];
Structure_Element2 = [1,0; 1,1];

origin =  0;
[m,n] = size(binary_array);

padding1 = [zeros(m,2),binary_array];
% 结构单元1膨胀
result1 = zeros(m,n);
for x = 1:m
    for y = 1:n
        if Structure_Element1(1,1) == padding1(x,y) || Structure_Element1(1,2) == padding1(x,y+1) || Structure_Element1(1,3) == padding1(x,y+2)
            result1(x,y) = 1;
        end
    end
end

padding1 = [binary_array,zeros(m,2)];
% 结构单元1腐蚀
result2 = zeros(m,n);
for x = 1:m
    for y = 1:n 
        if Structure_Element1 == padding1(x,y:y+2)
            result2(x,y) = 1;
        end
    end
end


padding2 = [zeros(1,n+1);binary_array,zeros(m,1)];
% 结构单元2膨胀
result3 = zeros(m,n);
for x = 1:m
    for y = 1:n 
        if Structure_Element2(1,1) == padding2(x,y) || Structure_Element2(2,1) == padding2(x+1,y) || Structure_Element2(2,2) == padding2(x+1,y+1)
            result3(x,y) = 1;
        end
    end
end

padding2 = [zeros(m,1),binary_array;zeros(1,n+1)];
Structure_Element2 = [1,1 ; 0,1];
% 结构单元2腐蚀
result4 = zeros(m,n);
for x = 1:m
    for y = 1:n 
        if Structure_Element2(1,1) == padding2(x,y) && Structure_Element2(1,2) == padding2(x,y+1) && Structure_Element2(2,2) == padding2(x+1,y+1)
            result4(x,y) = 1;
        end
    end
end

% 结构单元1开变换，先腐蚀再膨胀
result5 = zeros(m,n);
padding3 = [zeros(m,2),result2];
for x = 1:m
    for y = 1:n
        if Structure_Element1(1,1) == padding3(x,y) || Structure_Element1(1,2) == padding3(x,y+1) || Structure_Element1(1,3) == padding3(x,y+2)
            result5(x,y) = 1;
        end
    end
end

% 结构单元1闭变换，先膨胀再腐蚀
result6 = zeros(m,n);
padding4 = [result1,zeros(m,2)];
for x = 1:m
    for y = 1:n 
        if Structure_Element1 == padding4(x,y:y+2)
            result6(x,y) = 1;
        end
    end
end

Structure_Element2 = [1,0; 1,1];
% 结构单元2开变换，先腐蚀再膨胀
result7 = zeros(m,n);
padding5 = [zeros(1,n+1);result4,zeros(m,1)];
for x = 1:m
    for y = 1:n
        if Structure_Element2(1,1) == padding5(x,y) || Structure_Element2(2,1) == padding5(x+1,y) || Structure_Element2(2,2) == padding5(x+1,y+1)
            result7(x,y) = 1;
        end
    end
end

% 结构单元2闭变换，先膨胀再腐蚀
result8 = zeros(m,n);
padding6 = [zeros(m,1),result3;zeros(1,n+1)];
Structure_Element2 = [1,1 ; 0,1];
for x = 1:m
    for y = 1:n
        if Structure_Element2(1,1) == padding6(x,y) && Structure_Element2(1,2) == padding6(x,y+1) && Structure_Element2(2,2) == padding6(x+1,y+1)
            result8(x,y) = 1;
        end
    end
end