% 灰度图直方图匹配matlab(不使用histeq)
clear
 
im      = imread('../EightAM.png');
imRef   = imread('../LENA.png');
hist    = imhist(im);                % Compute histograms
histRef = imhist(imRef);
cdf     = cumsum(hist) / numel(im);  % Compute CDFs
cdfRef  = cumsum(histRef) / numel(imRef);
 
% Compute the mapping
M   = zeros(1,256);
for idx = 1 : 256
    [tmp,ind] = min(abs(cdf(idx) - cdfRef))
    M(idx)    = ind-1;
end
 
% Now apply the mapping to get first image to make
% the image look like the distribution of the second image
imMatch = M(double(im)+1);
 
figure;%显示原图像、匹配图像和匹配后的图像
subplot(1,3,1),imshow(im,[]);title('origin image');
subplot(1,3,2),imshow(imRef,[]);title('reference image');
subplot(1,3,3),imshow(imMatch,[]);title('match image');
figure;%显示原图像、匹配图像和匹配后图像的直方图
subplot(3,1,1),imhist(im,64);title('origin hist');
subplot(3,1,2),imhist(imRef,64);title('reference hist');
subplot(3,1,3),imhist(uint8(imMatch),64);title('match hist');
