clear all;

lena_img = imread('../LENA.png');
eight_img = imread('../EightAM.png');

[height_lena,width_lena] = size(lena_img)
[height_eight,width_eight] = size(eight_img)

lena_hist = imhist(lena_img,256);
eight_hist = imhist(eight_img,256);

figure , imhist(lena_img,256)
% figure , imhist(eight_img,256)

pr = eight_hist / numel(eight_img);
s1 = round(255*cumsum(pr));

pz = lena_hist / numel(lena_img);
s2 = round(255*cumsum(pz));

g = eight_img;
for i=1:height_eight
    for j=1:width_eight
        s = s1(eight_img(i,j));
        indices = find(s2==s);
        while isempty(indices)
            s = s -1;
            indices = find(s2==s);
        end
        indices = indices(1);
        g(i,j) = indices;
    end
end
% imwrite(g,'../EightAM_match.jpg')
t = imhistmatch(eight_img,lena_img);

figure;
subplot(2,4,1),imshow(eight_img);title('origin image');
subplot(2,4,2),imshow(lena_img);title('reference image');
subplot(2,4,3),imshow(g,[]);title('my match image');
subplot(2,4,4),imshow(t);title('test match image');
subplot(2,4,5),imhist(eight_img);title('origin hist');
subplot(2,4,6),imhist(lena_img);title('reference hist');
subplot(2,4,7),imhist(uint8(g));title('my match image');
subplot(2,4,8),imhist(t);title('test match image');