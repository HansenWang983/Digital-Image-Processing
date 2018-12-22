% Reference code for Project2 of DIP @SJTU IVM Lab

clear all;close all;clc;
% read original image
im=imread('../book_cover.jpg');
I=im2double(im);
[m,n]=size(I);
% fourier transform
F=fftshift(fft2(I));
figure;
subplot(2,2,1),
imshow(I,[]);
title('Original image')
% subplot(2,2,2),
% imshow(log(1+abs(F)),[]);
% title('fourier spectrum')
% generate additive gaussian noise
u=1:m; v=1:n;
[u,v]=meshgrid(u,v);
variance=500/255^2;
noise=imnoise(zeros(m,n),'gaussian',0,variance);
subplot(2,2,3),
imshow(noise,[]),
title('Additive noise-500')
% blurred filter
a=0.1;
b=0.1;
T=1;
uv=(u-m/2-1).*a+(v-n/2-1).*b+eps;
H=T.*sin(pi.*uv).*exp(-1i.*pi.*uv)./(pi.*uv);
G=H.*F;
f_blurred=ifft2(G);
subplot(2,2,2),
imshow(abs(f_blurred),[]);
title('T=1,a=0.1,b=0.1-blurred image');
f_blurred_noised=abs(f_blurred)+noise;
 subplot(2,2,4),
imshow(f_blurred_noised,[]);
title('Blurred and noised image')

G=fftshift(fft2(f_blurred_noised));
% direct inverse filtering
F_inverse=G./H;
R_inverse=abs(ifft2(F_inverse));
figure;
subplot(2,2,1)
imshow(R_inverse,[])
title('Direct inverse filtering');

%Parametric Wiener filtering
i=2;
for k=[1,1e-3,1e-4]
Fuvyp=(H.*conj(H)).*G./(H.*(H.*conj(H)+k));
Rtuxy=abs(ifft2(Fuvyp));
% subplot(2,2,i),
figure,
imshow(Rtuxy,[]),
i=i+1;
title(strcat('Parametric Wiener filtering, k=', num2str(k)));
end

