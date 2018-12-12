clear

office_img = imread('../office.jpg');

[r1,f1] = homomorphic_filter(office_img,1000,2,0.25,1);
[r2,f2] = butterworth_high_filter(office_img,1,1000);
