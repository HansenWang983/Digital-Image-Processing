clear all

f = zeros(512,512);
f(246:266,230:276)=1;
subplot(221),imshow(f,[]),title('������ͼ��')

% ��ͼ����ж�ά���ٸ���Ҷ�任
F = fft2(f);
S = abs(F);
% ��ʾ������
subplot(222),imshow(S,[]),title('�����ף�Ƶ������ԭ�������Ͻǣ�')

% ��Ƶ������ԭ�������Ͻ�������Ļ����
Fc =fftshift(F);
Fd=abs(Fc);
subplot(223),imshow(Fd,[]),title('�����ף�Ƶ������ԭ������Ļ���룩')

ratio=max(Fd(:))/min(Fd(:))
% ratio = 2.3306e+007,��̬��Χ̫����ʾ���޷�������ʾ

% ȡ����
S2=log(1+abs(Fc)); 
subplot(224),imshow(S2,[]),title('�Զ�����ʽ��ʾƵ��')


