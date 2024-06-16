
for i=1:3
index = i;

path1 = ['./test_imgs/ir/',num2str(index),'.jpg'];
path2 = ['./test_imgs/vis/',num2str(index),'.jpg'];
ir_base = ['./test_imgs/ir_base/',num2str(index),'.jpg'];
ir_detail = ['./test_imgs/ir_detail/',num2str(index),'.jpg'];
vis_base = ['./test_imgs/vis_base/',num2str(index),'.jpg'];
vis_detail = ['./test_imgs/vis_detail/',num2str(index),'.jpg'];

image1 = imread(path1);
image2 = imread(path2);

if size(image1,3)>1
    image1 = rgb2gray(image1);
    image2 = rgb2gray(image2);
end

image1 = im2double(image1);
image2 = im2double(image2);

lambda = 0.8;
disp('latlrr');
tic
X1 = image1;
[Z1,L1,E1] = latent_lrr(X1,lambda);
X2 = image2;
[Z2,L2,E2] = latent_lrr(X2,lambda);
toc
disp('latlrr');

I_lrr1 = X1*Z1;
I_saliency1 = L1*X1;
I_lrr1 = max(I_lrr1,0);
I_lrr1 = min(I_lrr1,1);
I_saliency1 = max(I_saliency1,0);
I_saliency1 = min(I_saliency1,1);
I_e1 = E1;

I_lrr2 = X2*Z2;
I_saliency2 = L2*X2;
I_lrr2 = max(I_lrr2,0);
I_lrr2 = min(I_lrr2,1);
I_saliency2 = max(I_saliency2,0);
I_saliency2 = min(I_saliency2,1);
I_e2 = E2;

imwrite(I_lrr1,ir_base,'jpg');
imwrite(I_saliency1,ir_detail,'jpg');
imwrite(I_lrr2,vis_base,'jpg');
imwrite(I_saliency2,vis_detail,'jpg');

end

