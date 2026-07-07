clc, clear;
close all;

%% 参数设置
NA=1.4;
NA_ill=0.95;
RI_bg=1.518;
RI_sp=1.550;
wavelength=0.641;
resolution=[1 1 1]*wavelength/2/(NA+NA_ill);
size=[128 128 101];

outer_size=size;
inner_size=round(ones(1,3) *2 ./ resolution);

%% 生成 MLA phantom
phantom=zeros(outer_size,'single');
d1=single(reshape(single(1:outer_size(1)),[],1,1)-(floor(outer_size(1)/2)+1));
d2=single(reshape(single(1:outer_size(2)),1,[],1)-(floor(outer_size(2)/2)+1));
d3=single(reshape(single(1:outer_size(3)),1,1,[])-(floor(outer_size(3)/2)+1));

tot_sample=8;  % antialiasing
for sample_num=1:tot_sample
    switch sample_num
        case 1, sample_shift=[1 1 1];
        case 2, sample_shift=[-1 1 1];
        case 3, sample_shift=[1 -1 1];
        case 4, sample_shift=[-1 -1 1];
        case 5, sample_shift=[1 1 -1];
        case 6, sample_shift=[-1 1 -1];
        case 7, sample_shift=[1 -1 -1];
        case 8, sample_shift=[-1 -1 -1];
    end
    sample_shift=(1/4).*sample_shift;

    d1_norm= 2.*(d1+sample_shift(1))./inner_size(1);
    d2_norm= 2.*(d2+sample_shift(2))./inner_size(2);
    d3_norm= 2.*(d3+sample_shift(3))./inner_size(3);
    r_norm=sqrt(d1_norm.^2+d2_norm.^2+d3_norm.^2);

    % MLA: 三维立方网格排列微球结构
    n_x = 5;  % x方向球数
    n_y = 5;  % y方向球数
    n_z = 1;  % z方向球数
    spacing_x = inner_size(1);
    spacing_y = inner_size(2);
    spacing_z = inner_size(3);

    for ix = 1:n_x
        for iy = 1:n_y
            for iz = 1:n_z
                shift_x = round((ix - (n_x+1)/2) * spacing_x);
                shift_y = round((iy - (n_y+1)/2) * spacing_y);
                shift_z = round((iz - (n_z+1)/2) * spacing_z);
                phantom = phantom + circshift(single(r_norm < 1), [shift_x, shift_y, shift_z]);
            end
        end
    end
    phantom(phantom > 0.5) = 1;
end

%% 归一化并映射折射率
phantom=phantom-min(phantom(:));
if max(phantom(:))~=0
    phantom=phantom./max(phantom(:));
end
RI = RI_bg + phantom .* (RI_sp - RI_bg);



figure(66)
orthosliceViewer(real(RI));
colormap jet
title('RI')          % 显示反演结果
    ce = colorbar('northoutside','horizional');
    ce.Position = [0.65, 0.1, 0.2, 0.05];
    clim([1.518, 1.555]);
set(gca,'Clim',[1.518,1.555]);
ce.Label.String = 'ni';