function [RI_out, hist] = solve_coverage_tgv(RI_nonneg, ORytov, retphase, Count, p)
% SOLVE_COVERAGE_TGV  Standalone SA2-CR reconstruction (no inheritance / no plotting / no sigma analysis)
% Two-stage structure-tensor adaptive anisotropic Hessian-L1 + z-TV + phase constraint.
%
% ---------------------------------------------------------------------------
% Method : SA2-CR (Spatially-Adaptive Anisotropic Composite Regularization)
% Authors: D. Wu, H. Wang, L. Zhong
%
% Code   : D. Wu, H. Wang, L. Zhong, "Spatially-Adaptive Anisotropic Composite
%          Regularization for High-Fidelity 3D Reconstruction of Optical
%          Diffraction Tomography," GitHub (2026).
%          https://github.com/Woody-Wu67/Spatially-Adaptive-Anisotropic-Composite-Regularization-for-Optical-Diffraction-Tomography
%
% Paper  : [To be added upon acceptance/publication]
%
% If you use this code, please cite the repository above (and the paper once available).
% ---------------------------------------------------------------------------
%
% Usage:
%   load('your_data.mat');   % must contain RI_nonneg, ORytov, (retphase), (Count)
%   p = default_params();    % see end of file; edit parameters as needed
%   [RI_out, hist] = solve_coverage_tgv(RI_nonneg, ORytov, retphase, Count, p);
%
% Inputs:
%   RI_nonneg  Ny x Nx x Nz   initial refractive index volume        (required)
%   ORytov     Ny x Nx x Nz   frequency-domain scattering potential, complex (required)
%   retphase   Ny x Nx        retrieved phase (optional; pass [] to disable phase constraint)
%   Count      Ny x Nx x Nz   frequency coverage count (optional; pass [] to auto-generate from ORytov)
%   p          struct         parameters (optional; pass [] for defaults)
%
% Outputs:
%   RI_out     Ny x Nx x Nz   reconstructed refractive index volume
%   hist       struct         convergence history (residuals, costs; no figures)

    if nargin<3, retphase=[]; end
    if nargin<4, Count=[]; end
    if nargin<5 || isempty(p), p=default_params(); end

    modules='Hessian';
    if p.use_ztv,modules=[modules '+zTV'];end
    if p.use_phase_constraint,modules=[modules '+Phase'];end
    if p.use_adaptive,modules=[modules '+ST-Adaptive'];end
    fprintf('\n  Modules: %s\n',modules);

    RI_nonneg=single(RI_nonneg); ORytov=single(ORytov);
    [Ny,Nx,Nz]=size(RI_nonneg); dz=single(p.resolution(3));
    if p.use_gpu, RI_nonneg=gpuArray(RI_nonneg); ORytov=gpuArray(ORytov); end

    %% Coverage mask
    if ~isempty(Count)
        Cr=single(Count); if p.use_gpu,Cr=gpuArray(Cr);end
        M=single(Cr>0);
        fprintf('  [Coverage] max=%d | nonzero=%.1f%%\n',...
            gather(max(Cr(:))),gather(100*sum(M(:))/numel(M)));
    else
        M=single(abs(ORytov)>1e-10);
    end

    %% Initial scattering potential
    u_init=(RI_nonneg/single(p.RI_bg)).^2-single(1);
    U_init=fft3c(u_init);

    %% Hessian anisotropic weights (global, Stage 1)
    sig=single(p.sigma);
    w=single([1, 1, sig, 2, 2*sqrt(sig), 2*sqrt(sig)]);
    fprintf('  [Aniso] sigma=%.2f\n',sig);

    %% Frequency-domain Hessian
    [HtH,Hxx_f,Hyy_f,Hzz_f,Hxy_f,Hxz_f,Hyz_f]=discrete_hessian_freq(Ny,Nx,Nz,p.use_gpu);
    H_f={Hxx_f,Hyy_f,Hzz_f,Hxy_f,Hxz_f,Hyz_f};
    mu=single(p.mu); alpha=single(p.alpha_hess);
    rho_H=single(p.rho_hess); rho_v=single(p.rho_v); beta=single(p.beta);
    tau=alpha/(rho_H+eps);

    %% z difference operator
    alpha_tv_z=single(p.alpha_tv_z); rho_tv_z=single(p.rho_tv_z);
    kz_1d=reshape(single((-floor(Nz/2):ceil(Nz/2)-1))/single(Nz),1,1,[]);
    if p.use_gpu,kz_1d=gpuArray(kz_1d);end
    Gz_f=exp(-1i*2*single(pi)*kz_1d)-single(1);
    Gz_f_conj=conj(Gz_f); Gz_f_sq=abs(Gz_f).^2;
    tau_gz=alpha_tv_z/(rho_tv_z+eps);

    %% Phase constraint
    use_pc=p.use_phase_constraint && ~isempty(retphase);
    kz0_idx=ceil((Nz+1)/2); mu_phase=single(0);
    U_kz0_target=[]; F_bp_2d_sq=[];
    if use_pc
        if ndims(retphase)==3,phase_raw=single(retphase(:,:,1));else,phase_raw=single(retphase);end
        if p.use_gpu,phase_raw=gpuArray(phase_raw);end
        k0=single(2*pi/p.wavelength);
        sum_u_target=phase_raw*2/(k0*dz*single(p.RI_bg));
        U_kz0_target=fftshift(fft2(ifftshift(sum_u_target)));
        kx_2d=single((-floor(Nx/2):ceil(Nx/2)-1))/single(Nx);
        ky_2d=single((-floor(Ny/2):ceil(Ny/2)-1))'/single(Ny);
        [KX_2d,KY_2d]=meshgrid(kx_2d,ky_2d); K_sq_2d=KX_2d.^2+KY_2d.^2;
        if p.use_gpu,K_sq_2d=gpuArray(K_sq_2d);end
        sig_h=single(p.phase_bandpass_sigma_high); sig_l=single(p.phase_bandpass_sigma_low);
        F_bp_2d=exp(-2*single(pi)^2*sig_h^2*K_sq_2d)-exp(-2*single(pi)^2*sig_l^2*K_sq_2d);
        F_bp_2d_sq=abs(F_bp_2d).^2;
        mu_phase=single(p.phase_constraint_weight)*mu;
    end

    fprintf('  [Data] mu=%.1f | [Hess] alpha=%.1e\n',mu,alpha);

    %% Global denominator
    denom_global=mu*M+rho_H*HtH+rho_v+beta;
    if p.use_ztv,denom_global=denom_global+rho_tv_z*Gz_f_sq;end
    if use_pc,denom_global(:,:,kz0_idx)=denom_global(:,:,kz0_idx)+mu_phase*F_bp_2d_sq;end

    %% Pack operators
    ops.H_f=H_f; ops.w=w; ops.tau=tau; ops.mu=mu; ops.rho_H=rho_H; ops.rho_v=rho_v;
    ops.Gz_f=Gz_f; ops.Gz_f_conj=Gz_f_conj; ops.tau_gz=tau_gz;
    ops.rho_tv_z=rho_tv_z; ops.alpha_tv_z=alpha_tv_z;
    ops.use_pc=use_pc; ops.kz0_idx=kz0_idx; ops.mu_phase=mu_phase;
    ops.U_kz0_target=U_kz0_target; ops.F_bp_2d_sq=F_bp_2d_sq;

    %% ==================== Stage 1 ====================
    fprintf('\n==== Stage 1: Global Anisotropic Hessian ====\n');
    [v_s1,hist]=run_admm(u_init,U_init,M,denom_global,ops,p,...
        p.max_iter,p.tol,[],'S1');
    RI_s1=gather(u_to_RI(v_s1,p));
    fprintf('  Stage 1 RI: [%.4f, %.4f]\n',min(RI_s1(:)),max(RI_s1(:)));

    if ~p.use_adaptive
        RI_out=RI_s1;
        hist.params=p; hist.modules=modules;
        return;
    end

    %% ==================== Structure tensor adaptive weights ====================
    fprintf('\n==== Structure Tensor Adaptive Weights ====\n');
    tau_map_dir=compute_st_weights(v_s1,p,w);

    %% ==================== Stage 2 ====================
    fprintf('\n==== Stage 2: ST-Adaptive Hessian ====\n');
    if p.s2_warm_start
        u_s2_start=(single(RI_s1)/single(p.RI_bg)).^2-single(1);
        if p.use_gpu,u_s2_start=gpuArray(u_s2_start);end
        fprintf('  [Stage 2] warm-start\n');
    else
        u_s2_start=u_init;
        fprintf('  [Stage 2] cold-start (default)\n');
    end
    [v_s2,hist2]=run_admm(u_s2_start,U_init,M,denom_global,ops,p,...
        p.s2_max_iter,p.tol,tau_map_dir,'S2');
    hist.s2=hist2;
    RI_s2=gather(u_to_RI(v_s2,p));
    fprintf('  Stage 2 RI: [%.4f, %.4f]\n',min(RI_s2(:)),max(RI_s2(:)));

    %% ==================== Post: z median filter ====================
    if p.use_z_medfilt
        fprintf('\n==== Post: z-median filter (k=%d) ====\n',p.z_medfilt_size);
        RI_out=z_median_filter(RI_s2,p.z_medfilt_size);
        hist.RI_stage2_raw=RI_s2;
    else
        RI_out=RI_s2;
    end
    hist.params=p; hist.RI_stage1=RI_s1; hist.modules=modules;
end

%% ======================================================================
%% ADMM engine
%% ======================================================================
function [v,hist]=run_admm(u_start,U_init,M,denom_global,ops,p,...
        max_iter,tol,tau_map,stage_name)
    [Ny,Nx,Nz]=size(u_start);
    N_total=single(numel(u_start));
    use_dir=iscell(tau_map);  % cell = directional adaptive; empty = global

    u=u_start; v=max(u,single(0)); Bv=zeros(Ny,Nx,Nz,'single');
    P=cell(1,6); B=cell(1,6);
    for idx=1:6,P{idx}=zeros(Ny,Nx,Nz,'single');B{idx}=P{idx};end
    Qz=zeros(Ny,Nx,Nz,'single'); B_gz=zeros(Ny,Nx,Nz,'single');
    if p.use_gpu
        v=gpuArray(v); Bv=gpuArray(Bv);
        for idx=1:6,P{idx}=gpuArray(P{idx});B{idx}=gpuArray(B{idx});end
        Qz=gpuArray(Qz); B_gz=gpuArray(B_gz);
    end

    hist=struct('rel',nan(max_iter,1),'data_fidelity',nan(max_iter,1),...
        'hess_reg',nan(max_iter,1),'tv_z',nan(max_iter,1),...
        'total_cost',nan(max_iter,1),'primal_res_v',nan(max_iter,1));
    tic_s=tic;

    for iter=1:max_iter
        u_old=u;
        %% U-subproblem
        HtPB_f=hessian_adjoint(P,B,ops.H_f);
        V_Bv_f=fft3c(v-Bv);
        numer=ops.mu*M.*U_init+ops.rho_H*HtPB_f+ops.rho_v*V_Bv_f;
        if p.use_ztv
            numer=numer+ops.rho_tv_z*ops.Gz_f_conj.*fft3c(Qz-B_gz);
        end
        if ops.use_pc
            numer(:,:,ops.kz0_idx)=numer(:,:,ops.kz0_idx)+ops.mu_phase*ops.F_bp_2d_sq.*ops.U_kz0_target;
        end
        U_curr=numer./denom_global;
        u=real(ifft3c(U_curr));

        %% v (non-negative)
        v_temp=u+Bv;
        if p.apply_RI_constraint
            v_max_val=(single(p.n_max)/single(p.RI_bg))^2-1;
            v=min(max(v_temp,single(0)),v_max_val);
        else
            v=max(v_temp,single(0));
        end

        %% P: Hessian
        V_f=fft3c(v);
        hess_L1=single(0);
        for idx=1:6
            Hv=real(ifft3c(ops.H_f{idx}.*V_f));
            if use_dir
                P{idx}=shrink(Hv+B{idx}, ops.tau .* tau_map{idx});
            else
                P{idx}=shrink(Hv+B{idx}, ops.tau * ops.w(idx));
            end
            B{idx}=B{idx}+Hv-P{idx};
            hess_L1=hess_L1+ops.w(idx)*L1norm(Hv);
        end

        %% z-TV
        tv_z_L1=single(0);
        if p.use_ztv
            Gz_v=real(ifft3c(ops.Gz_f.*V_f));
            if use_dir
                Qz=shrink(Gz_v+B_gz, ops.tau_gz .* tau_map{3}/ops.w(3));
            else
                Qz=shrink(Gz_v+B_gz, ops.tau_gz);
            end
            B_gz=B_gz+Gz_v-Qz;
            tv_z_L1=L1norm(Gz_v);
        end

        Bv=Bv+u-v;

        %% Diagnostics (no plotting)
        hist.data_fidelity(iter)=gather((0.5*ops.mu/N_total)*sum(M(:).*abs(U_curr(:)-U_init(:)).^2));
        hist.hess_reg(iter)=gather(single(p.alpha_hess)*hess_L1);
        hist.tv_z(iter)=gather(ops.alpha_tv_z*tv_z_L1);
        hist.total_cost(iter)=hist.data_fidelity(iter)+hist.hess_reg(iter)+hist.tv_z(iter);
        hist.primal_res_v(iter)=gather(norm3(u-v));
        rel_change=gather(norm3(u-u_old)/(norm3(u_old)+1e-12));
        hist.rel(iter)=rel_change;
        if p.verbose&&(mod(iter,10)==0||iter==1)
            fprintf('  [%s-%3d] rel=%.2e | Cost=%.2e\n',stage_name,iter,rel_change,hist.total_cost(iter));
        end
        if rel_change<tol,fprintf('  %s converged at iter %d\n',stage_name,iter);break;end
    end
    fprintf('  %s done: %d iters, %.1f s\n',stage_name,iter,toc(tic_s));
end

%% ======================================================================
%% Structure tensor adaptive weights (no plotting)
%% ======================================================================
function tau_map_dir=compute_st_weights(u_pilot,p,w)
    u_cpu=gather(u_pilot);[Ny,Nx,Nz]=size(u_cpu);
    u_max=max(u_cpu(:))+eps;

    %% 1. Morphological segmentation
    bw=u_cpu > p.adapt_seg_threshold * u_max;
    r=p.adapt_morph_radius; se=strel('disk',r);
    mask=false(Ny,Nx,Nz);
    for z=1:Nz
        mask(:,:,z)=imfill(imclose(bw(:,:,z),se),'holes');
    end
    for y=1:Ny
        slice=squeeze(mask(y,:,:));
        se_z=strel('line',2*r+1,90);
        mask(y,:,:)=reshape(imfill(imclose(slice,se_z),'holes'),1,Nx,Nz);
    end
    sample_mask=single(mask);
    if p.use_gpu,sample_mask=gpuArray(sample_mask);end
    fprintf('  [Segment] sample=%.1f%%\n',gather(100*sum(sample_mask(:))/numel(sample_mask)));

    %% 2. Gradient
    if p.use_gpu,u_g=gpuArray(single(u_cpu));else,u_g=single(u_cpu);end
    gx=zeros(Ny,Nx,Nz,'single');gy=gx;gz=gx;
    if p.use_gpu,gx=gpuArray(gx);gy=gpuArray(gy);gz=gpuArray(gz);end
    gx(:,2:end-1,:)=(u_g(:,3:end,:)-u_g(:,1:end-2,:))/2;
    gy(2:end-1,:,:)=(u_g(3:end,:,:)-u_g(1:end-2,:,:))/2;
    gz(:,:,2:end-1)=(u_g(:,:,3:end)-u_g(:,:,1:end-2))/2;

    %% 3. Structure tensor (locally smoothed gradient outer product)
    st_sig=p.st_sigma;
    S_xx=gaussian_smooth_3d(gx.*gx, st_sig);
    S_yy=gaussian_smooth_3d(gy.*gy, st_sig);
    S_zz=gaussian_smooth_3d(gz.*gz, st_sig);

    %% 4. Directional adaptive weights
    % S_ii large = edge along direction i -> keep edge (low weight)
    % S_ii small = flat along direction i -> smooth (high weight)
    S_sum=S_xx+S_yy+S_zz;
    % kappa2: median over sample region only (avoid background zeros dominating)
    valid_S=gather(S_sum(sample_mask>0));
    if isempty(valid_S)||numel(valid_S)<10
        kappa2=single(gather(median(S_sum(:))))+eps;
    else
        kappa2=single(median(valid_S))+eps;
    end
    fprintf('  [ST] sigma=%.1f | kappa2=%.2e (sample-only)\n',st_sig,kappa2);

    dw_xx=single(1)./(single(1)+S_xx/kappa2);
    dw_yy=single(1)./(single(1)+S_yy/kappa2);
    dw_zz=single(1)./(single(1)+S_zz/kappa2);
    dw_floor=single(p.adapt_dw_floor);   % lower bound: prevent edge-noise amplification
    dw_xx=max(dw_xx,dw_floor);
    dw_yy=max(dw_yy,dw_floor);
    dw_zz=max(dw_zz,dw_floor);
    dw_xy=sqrt(dw_xx.*dw_yy);   % cross terms: geometric mean
    dw_xz=sqrt(dw_xx.*dw_zz);
    dw_yz=sqrt(dw_yy.*dw_zz);

    %% 5. Background boost (soft transition to avoid boundary over-kill)
    bg=single(p.adapt_bg_boost);
    mask_soft=gaussian_smooth_3d(sample_mask,single(1.5));
    bg_factor=single(1)+(bg-single(1))*(single(1)-mask_soft);

    %% 6. Combine: global w x directional weight x background boost
    dw_all={dw_xx, dw_yy, dw_zz, dw_xy, dw_xz, dw_yz};
    tau_map_dir=cell(1,6);
    for idx=1:6
        tm=single(w(idx)).*dw_all{idx}.*bg_factor;
        if p.adapt_tau_smooth>0
            tm=gaussian_smooth_3d(tm,p.adapt_tau_smooth);
        end
        tau_map_dir{idx}=tm;
    end
end

%% ======================================================================
%% Utility functions
%% ======================================================================
function out=gaussian_smooth_3d(vol,sigma)
    ks=ceil(3*sigma);x=-ks:ks;
    g=single(exp(-x.^2/(2*sigma^2)));g=g/sum(g);
    if isa(vol,'gpuArray'),g=gpuArray(g);end
    out=convn(vol,reshape(g,[],1,1),'same');
    out=convn(out,reshape(g,1,[],1),'same');
    out=convn(out,reshape(g,1,1,[]),'same');
end
function out=z_median_filter(vol,k)
    % z-direction median filter: smooths z stratification only, keeps xy resolution
    [~,~,Nz]=size(vol);
    out=vol; half=floor(k/2);
    for z=1:Nz
        z_lo=max(1,z-half); z_hi=min(Nz,z+half);
        out(:,:,z)=median(vol(:,:,z_lo:z_hi),3);
    end
end
function [HtH,Hxx,Hyy,Hzz,Hxy,Hxz,Hyz]=discrete_hessian_freq(Ny,Nx,Nz,use_gpu)
    kx=single((-floor(Nx/2):ceil(Nx/2)-1))/single(Nx);
    ky=single((-floor(Ny/2):ceil(Ny/2)-1))'/single(Ny);
    kz=reshape(single((-floor(Nz/2):ceil(Nz/2)-1))/single(Nz),1,1,[]);
    [KX,KY,KZ]=meshgrid(kx,ky,squeeze(kz));
    if use_gpu,KX=gpuArray(KX);KY=gpuArray(KY);KZ=gpuArray(KZ);end
    Hxx=-4*sin(single(pi)*KX).^2;Hyy=-4*sin(single(pi)*KY).^2;Hzz=-4*sin(single(pi)*KZ).^2;
    Hxy=-sin(2*single(pi)*KX).*sin(2*single(pi)*KY);
    Hxz=-sin(2*single(pi)*KX).*sin(2*single(pi)*KZ);
    Hyz=-sin(2*single(pi)*KY).*sin(2*single(pi)*KZ);
    HtH=abs(Hxx).^2+abs(Hyy).^2+abs(Hzz).^2+abs(Hxy).^2+abs(Hxz).^2+abs(Hyz).^2;
end
function HtPB_f=hessian_adjoint(P,B,H_f)
    HtPB_f=zeros(size(P{1}),'like',fft3c(P{1}));
    for idx=1:6,HtPB_f=HtPB_f+conj(H_f{idx}).*fft3c(P{idx}-B{idx});end
end
function y=shrink(x,t),y=sign(x).*max(abs(x)-t,0);end
function RI=u_to_RI(u,p),RI=single(p.RI_bg)*sqrt(max(u,single(-1+1e-6))+single(1));end
function K=fft3c(u),K=fftshift(fftn(ifftshift(u)));end
function u=ifft3c(K),u=fftshift(ifftn(ifftshift(K)));end
function n=norm3(x),n=sqrt(sum(abs(x(:)).^2));end
function n=L1norm(x),n=sum(abs(x(:)));end

%% ======================================================================
%% Default parameters
%% ======================================================================
function p=default_params()
    p.RI_bg=1.3325; p.wavelength=0.641; p.resolution=[0.1563,0.1563,0.1563];
    p.n_min=1.3325; p.n_max=1.45;
    p.max_iter=300; p.tol=1e-6; p.verbose=true; p.use_gpu=true;
    p.mu=5;
    p.alpha_hess=2e-4; p.rho_hess=0.1; p.sigma=1.8;
    p.use_ztv=true; p.alpha_tv_z=1e-4; p.rho_tv_z=0.1;
    p.use_phase_constraint=true; p.phase_constraint_weight=3;
    p.phase_bandpass_sigma_low=20.0; p.phase_bandpass_sigma_high=1.0;
    p.rho_v=1; p.apply_RI_constraint=true; p.beta=1e-8;
    p.s2_warm_start=false;
    p.adapt_dw_floor=0.2; p.use_adaptive=true; p.st_sigma=4;
    p.adapt_bg_boost=8; p.adapt_seg_threshold=0.06; p.adapt_morph_radius=3;
    p.adapt_tau_smooth=1.5; p.s2_max_iter=300;
    p.use_z_medfilt=true; p.z_medfilt_size=3;
end