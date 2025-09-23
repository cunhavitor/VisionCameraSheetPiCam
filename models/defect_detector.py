import cv2
import numpy as np
import time

# =========================
#  MODO ESTÁTICO (troca aqui)
# =========================
mode = "Simple"   # <- "Simple" ou "Full"

# ---------------------------
# Hiper-parâmetros do modo Simple (podes afinar no código)
# ---------------------------
SIMPLE_ALPHA = 0.5           # peso DSSIM
SIMPLE_BETA  = 0.3           # peso cor (Δa/Δb)
SIMPLE_GAMMA = 0.2           # peso residual passa-banda
SIMPLE_PERCENTILE = 99.5     # percentil do score para binarizar (no ROI)
SIMPLE_BAND_KSIZE = 5        # janela da Gaussiana (ímpar)
SIMPLE_BAND_SIGMA1 = 1.0     # sigma1 para DoG
SIMPLE_BAND_SIGMA2 = 2.0     # sigma2 para DoG
SIMPLE_MORPH_KERNEL = 3      # morfologia final do Simple
SIMPLE_MORPH_ITERS  = 1

def _percentile_bin(img_float01, roi_mask_u8, pct):
    """Binariza img [0..1] por percentil dentro do ROI (0..100)."""
    roi = roi_mask_u8.astype(bool)
    vals = img_float01[roi]
    if vals.size == 0:
        return np.zeros_like(roi_mask_u8, dtype=np.uint8)
    thr = np.percentile(vals, float(pct))
    m = (img_float01 >= thr).astype(np.uint8) * 255
    return cv2.bitwise_and(m, roi_mask_u8)

def _norm01(x):
    x = x.astype(np.float32)
    mn, mx = x.min(), x.max()
    if mx <= mn:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)

# ---------------------------
# Helpers MS-SSIM (no skimage)
# ---------------------------

def _gaussian_kernel(ksize=7, sigma=1.5):
    k = cv2.getGaussianKernel(ksize, sigma)
    w = k @ k.T
    return w / w.sum()

def _gaussian_ssim_map(x, y, ksize=7, sigma=1.5, c1=(0.01*255)**2, c2=(0.03*255)**2):
    """
    SSIM local (mapa) entre x e y (uint8), janela gaussiana.
    Retorna mapa SSIM em float32 [0..1].
    """
    if x.dtype != np.float32:
        x = x.astype(np.float32)
    if y.dtype != np.float32:
        y = y.astype(np.float32)

    w = _gaussian_kernel(ksize, sigma)
    mu_x = cv2.filter2D(x, -1, w, borderType=cv2.BORDER_REFLECT)
    mu_y = cv2.filter2D(y, -1, w, borderType=cv2.BORDER_REFLECT)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = cv2.filter2D(x*x, -1, w, borderType=cv2.BORDER_REFLECT) - mu_x2
    sigma_y2 = cv2.filter2D(y*y, -1, w, borderType=cv2.BORDER_REFLECT) - mu_y2
    sigma_xy = cv2.filter2D(x*y, -1, w, borderType=cv2.BORDER_REFLECT) - mu_xy

    # SSIM por pixel
    num = (2*mu_xy + c1) * (2*sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)

    ssim = np.ones_like(num, dtype=np.float32)
    mask = den > 0
    ssim[mask] = num[mask] / den[mask]
    # estabiliza valores fora de faixa por ruído numérico
    return np.clip(ssim, 0.0, 1.0)

def _ms_ssim_map(x, y,
                 scales=(1.0, 0.5, 0.25),
                 ksizes=(7, 5, 3),
                 sigmas=(1.5, 1.0, 0.8),
                 weights=(0.5, 0.3, 0.2)):
    """
    MS-SSIM estilo “mapa” combinando escalas.
    Calcula SSIM em cada escala e faz UPSAMPLE para (H0, W0) antes de acumular.
    Retorna DSSIM = 1 - MS-SSIM em float32 [0..1] no tamanho original.
    """
    assert len(scales) == len(ksizes) == len(sigmas) == len(weights)

    if x.dtype != np.float32: x = x.astype(np.float32)
    if y.dtype != np.float32: y = y.astype(np.float32)

    H0, W0 = x.shape[:2]
    acc = np.zeros((H0, W0), dtype=np.float32)
    wsum = 0.0

    for s, ksz, sg, w in zip(scales, ksizes, sigmas, weights):
        if s == 1.0:
            xs, ys = x, y
        else:
            xs = cv2.resize(x, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            ys = cv2.resize(y, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

        ssim_s = _gaussian_ssim_map(xs, ys, ksize=ksz, sigma=sg)  # [0..1] no tamanho da escala
        ssim_up = cv2.resize(ssim_s, (W0, H0), interpolation=cv2.INTER_LINEAR)
        acc += w * ssim_up
        wsum += w

    msssim = acc / max(wsum, 1e-8)
    dssim = 1.0 - np.clip(msssim, 0.0, 1.0)
    return dssim

# ---------------------------
# Morfologia
# ---------------------------

def _apply_morphological_ops(mask, kernel_size, iterations):
    k = int(kernel_size)
    it = int(iterations)
    if mask is None:
        return None
    # permitir “sem morfologia”
    if k <= 1 or it <= 0:
        return mask
    k_eff = k if (k % 2 == 1) else (k + 1)
    kernel = np.ones((k_eff, k_eff), np.uint8)
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=it)
    m = cv2.morphologyEx(m,    cv2.MORPH_CLOSE, kernel, iterations=it)
    return m

# ---------------------------
# Detect Defects (+ Simple mode)
# ---------------------------

def detect_defects(tpl, aligned, mask,
                   dark_threshold, bright_threshold,
                   dark_morph_kernel_size, dark_morph_iterations,
                   bright_morph_kernel_size, bright_morph_iterations,
                   min_defect_area,
                   dark_gradient_threshold,
                   blue_threshold, red_threshold,
                   # ---- MS-SSIM ----
                   use_ms_ssim=True,
                   msssim_percentile=99.5,
                   msssim_weight=0.5,
                   msssim_kernel_sizes=(7,5,3),
                   msssim_sigmas=(1.5,1.0,0.8),
                   msssim_morph_kernel_size=3,
                   msssim_morph_iterations=1,
                   # ---- Overexposure handling ----
                   ignore_overexposed=False,
                   # ---- NOVO: Morfologia L (top/black) ----
                   use_morph_maps=True,
                   th_top_percentile=99.5,
                   th_black_percentile=99.5,
                   se_top=9,
                   se_black=9,
                   # ---- NOVO: Δa/Δb (cor) ----
                   use_color_delta=True,
                   color_metric="maxab",  # "maxab" ou "l2ab"
                   color_percentile=99.0,
                   # ---- NOVO: Fusão final ----
                   fusion_mode="or",      # "or" ou "weighted"
                   w_struct=0.50,
                   w_top=0.25,
                   w_black=0.15,
                   w_color=0.10,
                   fused_percentile=99.5,
                   # ---- retornos opcionais ----
                   return_msssim=False,
                   return_fusion=False):

    """
    Detecta defeitos comparando template vs imagem alinhada.
    Retorna (compatível):
        final_defect_mask, filtered_contours,
        darker_mask_filtered, brighter_mask, blue_mask, red_mask,
        [opcional msssim_mask], [opcional fused_mask]
    """
    start_time = time.perf_counter()

    # --- garantir máscara binária 0/255 ---
    if mask.dtype != np.uint8:
        mask_bin = (mask > 0).astype(np.uint8) * 255
    else:
        mask_bin = mask
    safe_roi = cv2.erode(mask_bin, np.ones((5,5), np.uint8), 1)  # afasta borda

    # --- Grayscale base (sem CLAHE) + desfoque leve ---
    t_gray = cv2.cvtColor(tpl,     cv2.COLOR_BGR2GRAY)
    a_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    t_blur = cv2.GaussianBlur(t_gray, (5, 5), 0)
    a_blur = cv2.GaussianBlur(a_gray, (5, 5), 0)

    # --- LAB (para cor) ---
    tpl_lab     = cv2.cvtColor(tpl,     cv2.COLOR_BGR2LAB)
    aligned_lab = cv2.cvtColor(aligned, cv2.COLOR_BGR2LAB)

    # =========================
    #        SIMPLE MODE
    # =========================
    if str(mode).lower() == "simple":
        # 1) DSSIM(L)
        dssim = _ms_ssim_map(
            t_blur, a_blur,
            scales=(1.0, 0.5, 0.25),
            ksizes=msssim_kernel_sizes,
            sigmas=msssim_sigmas,
            weights=(0.5, 0.3, 0.2)
        )  # [0..1]

        # 2) Residual passa-banda (DoG no |tpl - img|)
        resid = cv2.absdiff(t_blur, a_blur).astype(np.float32)
        resid01 = _norm01(resid)
        k = int(SIMPLE_BAND_KSIZE)
        if k < 3: k = 3
        if k % 2 == 0: k += 1
        r1 = cv2.GaussianBlur(resid01, (k, k), SIMPLE_BAND_SIGMA1)
        r2 = cv2.GaussianBlur(resid01, (k, k), SIMPLE_BAND_SIGMA2)
        rband = cv2.absdiff(r1, r2)
        rband = _norm01(rband)

        # 3) Cor (Δa/Δb) normalizada
        da = cv2.absdiff(aligned_lab[:,:,1], tpl_lab[:,:,1]).astype(np.float32)
        db = cv2.absdiff(aligned_lab[:,:,2], tpl_lab[:,:,2]).astype(np.float32)
        color = np.maximum(da, db)  # mais simples e rápido
        color = _norm01(color)

        # 4) Score único
        score = SIMPLE_ALPHA * dssim + SIMPLE_BETA * color + SIMPLE_GAMMA * rband
        score = _norm01(score)

        # 5) Binarização por percentil + morfologia leve
        mask_bin_simple = _percentile_bin(score, safe_roi, SIMPLE_PERCENTILE)
        mask_bin_simple = _apply_morphological_ops(mask_bin_simple, SIMPLE_MORPH_KERNEL, SIMPLE_MORPH_ITERS)

        final_defect_mask = cv2.bitwise_and(mask_bin_simple, mask_bin_simple, mask=safe_roi)

        # --- Contornos + filtros geométricos ---
        contours, _ = cv2.findContours(final_defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = []
        min_area = max(1, int(min_defect_area))
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            peri = max(cv2.arcLength(cnt, True), 1e-6)
            circularity = 4.0 * np.pi * area / (peri * peri)
            if circularity < 0.02 and area > 150:
                continue
            filtered_contours.append(cnt)

        print(f"[Simple] detect_defects took {time.perf_counter() - start_time:.4f} seconds")

        # preencher retornos “antigos” como zeros (compat)
        zeros = np.zeros_like(safe_roi, dtype=np.uint8)
        ret = [final_defect_mask, filtered_contours, zeros, zeros, zeros, zeros]
        # (opcional) devolver score binário/contínuo para debug: usa return_fusion/return_msssim se quiseres
        if return_msssim:
            # threshold por percentil no dssim para manter semântico
            msssim_mask = _percentile_bin(dssim, safe_roi, max(98.5, min(99.9, SIMPLE_PERCENTILE)))
            ret.append(msssim_mask)
        if return_fusion:
            ret.append(mask_bin_simple)
        return tuple(ret)

    # =========================
    #          FULL MODE
    # =========================

    # --- Overexposed mask (optional) ---
    over_mask = None
    if ignore_overexposed:
        hsv = cv2.cvtColor(aligned, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        _, over_mask = cv2.threshold(v, 250, 255, cv2.THRESH_BINARY)
        over_mask = cv2.bitwise_and(over_mask, safe_roi)
        over_mask = cv2.dilate(over_mask, np.ones((3, 3), np.uint8), iterations=1)

    # --- Edges finas (para cores) ---
    edges_tpl = cv2.Canny(t_blur,  60, 180)
    edges_aln = cv2.Canny(a_blur,  60, 180)
    edge_mask_thin = cv2.bitwise_or(edges_tpl, edges_aln)
    edge_mask_thin = cv2.erode(edge_mask_thin, np.ones((3,3), np.uint8), 1)
    edge_mask_inv_thin = cv2.bitwise_not(edge_mask_thin)

    # --- Darker: diff em grayscale liso (tpl - aligned) ---
    diff_dark_raw = cv2.subtract(t_blur, a_blur)         # ponto preto => positivo
    _, darker_mask = cv2.threshold(diff_dark_raw, int(dark_threshold), 255, cv2.THRESH_BINARY)

    # Gate de gradiente (em grayscale liso)
    morph_grad = cv2.morphologyEx(a_blur, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8))
    if dark_gradient_threshold <= 0:
        gradient_mask_dark = mask_bin.copy()
    else:
        _, gradient_mask_dark = cv2.threshold(morph_grad, int(dark_gradient_threshold), 255, cv2.THRESH_BINARY)

    darker_mask_filtered = cv2.bitwise_and(darker_mask, gradient_mask_dark)

    # --- Booster para micro-pontos escuros (blackhat) ---
    bh_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))  # testa 5/7/9
    bh_tpl = cv2.morphologyEx(t_blur, cv2.MORPH_BLACKHAT, bh_kernel)
    bh_cur = cv2.morphologyEx(a_blur, cv2.MORPH_BLACKHAT, bh_kernel)
    bh_diff = cv2.subtract(bh_cur, bh_tpl)
    bh_th = max(6, min(20, int(dark_threshold)//2 + 6))  # auto-ajuste simples
    _, micro_dark = cv2.threshold(bh_diff, bh_th, 255, cv2.THRESH_BINARY)
    micro_dark = cv2.bitwise_and(micro_dark, mask_bin)
    darker_mask_filtered = cv2.bitwise_or(darker_mask_filtered, micro_dark)

    # --- LAB para cores (com supressão leve de arestas finas) ---
    diff_bright_yellow_raw = cv2.subtract(aligned_lab[:, :, 2], tpl_lab[:, :, 2])  # +amarelo
    _, brighter_mask = cv2.threshold(diff_bright_yellow_raw, int(bright_threshold), 255, cv2.THRESH_BINARY)
    brighter_mask = cv2.bitwise_and(brighter_mask, edge_mask_inv_thin)

    diff_blue_raw = cv2.subtract(tpl_lab[:, :, 2], aligned_lab[:, :, 2])           # +azul
    _, blue_mask = cv2.threshold(diff_blue_raw, int(blue_threshold), 255, cv2.THRESH_BINARY)
    blue_mask = cv2.bitwise_and(blue_mask, edge_mask_inv_thin)

    diff_red_raw = cv2.subtract(aligned_lab[:, :, 1], tpl_lab[:, :, 1])            # +vermelho
    _, red_mask = cv2.threshold(diff_red_raw, int(red_threshold), 255, cv2.THRESH_BINARY)
    red_mask = cv2.bitwise_and(red_mask, edge_mask_inv_thin)

    # --- MS-SSIM (opcional) no canal L ---
    msssim_mask = None
    if use_ms_ssim:
        dssim = _ms_ssim_map(
            t_blur, a_blur,
            scales=(1.0, 0.5, 0.25),
            ksizes=msssim_kernel_sizes,
            sigmas=msssim_sigmas,
            weights=(0.5, 0.3, 0.2)
        )  # float32 [0..1]

        roi_vals = dssim[safe_roi.astype(bool)]
        thr = np.percentile(roi_vals, float(msssim_percentile)) if roi_vals.size > 0 else 1.0
        msssim_mask = (dssim >= thr).astype(np.uint8) * 255
        msssim_mask = cv2.bitwise_and(msssim_mask, safe_roi)
        msssim_mask = _apply_morphological_ops(msssim_mask, msssim_morph_kernel_size, msssim_morph_iterations)

    # --- Morfologia (permitindo k<=1 ou it=0 = sem morfologia) ---
    darker_clean   = _apply_morphological_ops(darker_mask_filtered, dark_morph_kernel_size,   dark_morph_iterations)
    brighter_clean = _apply_morphological_ops(brighter_mask,         bright_morph_kernel_size, bright_morph_iterations)
    blue_clean     = _apply_morphological_ops(blue_mask,             bright_morph_kernel_size, bright_morph_iterations)
    red_clean      = _apply_morphological_ops(red_mask,              bright_morph_kernel_size, bright_morph_iterations)

    # --- Combinar mapas antigos (retro-compat) ---
    combined = cv2.bitwise_or(darker_clean, brighter_clean)
    combined = cv2.bitwise_or(combined,     blue_clean)
    combined = cv2.bitwise_or(combined,     red_clean)

    if use_ms_ssim and msssim_mask is not None and msssim_weight > 0:
        combined = cv2.bitwise_or(combined, msssim_mask)

    # ---- NOVO: Top-hat / Black-hat em L ----
    top_score = np.zeros_like(a_blur, dtype=np.float32)
    black_score = np.zeros_like(a_blur, dtype=np.float32)
    top_bin = np.zeros_like(safe_roi, dtype=np.uint8)
    black_bin = np.zeros_like(safe_roi, dtype=np.uint8)

    if use_morph_maps:
        se_top_eff = int(se_top)
        se_black_eff = int(se_black)
        if se_top_eff < 1: se_top_eff = 1
        if se_black_eff < 1: se_black_eff = 1
        if se_top_eff % 2 == 0: se_top_eff += 1
        if se_black_eff % 2 == 0: se_black_eff += 1
        k_top = cv2.getStructuringElement(cv2.MORPH_RECT, (se_top_eff, se_top_eff))
        k_blk = cv2.getStructuringElement(cv2.MORPH_RECT, (se_black_eff, se_black_eff))

        th_tpl = cv2.morphologyEx(t_blur, cv2.MORPH_TOPHAT, k_top)
        th_cur = cv2.morphologyEx(a_blur, cv2.MORPH_TOPHAT, k_top)
        th_diff = cv2.subtract(th_cur, th_tpl)
        top_score = _norm01(th_diff)
        top_bin = _percentile_bin(top_score, safe_roi, th_top_percentile)

        bh_tpl2 = cv2.morphologyEx(t_blur, cv2.MORPH_BLACKHAT, k_blk)
        bh_cur2 = cv2.morphologyEx(a_blur, cv2.MORPH_BLACKHAT, k_blk)
        bh_diff2 = cv2.subtract(bh_cur2, bh_tpl2)
        black_score = _norm01(bh_diff2)
        black_bin = _percentile_bin(black_score, safe_roi, th_black_percentile)

    # ---- NOVO: Δa/Δb (cor) ----
    color_score = np.zeros_like(a_blur, dtype=np.float32)
    color_bin = np.zeros_like(safe_roi, dtype=np.uint8)

    if use_color_delta:
        da = np.abs(aligned_lab[:, :, 1].astype(np.float32) - tpl_lab[:, :, 1].astype(np.float32))
        db = np.abs(aligned_lab[:, :, 2].astype(np.float32) - tpl_lab[:, :, 2].astype(np.float32))
        if str(color_metric).lower() == "l2ab":
            color_score = np.sqrt(da*da + db*db)
        else:
            color_score = np.maximum(da, db)
        color_score = _norm01(color_score)
        color_bin = _percentile_bin(color_score, safe_roi, color_percentile)

    # ---- Fusão final (novo) ----
    struct_score = dssim if (use_ms_ssim and 'dssim' in locals()) else np.zeros_like(a_blur, dtype=np.float32)

    if str(fusion_mode).lower() == "weighted":
        fused_score = (w_struct * struct_score +
                       w_top    * top_score +
                       w_black  * black_score +
                       w_color  * color_score)
        fused_score = _norm01(fused_score)
        fused_mask = _percentile_bin(fused_score, safe_roi, fused_percentile)
    else:
        fused_mask = np.zeros_like(safe_roi, dtype=np.uint8)
        if use_morph_maps:
            fused_mask = cv2.bitwise_or(fused_mask, top_bin)
            fused_mask = cv2.bitwise_or(fused_mask, black_bin)
        if use_color_delta:
            fused_mask = cv2.bitwise_or(fused_mask, color_bin)
        if use_ms_ssim and msssim_mask is not None and msssim_weight > 0:
            fused_mask = cv2.bitwise_or(fused_mask, msssim_mask)

    combined_fused = cv2.bitwise_or(combined, fused_mask)
    final_defect_mask = cv2.bitwise_and(combined_fused, combined_fused, mask=safe_roi)
    if ignore_overexposed and over_mask is not None:
        final_defect_mask = cv2.bitwise_and(final_defect_mask, cv2.bitwise_not(over_mask))

    # --- Contornos + filtros geométricos ---
    contours, _ = cv2.findContours(final_defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = []
    min_area = max(1, int(min_defect_area))
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        peri = max(cv2.arcLength(cnt, True), 1e-6)
        circularity = 4.0 * np.pi * area / (peri * peri)
        if circularity < 0.02 and area > 150:
            continue
        filtered_contours.append(cnt)

    print(f"[Full] detect_defects took {time.perf_counter() - start_time:.4f} seconds")
    ret = [final_defect_mask, filtered_contours,
           darker_mask_filtered, brighter_mask, blue_mask, red_mask]
    if return_msssim:
        ret.append(msssim_mask)
    if return_fusion:
        ret.append(fused_mask)
    return tuple(ret)
