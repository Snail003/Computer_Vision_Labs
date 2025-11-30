import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt


def blur_frame(img, k=5, s=3, show=False):
    out = cv2.GaussianBlur(img, (k, k), s)
    if show:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.title("Original")
        plt.imshow(img)
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.title(f"GaussianBlur k={k}, σ={s}")
        plt.imshow(out)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
    return out
def quantize_palette(img_rgb, k=4, attempts=10, show=False):
    flat = img_rgb.reshape((-1, 3)).astype(np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        20,
        1.0,
    )
    _, labels, centers = cv2.kmeans(
        flat,
        k,
        None,
        criteria,
        attempts,
        cv2.KMEANS_RANDOM_CENTERS,
    )
    centers_u8 = np.uint8(centers)
    quant = centers_u8[labels.flatten()]
    quant_img = quant.reshape(img_rgb.shape)
    if show:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.title("Original")
        plt.imshow(img_rgb)
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.title(f"KMeans (k={k})")
        plt.imshow(quant_img)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
    return quant_img
def edge_mask_canny(src, t1=10, t2=20, aperture=3, use_l2=True, blur_kernel=3, ignore_rgb=(0, 0, 0), ignore_tol=30,
                    ignore_dilate=10, show=False):
    if len(src.shape) == 3:
        gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    else:
        gray = src.copy()

    blur_gray = cv2.GaussianBlur(gray, (3, 3), 1.0)

    edges = cv2.Canny(
        blur_gray,
        threshold1=t1,
        threshold2=t2,
        apertureSize=aperture,
        L2gradient=use_l2,
    )

    if blur_kernel and blur_kernel > 1:
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        fused = cv2.GaussianBlur(edges, (blur_kernel, blur_kernel), 0)
        _, edges = cv2.threshold(
            fused, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    if ignore_rgb is not None:
        if len(src.shape) == 3:
            img_i16 = src.astype(np.int16)
            tgt = np.array(ignore_rgb, dtype=np.int16).reshape(1, 1, 3)
            diff = img_i16 - tgt
            dist = np.linalg.norm(diff, axis=2)
            mask_col = dist < ignore_tol
        else:
            col = np.uint8([[ignore_rgb[:3]]])
            gray_tgt = cv2.cvtColor(col, cv2.COLOR_RGB2GRAY)[0, 0]
            dist = np.abs(gray.astype(np.int16) - int(gray_tgt))
            mask_col = dist < ignore_tol

        if ignore_dilate and ignore_dilate > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (ignore_dilate, ignore_dilate),
            )
            mask_col = cv2.dilate(
                mask_col.astype(np.uint8), kernel
            ).astype(bool)

        edges[mask_col] = 0

    if show:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title("Source (gray)")
        plt.imshow(gray, cmap="gray")
        plt.axis("off")
        plt.subplot(1, 3, 2)
        plt.title("Edges")
        plt.imshow(edges, cmap="gray")
        plt.axis("off")
        plt.subplot(1, 3, 3)
        plt.title("Edges (ignored color removed)")
        plt.imshow(edges, cmap="gray")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return edges
def extract_green_mask(src_rgb, pal_rgb, green_ref=(32, 92, 24), primary_tol=80.0, excluded_color=(10, 20, 30),
                       excluded_tol=15.0, show=False):
    def is_excluded(col_f):
        if excluded_color is None:
            return False
        ex = np.array(excluded_color, dtype=np.float32)
        d = np.linalg.norm(col_f - ex)
        return d <= excluded_tol

    if src_rgb.shape[:2] != pal_rgb.shape[:2]:
        pal_rgb = cv2.resize(
            pal_rgb,
            (src_rgb.shape[1], src_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    flat = pal_rgb.reshape(-1, 3)
    uniq = np.unique(flat, axis=0)

    brightness = (
            0.299 * uniq[:, 0] + 0.587 * uniq[:, 1] + 0.114 * uniq[:, 2]
    )
    order = np.argsort(brightness)

    target = np.array(green_ref, dtype=np.float32)

    primary = None

    for idx in order:
        col = uniq[idx].astype(np.float32)
        if is_excluded(col):
            continue
        d = np.linalg.norm(col - target)
        if d <= primary_tol:
            primary = uniq[idx]
            break

    if primary is None:
        for idx in order:
            col = uniq[idx].astype(np.float32)
            if is_excluded(col):
                continue
            primary = uniq[idx]
            break
        if primary is None:
            primary = uniq[order[0]]

    mask_green = np.all(
        pal_rgb == primary.reshape(1, 1, 3), axis=2
    ).astype(np.uint8) * 255

    if show:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title("Source RGB")
        plt.imshow(src_rgb)
        plt.axis("off")
        plt.subplot(1, 3, 2)
        plt.title("KMeans Palette")
        plt.imshow(pal_rgb)
        plt.axis("off")
        plt.subplot(1, 3, 3)
        plt.title("Green-like mask")
        plt.imshow(mask_green, cmap="gray")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return mask_green
def mask_to_black(img_rgb, mask_bin, show=False):
    if mask_bin.ndim == 3:
        m = cv2.cvtColor(mask_bin, cv2.COLOR_RGB2GRAY)
    else:
        m = mask_bin.copy()

    if img_rgb.shape[:2] != m.shape[:2]:
        m = cv2.resize(
            m,
            (img_rgb.shape[1], img_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    out = img_rgb.copy()
    out[m == 255] = (0, 0, 0)

    if show:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.title("Original")
        plt.imshow(img_rgb)
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.title("Masked (black-out)")
        plt.imshow(out)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return out
def merge_palette(pal_rgb, thr=20.0, show=False):
    flat = pal_rgb.reshape(-1, 3)
    uniq = np.unique(flat, axis=0)

    if len(uniq) <= 1:
        return pal_rgb.copy()

    centers = uniq.copy()

    while True:
        cf = centers.astype(np.float32)
        diff = cf[:, None, :] - cf[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, 1e9)
        midx = np.argmin(dist)
        i, j = divmod(midx, dist.shape[1])
        dmin = dist[i, j]
        if dmin > thr:
            break
        new_c = ((cf[i] + cf[j]) / 2.0).astype(np.uint8)
        keep = np.ones(len(centers), dtype=bool)
        keep[i] = False
        keep[j] = False
        centers = np.concatenate(
            [centers[keep], new_c.reshape(1, 3)], axis=0
        )

    h, w, _ = pal_rgb.shape
    cf = centers.astype(np.float32)
    pix = flat.astype(np.float32)
    diff_all = pix[:, None, :] - cf[None, :, :]
    dist_all = np.sum(diff_all ** 2, axis=2)
    idx_near = np.argmin(dist_all, axis=1)
    merged_flat = cf[idx_near].astype(np.uint8)
    merged_img = merged_flat.reshape(h, w, 3)

    if show:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.title("Original palette image")
        plt.imshow(pal_rgb)
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.title("Fused palette image")
        plt.imshow(merged_img)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return merged_img
def filter_regions(src_rgb, pal_rgb, excluded_colors, excl_tol=40.0, min_area=100, max_holes=0.2, min_fill=0.4,
                   max_fill=1.0, show=False):
    if src_rgb.shape[:2] != pal_rgb.shape[:2]:
        pal_rgb = cv2.resize(
            pal_rgb,
            (src_rgb.shape[1], src_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    if isinstance(excluded_colors, (tuple, list)) and excluded_colors:
        if not isinstance(excluded_colors[0], (tuple, list, np.ndarray)):
            excluded_colors = [excluded_colors]

    if not excluded_colors:
        excl_arr = None
    else:
        excl_arr = np.array(excluded_colors, dtype=np.float32).reshape(-1, 3)

    h, w, _ = pal_rgb.shape
    flat = pal_rgb.reshape(-1, 3)
    uniq = np.unique(flat, axis=0)

    mask_out = np.zeros((h, w), dtype=np.uint8)

    for col in uniq:
        cluster_mask = np.all(
            pal_rgb == col.reshape(1, 1, 3), axis=2
        ).astype(np.uint8) * 255

        if not np.any(cluster_mask):
            continue

        n_labels, labels = cv2.connectedComponents(cluster_mask)

        for lbl in range(1, n_labels):
            region = (labels == lbl).astype(np.uint8) * 255
            area = int(np.count_nonzero(region))
            if area < min_area:
                continue

            if excl_arr is not None:
                cf = col.astype(np.float32)
                dists = np.linalg.norm(excl_arr - cf, axis=1)
                if float(np.min(dists)) <= excl_tol:
                    continue

            contours, _ = cv2.findContours(
                region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue

            filled = np.zeros_like(region)
            cv2.drawContours(filled, contours, -1, 255, thickness=-1)

            filled_area = int(np.count_nonzero(filled))
            if filled_area == 0:
                continue

            holes_area = filled_area - area
            hole_ratio = holes_area / float(filled_area)

            largest = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(largest)
            w_rect, h_rect = rect[1]
            rect_area = float(w_rect * h_rect) if w_rect > 0 and h_rect > 0 else 0.0
            if rect_area <= 0:
                rect_fill = 0.0
            else:
                rect_fill = area / rect_area

            if (
                    hole_ratio <= max_holes
                    and min_fill <= rect_fill <= max_fill
            ):
                mask_out = cv2.bitwise_or(mask_out, region)

    if show:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.title("Palette RGB")
        plt.imshow(pal_rgb)
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.title("Selected regions (mask)")
        plt.imshow(mask_out, cmap="gray")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return mask_out
def compose_overlay(base_rgb, masks, colors, labels=None, alpha=0.25, out_dir=".", filename="", font_path=None,
                    font_size=18, show=False):
    h, w = base_rgb.shape[:2]
    blended = base_rgb.copy().astype(np.float32)

    prep_masks = []
    for m in masks:
        cur = m[-1] if isinstance(m, tuple) else m
        if cur.ndim == 3:
            g = cv2.cvtColor(cur, cv2.COLOR_RGB2GRAY)
        else:
            g = cur.copy()
        if g.shape[:2] != (h, w):
            g = cv2.resize(g, (w, h), interpolation=cv2.INTER_NEAREST)
        prep_masks.append(g)

    for i, g in enumerate(prep_masks):
        col = np.array(colors[i], dtype=np.float32)
        mb = g > 0
        blended[mb] = (1.0 - alpha) * blended[mb] + alpha * col

    blended_u8 = np.clip(blended, 0, 255).astype(np.uint8)
    final_img = blended_u8

    if labels is not None:
        rows = max(1, len(labels))
        row_h = 30
        legend_h = rows * row_h + 10
        legend = np.ones((legend_h, w, 3), dtype=np.uint8) * 255

        for i, _ in enumerate(labels):
            y = (i + 1) * row_h
            x0, x1 = 10, 40
            y0, y1 = y - 15, y + 5
            col = tuple(int(c) for c in colors[i])
            cv2.rectangle(
                legend,
                (x0, max(5, y0)),
                (x1, min(legend_h - 5, y1)),
                col,
                thickness=-1,
            )

        legend_pil = Image.fromarray(legend)
        draw = ImageDraw.Draw(legend_pil)

        if font_path is None:
            candidates = [
                "C:/Windows/Fonts/arial.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            ]
            for p in candidates:
                if os.path.exists(p):
                    font_path = p
                    break

        if font_path is not None and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()

        for i, label in enumerate(labels):
            y = (i + 1) * row_h
            draw.text((50, y - 10), str(label), font=font, fill=(0, 0, 0))

        legend = np.array(legend_pil)
        final_img = np.vstack([blended_u8, legend])

    if filename:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, filename)
        Image.fromarray(final_img).save(path)

    if show:
        plt.figure(figsize=(8, 6))
        plt.title("Final overlay with legend")
        plt.imshow(final_img)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return final_img
def mask_gray_and_red(img, tol=35, show=False, red_delta=40, red_min=60):
    if img.ndim == 3:
        tmp = img.astype(np.int16)
        r, g, b = tmp[:, :, 0], tmp[:, :, 1], tmp[:, :, 2]
        mx = np.maximum(np.maximum(r, g), b)
        mn = np.minimum(np.minimum(r, g), b)
        spread = mx - mn
        not_black = (r != 0) | (g != 0) | (b != 0)
        gray_like = (spread <= tol) & not_black
        gb_max = np.maximum(g, b)
        red_like = ((r - gb_max) >= red_delta) & (r >= red_min) & not_black
        mask_bool = gray_like | red_like
        mask = (mask_bool.astype(np.uint8) * 255)
    else:
        g = img if img.ndim == 2 else img[:, :, 0]
        mask = (g > 0).astype(np.uint8) * 255

    if show:
        if img.ndim == 3:
            tmp = img.astype(np.int16)
            r, g, b = tmp[:, :, 0], tmp[:, :, 1], tmp[:, :, 2]
            mx = np.maximum(np.maximum(r, g), b)
            mn = np.minimum(np.minimum(r, g), b)
            spread = mx - mn
            not_black = (r != 0) | (g != 0) | (b != 0)
            gray_like = (spread <= tol) & not_black
            gb_max = np.maximum(g, b)
            red_like = ((r - gb_max) >= red_delta) & (r >= red_min) & not_black
            gray_vis = (gray_like.astype(np.uint8) * 255)
        else:
            gray_vis = mask.copy()

        plt.figure(figsize=(15, 4))
        plt.subplot(1, 3, 1)
        plt.title("Input image")
        if img.ndim == 3:
            plt.imshow(img)
        else:
            plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.subplot(1, 3, 2)
        plt.title("Gray-only mask")
        plt.imshow(gray_vis, cmap="gray")
        plt.axis("off")
        plt.subplot(1, 3, 3)
        plt.title("Gray + Red mask")
        plt.imshow(mask, cmap="gray")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return mask

def main():
    n_images = 10
    for i in range(n_images):
        bgr = cv2.imread(f"{i}.png")
        if bgr is None:
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        green_blur = blur_frame(rgb, k=9, s=9, show=False)
        green_kmeans = quantize_palette(green_blur, k=4, attempts=9, show=False)
        green_mask = extract_green_mask(
            rgb,
            green_kmeans,
            show=False,
        )
        rgb_no_green = mask_to_black(rgb, green_mask, show=False)

        land_no_green = mask_to_black(rgb_no_green, green_mask, show=False)
        edges = edge_mask_canny(
            land_no_green,
            t1=45,
            t2=60,
            blur_kernel=25,
            ignore_rgb=(0, 0, 0),
            show=False,
        )
        edges_inv = cv2.bitwise_not(edges)
        buildings_rgb = mask_to_black(land_no_green, edges_inv, show=False)
        buildings_mask = mask_gray_and_red(buildings_rgb, tol=50, show=False)

        compose_overlay(
            rgb,
            [green_mask, buildings_mask],
            [(0, 255, 0), (0, 0, 0)],
            labels=["Рослинність", "Будівлі"],
            alpha=0.75,
            out_dir=".",
            filename=f"{i}_masked.png",
            show=False,
        )

if __name__ == "__main__":
    main()