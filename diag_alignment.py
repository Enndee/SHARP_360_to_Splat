"""Diagnostic script to test per-Gaussian DA360 depth alignment."""
import sys, os, torch, numpy as np
os.chdir(r"D:\Gaussian_Splatting\SHARP_360_to_Splat")
sys.path.insert(0, r"D:\Gaussian_Splatting\SHARP_360_to_Splat\ml-sharp\src")
import insp_to_splat as ist
from sharp.cli.predict import predict_image
from sharp.utils.gaussians import apply_transform

config = ist.load_config(ist.DEFAULT_CONFIG_PATH)
device = ist.resolve_device("cuda")
print("device:", device)

pano = ist.load_input_panorama(ist.ROOT_DIR / "Example files" / "IMG_20250920_085416_00_030.jpg")
pw, ph = ist.validate_equirectangular_shape(pano)
print(f"panorama: {pw} x {ph}")

side_count = 6
face_size = ist.resolve_face_size(0, pw, side_count, config)
layout = ist.build_extraction_layout(face_size, side_count, config)
print(f"face_size: {face_size}  focal_px: {layout.focal_px:.4f}")

faces = ist.extract_perspective_views(layout, pano, face_size)

ckpt = ist.DEFAULT_DA360_CHECKPOINT_PATH
da360_pred = ist.build_da360_predictor(ckpt, device)
ref_pano_disp = ist.predict_da360_disparity_panorama(da360_pred, pano, device)
print(f"DA360 pano disparity: min={ref_pano_disp.min():.6f} max={ref_pano_disp.max():.6f} mean={ref_pano_disp.mean():.6f}")

ref_views = ist.extract_perspective_scalar_views(layout, ref_pano_disp, face_size)
del da360_pred
torch.cuda.empty_cache()

predictor = ist.build_predictor(None, device)

original_meds = []
aligned_world_ys = {}  # view_name -> world-y values for ground-level check

for view in layout.views:
    face_img = faces[view.name]
    gaussians = predict_image(predictor, face_img, layout.focal_px, device)

    mv = gaussians.mean_vectors[0].detach().cpu().numpy()
    radii = np.linalg.norm(mv, axis=1)
    orig_med = float(np.median(radii))
    original_meds.append(orig_med)

    rv = ref_views[view.name]

    # Align using new per-Gaussian function
    aligned, med_scale, count = ist.align_gaussians_to_reference(
        gaussians, rv, layout.focal_px, face_size,
    )
    amv = aligned.mean_vectors[0].detach().cpu().numpy()
    aligned_radii = np.linalg.norm(amv, axis=1)

    # Rotate to world
    rotated = apply_transform(aligned, ist.face_transform_tensor(view, device)).to("cpu")
    rmv = rotated.mean_vectors[0].numpy()

    # Ground check: world-y of points in the bottom 30% of the image
    depth_z = mv[:, 2]
    valid = depth_z > 1e-6
    py = (mv[:, 1] / np.clip(depth_z, 1e-6, None)) * layout.focal_px + (face_size / 2.0) - 0.5
    bottom_mask = valid & (py >= face_size * 0.7)
    ground_world_y = rmv[bottom_mask, 1] if bottom_mask.any() else np.array([0.0])

    aligned_world_ys[view.name] = ground_world_y

    print(f"\n--- {view.name} ---")
    print(f"  SHARP radii: min={radii.min():.4f} max={radii.max():.4f} median={orig_med:.4f}")
    print(f"  Aligned radii: min={aligned_radii.min():.6f} max={aligned_radii.max():.6f} median={np.median(aligned_radii):.6f}")
    print(f"  DA360 scale: median_scale={med_scale:.6f} count={count}")
    print(f"  World Y (ground): median={np.median(ground_world_y):.4f} p10={np.quantile(ground_world_y, 0.1):.4f} p90={np.quantile(ground_world_y, 0.9):.4f}")
    print(f"  World position range: x=[{rmv[:,0].min():.4f}, {rmv[:,0].max():.4f}] y=[{rmv[:,1].min():.4f}, {rmv[:,1].max():.4f}] z=[{rmv[:,2].min():.4f}, {rmv[:,2].max():.4f}]")

# Summary: ground consistency across views
print("\n=== Ground-level consistency (lower = better) ===")
ground_medians = []
for name, gy in aligned_world_ys.items():
    med = float(np.median(gy))
    ground_medians.append(med)
    print(f"  {name}: median ground Y = {med:.4f}")
if len(ground_medians) > 1:
    gm = np.array(ground_medians)
    print(f"  Spread (max-min): {gm.max() - gm.min():.4f}")
    print(f"  Std: {gm.std():.4f}")

# Global restore preview
original_scene_median = float(np.median(original_meds))
print(f"\nOriginal SHARP median radii: {original_meds}")
print(f"Original scene median: {original_scene_median:.4f}")

print("\nDone.")
