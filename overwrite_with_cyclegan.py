import os
import shutil
import subprocess

# === CONFIG ===
images_dir = "/home/mundus/hrashid173/ChatSim/data/waymo_multi_view/segment-11379226583756500423_6230_810_6250_810_with_camera_labels/images_all"
cycle_root = "/home/mundus/hrashid173/CycleGAN-master"
checkpoint_name = "waymo2bdd100k"
checkpoint_suffix = "_A"  # For latest_net_G_A.pth
translated_dir = os.path.join(cycle_root, "results", checkpoint_name, "test_latest", "images")

# === 1. Run CycleGAN test ===
print("[INFO] Running CycleGAN on original images...")
subprocess.run([
    "python", "test.py",
    "--dataroot", images_dir,
    "--name", checkpoint_name,
    "--model", "test",
    "--dataset_mode", "single",
    "--preprocess", "none",
    "--no_dropout",
    "--model_suffix", checkpoint_suffix,
    "--checkpoints_dir", os.path.join(cycle_root, "checkpoints"),
    "--num_test", str(len(os.listdir(images_dir)))
], cwd=cycle_root, check=True)

# === 2. Overwrite original files ===
print("[INFO] Overwriting original images with snowy versions...")
fake_images = sorted([f for f in os.listdir(translated_dir) if f.endswith("_fake.png")])
original_images = sorted([f for f in os.listdir(images_dir) if f.endswith(".jpg") or f.endswith(".png")])

if len(fake_images) != len(original_images):
    raise ValueError(f"Mismatch: {len(fake_images)} translated vs {len(original_images)} original images")

for fake, original in zip(fake_images, original_images):
    src = os.path.join(translated_dir, fake)
    dst = os.path.join(images_dir, original)
    shutil.copyfile(src, dst)

print(f"[✅] Replaced {len(fake_images)} images in {images_dir} with snowy versions.")
