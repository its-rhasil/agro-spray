import os, shutil, random
random.seed(42)

def split_class(source_dir, plant, class_name,
                val_ratio=0.15, test_ratio=0.15):
    files = [f for f in os.listdir(source_dir)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(files)

    n_val = int(len(files) * val_ratio)
    n_test = int(len(files) * test_ratio)

    splits = {
        'train': files[n_val + n_test:],
        'val':   files[:n_val],
        'test':  files[n_val:n_val + n_test]
    }

    for split_name, split_files in splits.items():
        out_dir = os.path.join("data", plant, split_name, class_name)
        os.makedirs(out_dir, exist_ok=True)
        for f in split_files:
            shutil.copy(
                os.path.join(source_dir, f),
                os.path.join(out_dir, f)
            )
        print(f"  {split_name}/{class_name}: {len(split_files)} images")


def prepare_plant(plant_name, raw_healthy_dir, raw_diseased_dir):
    print(f"\n{plant_name.upper()} - Healthy:")
    split_class(raw_healthy_dir, plant_name, "healthy")
    print(f"{plant_name.upper()} - Diseased:")
    split_class(raw_diseased_dir, plant_name, "diseased")


# ── ADD PLANTS HERE AS YOU COLLECT DATA ──────────────────────────
PLANTS = {
    "betel":    ("data/raw/Betel_Healthy",    "data/raw/Betel_Diseased"),
    # "plumeria": ("data/raw/Plumeria_Healthy", "data/raw/Plumeria_Diseased"),
    # "hibiscus": ("data/raw/Hibiscus_Healthy", "data/raw/Hibiscus_Diseased"),
}
# ─────────────────────────────────────────────────────────────────

for plant, (healthy_dir, diseased_dir) in PLANTS.items():
    prepare_plant(plant, healthy_dir, diseased_dir)

print("\nDone. Structure created:")
for plant in PLANTS:
    for split in ['train', 'val', 'test']:
        for cls in ['healthy', 'diseased']:
            path = os.path.join("data", plant, split, cls)
            count = len(os.listdir(path))
            print(f"  data/{plant}/{split}/{cls}/  →  {count} images")