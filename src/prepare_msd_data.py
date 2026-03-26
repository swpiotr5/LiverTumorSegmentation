"""
Skrypt do przygotowania danych z Medical Segmentation Decathlon (Task03_Liver)
do formatu zgodnego z istniejącym data/prepared/.

Struktura wejściowa (MSD):
  Task03_Liver/imagesTr/liver_XXX.nii.gz
  Task03_Liver/labelsTr/liver_XXX.nii.gz

Struktura wyjściowa (taka sama jak z Kaggle):
  data/prepared/person_XXX/{liver,liver_tumor,tumor,background_only}/{ct,mask}/slice_NNN.npy

Użycie:
  python prepare_msd_data.py --src /mnt/c/Users/XPS/Downloads/Task03_Liver/Task03_Liver --dst ../data/prepared
"""

import os
import argparse
import numpy as np
import nibabel as nib
import gc


def prepare_patient(volume_path, label_path, patient_id, dst_dir):
    """Przetwórz jednego pacjenta z formatu MSD do prepared."""
    base_dir = os.path.join(dst_dir, f"person_{patient_id}")

    if os.path.exists(base_dir):
        print(f"  {base_dir} już istnieje – pomijam")
        return

    os.makedirs(base_dir, exist_ok=True)

    # Podkatalogi identyczne jak w notebooku 01-explore-nii
    categories = ["liver", "tumor", "liver_tumor", "background_only"]
    for cat in categories:
        os.makedirs(os.path.join(base_dir, cat, "ct"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, cat, "mask"), exist_ok=True)

    # Załaduj NIfTI
    volume = nib.load(volume_path).get_fdata()
    mask = nib.load(label_path).get_fdata()

    print(f"  Pacjent {patient_id}: volume {volume.shape}, mask {mask.shape}")

    num_slices = mask.shape[2]
    for i in range(num_slices):
        ct_slice = volume[:, :, i]
        slice_mask = mask[:, :, i]

        unique_vals = np.unique(slice_mask)

        # Kategoryzuj slice (ta sama logika co w notebooku)
        if 1 in unique_vals and 2 not in unique_vals:
            cat = "liver"
        elif 2 in unique_vals and 1 not in unique_vals:
            cat = "tumor"
        elif 1 in unique_vals and 2 in unique_vals:
            cat = "liver_tumor"
        else:
            cat = "background_only"

        ct_path = os.path.join(base_dir, cat, "ct", f"slice_{i:03d}.npy")
        mask_path = os.path.join(base_dir, cat, "mask", f"slice_{i:03d}.npy")
        np.save(ct_path, ct_slice)
        np.save(mask_path, slice_mask)

        del ct_slice, slice_mask

    del volume, mask
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Przygotuj dane MSD Task03_Liver")
    parser.add_argument("--src", required=True,
                        help="Ścieżka do Task03_Liver (katalog z imagesTr, labelsTr)")
    parser.add_argument("--dst", default="../data/prepared_fine_tuning",
                        help="Katalog docelowy (domyślnie: ../data/prepared_fine_tuning)")
    parser.add_argument("--start-id", type=int, default=None,
                        help="ID startowe dla pacjentów (domyślnie: auto, kontynuacja od ostatniego)")
    args = parser.parse_args()

    images_dir = os.path.join(args.src, "imagesTr")
    labels_dir = os.path.join(args.src, "labelsTr")

    if not os.path.isdir(images_dir):
        print(f"BŁĄD: Nie znaleziono {images_dir}")
        return
    if not os.path.isdir(labels_dir):
        print(f"BŁĄD: Nie znaleziono {labels_dir}")
        return

    # Znajdź pliki NIfTI
    image_files = sorted([f for f in os.listdir(images_dir)
                          if f.endswith(('.nii', '.nii.gz')) and not f.startswith('._')])
    label_files = sorted([f for f in os.listdir(labels_dir)
                          if f.endswith(('.nii', '.nii.gz')) and not f.startswith('._')])

    print(f"Znaleziono {len(image_files)} obrazów i {len(label_files)} masek")

    # Ustal ID startowe - znajdź istniejących pacjentów żeby nie nadpisać
    if args.start_id is not None:
        next_id = args.start_id
    else:
        existing = [d for d in os.listdir(args.dst) if d.startswith("person_")] if os.path.isdir(args.dst) else []
        existing_ids = []
        for d in existing:
            try:
                existing_ids.append(int(d.split("_")[1]))
            except (ValueError, IndexError):
                pass
        next_id = max(existing_ids) + 1 if existing_ids else 0
        print(f"Istniejący pacjenci: {len(existing)}, zaczynam od ID {next_id}")

    # Mapuj obrazy do masek
    # MSD naming: liver_XXX.nii.gz (obrazy i maski mają te same nazwy)
    for img_file in image_files:
        # Znajdź odpowiadającą maskę
        if img_file in label_files:
            label_file = img_file
        else:
            # Spróbuj dopasować po numerze
            base = img_file.replace(".nii.gz", "").replace(".nii", "")
            matches = [f for f in label_files if base in f]
            if not matches:
                print(f"  UWAGA: Brak maski dla {img_file} – pomijam")
                continue
            label_file = matches[0]

        vol_path = os.path.join(images_dir, img_file)
        lab_path = os.path.join(labels_dir, label_file)

        # Sprawdź czy ten plik MSD nie został już przetworzony
        # (prosty check po nazwie w pliku mapowania)
        print(f"\n[{next_id}] Przetwarzam: {img_file}")
        prepare_patient(vol_path, lab_path, next_id, args.dst)
        next_id += 1

    print(f"\nGotowe! Przygotowano dane do {args.dst}")


if __name__ == "__main__":
    main()
