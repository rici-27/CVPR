import os
import argparse
import numpy as np
import numpy.typing as npt
import cv2
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from get_boxes import show_box_cv2, show_mask_cv2

""" Visualisierung aller Slices mit Boxen, Boxen in GT Datei """

sns.set(style="whitegrid")

def get_middle(start: int, end: int):
    return start + (end - start) // 2 # Ganzzahl Division

def plot_slice(
    img_slice: npt.NDArray[np.uint8],
    grid_position: int,
    n_rows: int,
    slice_idx: int
):
    h = img_slice.shape[0]
    w = img_slice.shape[1]
    aspect_ratio = h / w
    new_width = 1024
    new_height = int(new_width * aspect_ratio)
    resized_image = cv2.resize(img_slice, (new_width, new_height))

    plt.subplot(n_rows, 5, grid_position)
    plt.imshow(resized_image)
    plt.axis('off')
    if slice_idx > -1:
        plt.title(f"Slice {slice_idx}") 

def visualize_file(
    img_path: str,
    gt_path: str,
    save_path: str
):
    
    plt.close('all')
    case_name = os.path.basename(img_path)

    case_img = np.load(img_path, allow_pickle=True)
    case_gt = np.load(gt_path, allow_pickle=True)
    imgs = case_img['imgs']
    boxes = case_gt['boxes']
    box = boxes[0]
    print(box)

    länge = box["z_max"] - box["z_min"]

    gts = case_gt['gts']

    grid_position = 0
    n_rows = int(np.ceil(länge/5)+1)

    plt.figure(figsize=(6, n_rows * 1.7))

    z_indices = range(box["z_min"], box["z_max"]+1)


    # combined row
    for j, idx in enumerate(z_indices):
        box2D = [box["z_mid_x_min"], box["z_mid_y_min"], box["z_mid_x_max"], box["z_mid_y_max"]]
        img_slice = cv2.cvtColor(imgs[idx].copy(), cv2.COLOR_GRAY2RGB)
        img_slice = show_box_cv2(img_slice, box2D)
        index = sorted(np.unique(gts))[1]
        print(j)
        gt_current_class = np.where(gts[idx] == index, 1, 0)
        img_slice = show_mask_cv2(gt_current_class, img_slice)
        grid_position += 1
        plot_slice(img_slice, grid_position, n_rows, idx)


    plt.suptitle(case_name, y=0.98)   
    plt.savefig(save_path, bbox_inches = "tight", dpi=500)


pfad_gt = "/Users/ricardabuttmann/Desktop/CVPR/SegFM3D/3D_val_gt_interactive_seg/US"
pfad_im = "/Users/ricardabuttmann/Desktop/CVPR/SegFM3D/3D_val_npz/US"
pfad_ziel = "/Users/ricardabuttmann/Desktop/CVPR/Bilder/Variante/3D_val_Bilder/US"

ordner_liste = sorted([name for name in os.listdir(pfad_gt) if os.path.isdir(os.path.join(pfad_gt, name))])
#print(ordner_liste)
os.makedirs(pfad_ziel, exist_ok=True)

for ordner in ordner_liste:

    dateiliste = sorted([name for name in os.listdir(os.path.join(pfad_gt, ordner)) if name.endswith(".npz")])
    #print(dateiliste)

    dateiname = dateiliste[0]

    dateipfad_gt = os.path.join(pfad_gt, ordner, dateiname)
    #print(dateipfad_gt)

    case_gt = np.load(dateipfad_gt, allow_pickle=True)
    if "boxes" in case_gt:

        dateipfad_im = os.path.join(pfad_im, ordner, dateiname)
        #print(dateipfad_im)

        zielname = dateiname.replace(".npz", ".pdf")
        dateipfad_ziel = os.path.join(pfad_ziel, zielname)
        #print(dateipfad_ziel)

        visualize_file(dateipfad_im, dateipfad_gt, dateipfad_ziel)
    else:
        print("skip")


""" Einzelne Datei visualisieren"""

#CT_AbdomenAtlas_BDMAP_00000027
# dateipfad_gt = "/Users/ricardabuttmann/Desktop/CVPR/SegFM3D/3D_val_gt_interactive_seg/CT/CT_AbdomenAtlas_BDMAP/CT_AbdomenAtlas_BDMAP_00000027.npz"
# dateipfad_im = "/Users/ricardabuttmann/Desktop/CVPR/SegFM3D/3D_val_npz/CT/CT_AbdomenAtlas_BDMAP/CT_AbdomenAtlas_BDMAP_00000027.npz"
                                                                       
# dateipfad_ziel = "/Users/ricardabuttmann/Desktop/CVPR/Bilder/Variante/3D_val_Bilder/CT/CT_AbdomenAtlas_BDMAP_00000027.pdf"
# visualize_file(dateipfad_im, dateipfad_gt, dateipfad_ziel)



