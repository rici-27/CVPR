import os
import argparse
import numpy as np
import numpy.typing as npt
import cv2
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from get_boxes import show_box_cv2, show_mask_cv2

""" Variante bei der die Boxen woanders abgelegt sind """

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
    box_path: str,
    save_path: str
):
    
    plt.close('all')
    case_name = os.path.basename(img_path)

    case_img = np.load(img_path, allow_pickle=True)
    case_gt = np.load(gt_path, allow_pickle=True)
    case_box = np.load(box_path, allow_pickle=True)
    imgs = case_img['imgs']
    boxes = case_box['boxes']
    gts = case_gt['gts']
    indexmenge = sorted(np.unique(gts))[1:]

    grid_position = 0
    n_rows = len(boxes) * 3

    plt.figure(figsize=(6, n_rows * 1.5))

    for i, box in enumerate(boxes):
        backwards_mid = get_middle(box['z_min'], box['z_mid'])
        forward_mid = get_middle(box['z_mid'], box['z_max'])
        z_indices = [box['z_min'], backwards_mid, box['z_mid'], forward_mid, box['z_max']]

        index = indexmenge[i]

        # box row
        for j, idx in enumerate(z_indices):
            box2D = [box['z_mid_x_min'], box['z_mid_y_min'], box['z_mid_x_max'], box['z_mid_y_max']]
            img_slice = cv2.cvtColor(imgs[idx].copy(), cv2.COLOR_GRAY2RGB)
            img_slice = show_box_cv2(img_slice, box2D)
            grid_position += 1
            plot_slice(img_slice, grid_position, n_rows, idx)
        
        # mask row
        for j, idx in enumerate(z_indices):
            img_slice = cv2.cvtColor(imgs[idx].copy(), cv2.COLOR_GRAY2RGB)
            gt_current_class = np.where(gts[idx] == index, 1, 0)
            img_slice = show_mask_cv2(gt_current_class, img_slice)
            grid_position += 1
            plot_slice(img_slice, grid_position, n_rows, -1)

        # combined row
        for j, idx in enumerate(z_indices):
            box2D = [box["z_mid_x_min"], box["z_mid_y_min"], box["z_mid_x_max"], box["z_mid_y_max"]]
            img_slice = cv2.cvtColor(imgs[idx].copy(), cv2.COLOR_GRAY2RGB)
            img_slice = show_box_cv2(img_slice, box2D)
            gt_current_class = np.where(gts[idx] == index, 1, 0)
            img_slice = show_mask_cv2(gt_current_class, img_slice)
            grid_position += 1
            plot_slice(img_slice, grid_position, n_rows, -1)

    # Zusätzliche Informationen mit figtext hinzufügen
    number_of_boxes = len(boxes)
    number_of_slices = imgs.shape[0]
    info_text = f"Anzahl der Boxen: {number_of_boxes}, Anzahl der Slices: {number_of_slices} "

    # Größerer Wert als 0.93 für größere Anzahl an boxen besser

    plt.figtext(0.5, 0.93, info_text, ha='center', va='center', fontsize=9, color='black')


    plt.suptitle(case_name, y=0.98)   
    plt.savefig(save_path, bbox_inches = "tight", dpi=500)


#img_path = "/Users/ricardabuttmann/Desktop/CVPR/SegFM3D/3D_val_npz/CT/CT_AbdomenAtlas_BDMAP/CT_AbdomenAtlas_BDMAP_00000006.npz"
#gt_path = "/Users/ricardabuttmann/Desktop/CVPR/SegFM3D/3D_val_gt_interactive_seg/CT/CT_AbdomenAtlas_BDMAP/CT_AbdomenAtlas_BDMAP_00000006.npz"
#save_path = "/Users/ricardabuttmann/Desktop/CVPR/Bilder/3D_val_Bilder/CT/CT_AbdomenAtlas_BDMAP_00000006.pdf"


img_path = "/Users/ricardabuttmann/Desktop/CVPR/SegFM3D/3D_train_npz_random_10percent_16G/US"
gt_path = "/Users/ricardabuttmann/Desktop/CVPR/SegFM3D/3D_train_npz_random_10percent_16G/US"
box_path = "/Users/ricardabuttmann/Desktop/CVPR/SegFM3D/3D_train_boxes/US"
save_path = "/Users/ricardabuttmann/Desktop/CVPR/Bilder/3D_train_Bilder/US"


ordner_liste = sorted([name for name in os.listdir(gt_path) if os.path.isdir(os.path.join(gt_path, name))])
print(ordner_liste)
os.makedirs(save_path, exist_ok=True)

for ordner in ordner_liste:

    dateiliste = sorted([name for name in os.listdir(os.path.join(gt_path, ordner)) if name.endswith(".npz")])
    #print(dateiliste)

    dateiname = dateiliste[0]
    print(dateiname)

    dateipfad_gt = os.path.join(gt_path, ordner, dateiname)
    dateipfad_im = os.path.join(img_path, ordner, dateiname)

    zielname = dateiname.replace(".npz", ".pdf")
    dateipfad_ziel = os.path.join(save_path, zielname)
  
    dateipfad_box = os.path.join(box_path, ordner, dateiname)

    visualize_file(img_path=dateipfad_im, gt_path=dateipfad_gt, box_path=dateipfad_box, save_path=dateipfad_ziel)

