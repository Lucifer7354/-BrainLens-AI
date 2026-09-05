# BRISC2025 Dataset Analysis — Thesis Records

## 1. Dataset Overview

- Classification dataset: 6,000 images
- Training images: 5,000
- Test images: 1,000
- Number of tumor classes: 4
- MRI sequence: T1
- MRI planes: axial, coronal, sagittal

## 2. Class Distribution

| Class | Number of Images | Percentage |
|---|---:|---:|
| glioma | 1401 | 23.35% |
| meningioma | 1635 | 27.25% |
| pituitary | 1757 | 29.28% |
| no_tumor | 1207 | 20.12% |

### Interpretation

The classification dataset contains four classes with moderate class imbalance. Pituitary tumor images constitute the largest class, while no-tumor images constitute the smallest class. The imbalance should be considered during model training and evaluation, with class-wise metrics reported alongside overall accuracy.

### Figure

Figure: BRISC2025 classification dataset class distribution. Saved as `outputs/figures/01_class_distribution.png`.

## 3. Train-Test Class Distribution

| Class | Train | Test | Total |
|---|---:|---:|---:|
| glioma | 1147 | 254 | 1401 |
| meningioma | 1329 | 306 | 1635 |
| pituitary | 1457 | 300 | 1757 |
| no_tumor | 1067 | 140 | 1207 |

### Interpretation

The BRISC2025 classification dataset contains a predefined training set of 5,000 images and a test set of 1,000 images. All four tumor classes are represented in both splits. The class proportions differ between the training and test sets, therefore model evaluation should include class-wise performance metrics in addition to overall accuracy.

Figure: Train versus test class distribution. Saved as `outputs/figures/02_train_test_class_distribution.png`.


## 4. RGB vs Grayscale Image Representation

The classification dataset contains both RGB and grayscale images. Image mode was determined directly from the image files using the Pillow library.

| Class | Grayscale | RGB | Total |
|---|---:|---:|---:|
| glioma | 1227 | 174 | 1401 |
| meningioma | 709 | 926 | 1635 |
| pituitary | 930 | 827 | 1757 |
| no_tumor | 0 | 1207 | 1207 |

### Within-Class Percentages

| Class | Grayscale (%) | RGB (%) |
|---|---:|---:|
| glioma | 87.58 | 12.42 |
| meningioma | 43.36 | 56.64 |
| pituitary | 52.93 | 47.07 |
| no_tumor | 0.00 | 100.00 |

### Interpretation

A substantial difference in image representation was observed between tumor classes. In particular, all no-tumor images in the classification dataset were stored as RGB, whereas glioma, meningioma, and pituitary classes contained both RGB and grayscale images. Because image representation is associated with the class label, directly training on the original image modes could allow a model to exploit representation-related features rather than pathology-related features. Therefore, the planned preprocessing pipeline will standardize all classification images to a common RGB representation before model training.

Figure: RGB versus grayscale distribution by tumor class. Saved as `outputs/figures/03_rgb_grayscale_by_class.png`.


## 5. RGB vs Grayscale Representation by Dataset Split

Image representation was further examined separately within the predefined training and test splits.

| Class | Split | Grayscale | RGB | Total |
|---|---|---:|---:|---:|
| glioma | train | 1038 | 109 | 1147 |
| glioma | test | 189 | 65 | 254 |
| meningioma | train | 549 | 780 | 1329 |
| meningioma | test | 160 | 146 | 306 |
| pituitary | train | 697 | 760 | 1457 |
| pituitary | test | 233 | 67 | 300 |
| no_tumor | train | 0 | 1067 | 1067 |
| no_tumor | test | 0 | 140 | 140 |

### Interpretation

The analysis demonstrates that the difference in image representation is present in both the training and test splits. Notably, all no-tumor images are represented as RGB in both splits. Consequently, the original image representation could potentially provide a non-pathological statistical cue associated with the target class. To reduce this potential shortcut, all classification images will be converted to a common RGB representation during preprocessing. This standardization will be applied consistently to training, validation, and test images.

Figure: RGB versus grayscale distribution by class and dataset split. Saved as `outputs/figures/04_rgb_grayscale_by_class_and_split.png`.


## 6. MRI Plane Distribution

The classification dataset contains images from three MRI anatomical planes: axial, coronal, and sagittal.

| Class | Axial | Coronal | Sagittal | Total |
|---|---:|---:|---:|---:|
| glioma | 479 | 511 | 411 | 1401 |
| meningioma | 560 | 512 | 563 | 1635 |
| pituitary | 550 | 600 | 607 | 1757 |
| no_tumor | 404 | 358 | 445 | 1207 |

### Overall Plane Distribution

| MRI Plane | Number of Images | Percentage |
|---|---:|---:|
| axial | 1993 | 33.22% |
| coronal | 1981 | 33.02% |
| sagittal | 2026 | 33.77% |

### Interpretation

The classification dataset contains axial, coronal, and sagittal MRI images. The distribution of anatomical planes is relatively balanced across the four tumor classes, with no single plane being exclusive to a particular class. Therefore, all three anatomical planes will be retained during model development rather than excluding images based on plane orientation.

Figure: MRI plane distribution by tumor class. Saved as `outputs/figures/05_mri_plane_distribution_by_class.png`.


## 7. Image Dimension Analysis

The classification dataset contains 183 unique width × height combinations across 6,000 images.

| Dimension Category | Number of Images | Percentage |
|---|---:|---:|
| 512 × 512 | 4881 | 81.35% |
| Other dimensions | 1119 | 18.65% |

### Top 20 Dimensions

| Rank | Width | Height | Count |
|---:|---:|---:|---:|
| 1 | 512 | 512 | 4881 |
| 2 | 369 | 369 | 363 |
| 3 | 216 | 369 | 354 |
| 4 | 256 | 256 | 32 |
| 5 | 187 | 369 | 21 |
| 6 | 225 | 225 | 20 |
| 7 | 202 | 369 | 19 |
| 8 | 210 | 369 | 12 |
| 9 | 236 | 236 | 11 |
| 10 | 442 | 442 | 10 |
| 11 | 201 | 251 | 5 |
| 12 | 554 | 554 | 5 |
| 13 | 455 | 500 | 5 |
| 14 | 341 | 395 | 4 |
| 15 | 200 | 223 | 4 |
| 16 | 212 | 237 | 4 |
| 17 | 210 | 240 | 4 |
| 18 | 213 | 237 | 3 |
| 19 | 289 | 354 | 2 |
| 20 | 441 | 427 | 2 |

### Interpretation

Substantial variation in image dimensions was observed in the classification dataset. Although 4,881 images (81.35%) have dimensions of 512 × 512 pixels, the remaining images exhibit multiple different width and height combinations. Therefore, a fixed input size will be required for CNN-based classification. The preprocessing pipeline will resize all images to a common model input resolution while preserving the anatomical image content as much as possible.

Figure: Ten most common image dimensions. Saved as `outputs/figures/06_top_image_dimensions.png`.


## 8. Aspect Ratio and Square Image Analysis

The classification dataset contains 5341 square images (89.02%) and 659 non-square images (10.98%).

| Image Geometry | Number of Images | Percentage |
|---|---:|---:|
| Square | 5341 | 89.02% |
| Non-square | 659 | 10.98% |

### Aspect Ratio Statistics

| Statistic | Value |
|---|---:|
| count | 6000.0000 |
| mean | 0.9693 |
| std | 0.1102 |
| min | 0.5068 |
| 25% | 1.0000 |
| 50% | 1.0000 |
| 75% | 1.0000 |
| max | 1.6103 |

### Interpretation

The classification dataset contains both square and non-square MRI images, with variation in aspect ratio. This finding is important for CNN preprocessing because direct resizing to a square input resolution may distort the anatomical structures in non-square images. The final preprocessing pipeline will therefore consider an approach that standardizes the model input dimensions while minimizing geometric distortion, such as aspect-ratio preserving resizing with appropriate padding where required.

Figure: Distribution of image aspect ratios. Saved as `outputs/figures/07_aspect_ratio_distribution.png`.


## 9. Pixel Intensity Analysis

Pixel intensity characteristics were examined using a reproducible random sample of up to 100 classification images per tumor class (random seed = 42). Images were converted to grayscale for calculation of intensity statistics so that RGB and grayscale source images could be compared on the same intensity scale.

The following statistics were calculated for each sampled image: mean intensity, standard deviation of intensity, minimum intensity, maximum intensity, median intensity, and percentage of non-zero pixels.

The detailed per-image statistics are stored in `outputs/reports/pixel_intensity_sample_statistics.csv`.

Figures generated:

- `outputs/figures/08_mean_pixel_intensity_by_class.png`
- `outputs/figures/09_pixel_intensity_variability_by_class.png`


## 10. Exact Duplicate and Cross-Split Leakage Analysis

SHA-256 analysis identified 46 duplicate hash groups among the 6,000 classification images. A total of 96 image records were associated with duplicate hashes.

Of these, 7 exact duplicate groups were found to occur across both the predefined training and test splits. These groups involved 9 training records and 7 test records.

The cross-split duplicates therefore involve 0.70% of the 1,000-image test set.

Label consistency analysis identified 0 cross-split duplicate groups with conflicting tumor labels.

No conflicting labels were identified among the exact cross-split duplicate groups.

Because exact image duplicates are present across the predefined train/test boundary, evaluation on the original test set may contain a small degree of leakage. This will be explicitly documented in the thesis. During model development, duplicate-aware evaluation will be considered so that reported performance is not interpreted without accounting for the identified cross-split duplicates.

The complete list of cross-split duplicate images is stored in `outputs/reports/cross_split_duplicate_images.csv`.

Figure: Exact cross-split duplicate image records by tumor class and split. Saved as `outputs/figures/10_cross_split_duplicate_analysis.png`.


## 11. Identifier, Filename and Index Structure Analysis

The BRISC2025 classification manifest contains 6000 classification records. Filename parsing was successfully performed for 6000 records (100.00%).

The filename structure contains dataset split, image index, tumor code, MRI plane code and T1 sequence information. These metadata elements were compared with the corresponding manifest fields.

Filename split information matched the manifest split for 6000/6000 records. Filename tumor codes matched manifest tumor codes for 6000/6000 records. Filename plane codes matched manifest plane codes for 6000/6000 records, while filename sequence information matched the manifest sequence for 0/6000 records.

The manifest contains 5000 unique classification indices. The index ranges from 1 to 5000.

2000 classification records had duplicated manifest indices. 2000 records had duplicated parsed filename indices.

Filename indices matched manifest indices for 6000/6000 records where filename indices could be parsed.

The analysis confirms that filename metadata contains explicit class and acquisition-related information. Therefore, filenames and metadata fields will NOT be provided as model input features. The classification model will receive image pixels only.

This restriction is important to prevent metadata leakage, where the model could potentially learn the tumor label from the filename or encoded tumor code rather than from MRI image features.

Detailed identifier-level results are stored in `outputs/reports/identifier_filename_analysis.csv` and the filename-index distribution figure is stored as `outputs/figures/11_filename_index_distribution.png`.


## 12. Final Dataset Integrity Verification

The classification component contained 6000 records in the manifest. All expected classification files were checked against their manifest paths.

Files found: 6000; files missing: 0.

Folder labels were compared with manifest tumor labels. 6000 of 6000 records matched, with 0 mismatches.

Folder split information was compared with manifest split information. 6000 of 6000 records matched, with 0 mismatches.

Actual image dimensions were compared with the dimensions recorded in the manifest. 0 dimension mismatches were detected and 0 image read errors were detected.

The classification dataset contains 6000 images across four tumor classes. The predefined split contains 5000 training images and 1000 test images.

The classification manifest contains 96 records belonging to 46 duplicate SHA-256 groups. 7 duplicate SHA-256 groups cross the predefined train/test boundary.

The original BRISC2025 dataset will remain unchanged. Metadata fields will be used for quality control and statistical analysis only and will not be supplied as predictive features to the image classification models.

Overall dataset integrity verification status: PASS. The dataset is ready for the next stage of analysis subject to the documented duplicate and metadata considerations.
