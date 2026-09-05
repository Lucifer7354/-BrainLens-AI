# BRISC2025 Dataset — Master Dataset Analysis

## 1. Dataset Overview

- Total classification images: **6000**
- Training images: **5000**
- Test images: **1000**
- Number of tumor classes: **4**
- Unique image dimensions: **183**

## 2. Class Distribution

| tumor_label   |   image_count |   percentage |
|:--------------|--------------:|-------------:|
| glioma        |          1401 |        23.35 |
| meningioma    |          1635 |        27.25 |
| no_tumor      |          1207 |        20.12 |
| pituitary     |          1757 |        29.28 |

## 3. Train/Test Distribution

| tumor_label   |   test |   train |
|:--------------|-------:|--------:|
| glioma        |    254 |    1147 |
| meningioma    |    306 |    1329 |
| no_tumor      |    140 |    1067 |
| pituitary     |    300 |    1457 |

## 4. Image Mode Distribution

| tumor_label   |    L |   RGB |
|:--------------|-----:|------:|
| glioma        | 1227 |   174 |
| meningioma    |  709 |   926 |
| no_tumor      |    0 |  1207 |
| pituitary     |  930 |   827 |

## 5. MRI Plane Distribution

| tumor_label   |   axial |   coronal |   sagittal |
|:--------------|--------:|----------:|-----------:|
| glioma        |     479 |       511 |        411 |
| meningioma    |     560 |       512 |        563 |
| no_tumor      |     404 |       358 |        445 |
| pituitary     |     550 |       600 |        607 |

## 6. Image Dimensions

- Unique width × height combinations: **183**
- Square images: **5341 (89.02%)**
- Non-square images: **659 (10.98%)**
- Mean width: **472.57 px**
- Mean height: **483.88 px**
- Minimum width: **174 px**
- Maximum width: **1365 px**
- Minimum height: **195 px**
- Maximum height: **1427 px**

## 7. Aspect Ratio Analysis

- Mean aspect ratio: **0.969283**
- Standard deviation: **0.110219**
- Minimum aspect ratio: **0.506775**
- Maximum aspect ratio: **1.610256**

## 8. Duplicate Analysis

- Duplicate records: **96**
- Duplicate SHA-256 groups: **46**
- Cross-split duplicate groups: **7**
- Test images involved: **7**
- Training images involved: **9**
- Percentage of test set involved: **0.70%**

Cross-split exact duplicates were identified using SHA-256 hashes. These records will be explicitly considered during model evaluation to avoid overstating generalization performance.

## 9. Metadata Quality

- Valid SHA-256 entries: **6000/6000**
- Invalid SHA-256 entries: **0**
- Missing critical metadata values: **0**

## 10. Integrity Status

**PASS** — All classification files exist, folder labels and splits match the manifest, image dimensions are consistent with the manifest, no image read errors were detected, SHA-256 entries are valid, and no critical metadata values are missing.

## 11. Important Methodological Note

The original BRISC2025 dataset will remain unchanged. Metadata are used for quality control, dataset characterization, and statistical analysis. Metadata fields will not be supplied as predictive features to the image classification models.


## 12. Class Imbalance Analysis

The BRISC2025 classification dataset contains 6000 images distributed across four tumor classes.

| class      |   count |   percentage |
|:-----------|--------:|-------------:|
| pituitary  |    1757 |        29.28 |
| meningioma |    1635 |        27.25 |
| glioma     |    1401 |        23.35 |
| no_tumor   |    1207 |        20.12 |

The largest class is **pituitary** with 1757 images, while the smallest class is **no_tumor** with 1207 images. The largest-to-smallest class ratio is **1.456**.

The predefined training and test distributions were also examined separately to ensure that class composition is documented for model development and evaluation.

| class      |   train_count |   test_count |   train_percentage |   test_percentage |
|:-----------|--------------:|-------------:|-------------------:|------------------:|
| pituitary  |          1457 |          300 |              29.14 |              30   |
| meningioma |          1329 |          306 |              26.58 |              30.6 |
| glioma     |          1147 |          254 |              22.94 |              25.4 |
| no_tumor   |          1067 |          140 |              21.34 |              14   |

Balanced training class weights were calculated as a potential strategy for mitigating class imbalance during model training. The weights will be evaluated during the model-development stage rather than being automatically applied at this stage.

Class distribution figure: `outputs/figures/12_class_distribution.png`

Training/test class distribution figure: `outputs/figures/13_train_test_class_distribution.png`


## 13. Representative MRI Image Analysis

Representative T1-weighted MRI images were selected across all four classification classes and the three available anatomical planes (axial, coronal and sagittal). One deterministic representative image was selected for each class-plane combination.

The representative-image figure is stored at `outputs/figures/14_representative_mri_images.png`.

## 14. Pixel Intensity Analysis

Pixel-intensity characteristics were examined using a reproducible sample of up to 200 classification images per tumor class (random_state = 42), giving a maximum sample size of 800 images. Intensity statistics were calculated without modifying the original dataset. RGB images were converted to grayscale only for the purpose of intensity statistics.

| class      |   images |   mean_intensity |   std_of_mean_intensity |   median_intensity |   mean_pixel_std |   mean_nonzero_percentage |
|:-----------|---------:|-----------------:|------------------------:|-------------------:|-----------------:|--------------------------:|
| glioma     |      200 |           32.433 |                   8.428 |             13.215 |           38.561 |                    78.132 |
| meningioma |      200 |           45.133 |                  15.658 |             28.41  |           48.197 |                    85.489 |
| pituitary  |      200 |           48.2   |                   7.533 |             44.59  |           41.094 |                    91.703 |
| no_tumor   |      200 |           50.426 |                  14.113 |             34.935 |           51.955 |                    74.722 |

The image-level mean-intensity distribution is visualized using a boxplot and density plot. These analyses are descriptive and are intended to identify potential class-associated intensity differences before model development.

Intensity boxplot: `outputs/figures/15_mean_intensity_by_class.png`

Intensity density plot: `outputs/figures/16_mean_intensity_density_by_class.png`


## 15. Statistical Association Analysis

Statistical analyses were performed to evaluate whether selected image characteristics were associated with tumor class. Statistical significance was considered together with effect size.

### 15.1 Image Mode and Tumor Class

A chi-square test of independence was used to assess the association between image mode and tumor class. The chi-square statistic was **2025.3500** with **3 degrees of freedom** and a p-value of **0.000000e+00**. Cramer's V was **0.5810**, indicating a **relatively strong association**.

Image mode was determined directly from the image files rather than assumed to be a manifest variable.

### 15.2 MRI Plane and Tumor Class

A chi-square test of independence was used to assess the association between MRI plane and tumor class. The chi-square statistic was **26.3084** with **6 degrees of freedom** and a p-value of **1.950362e-04**. Cramer's V was **0.0468**, indicating a **negligible association**.

### 15.3 Mean Pixel Intensity and Tumor Class

A Kruskal-Wallis H test was used to assess differences in image-level mean intensity among the four classes. The H statistic was **260.3581** with a p-value of **3.761346e-56**. The epsilon-squared effect size was **0.3233**, corresponding to a **large effect**.

The intensity analysis used the reproducible sample described in Cell 17 and did not modify the original images.

### Statistical Output Files

- `outputs/reports/image_mode_image_level_analysis.csv`
- `outputs/reports/image_mode_class_contingency_table.csv`
- `outputs/reports/image_mode_class_percentage.csv`
- `outputs/reports/mri_plane_class_contingency_table.csv`
- `outputs/reports/mri_plane_class_percentage.csv`
- `outputs/reports/intensity_statistical_summary.csv`
- `outputs/reports/statistical_association_results.csv`
- `outputs/reports/statistical_analysis_interpretation.md`

### Statistical Figures

- `outputs/figures/17_image_mode_by_class_statistical_analysis.png`
- `outputs/figures/18_mri_plane_by_class_statistical_analysis.png`


## 16. Image Dimension and Aspect-Ratio Analysis

The classification dataset contained **183 unique width × height combinations** across 6000 images. Of these, **4881 images (81.35%)** were 512 × 512 pixels, while **1119 images (18.65%)** had other dimensions.

Overall, **5341 images (89.02%)** were square and **659 images (10.98%)** were non-square.

The aspect-ratio distribution was examined because naive resizing of non-square MRI images to a fixed square CNN input can potentially introduce geometric distortion. The analysis therefore provides an empirical basis for selecting an appropriate aspect-ratio-preserving preprocessing strategy.

### Aspect-Ratio Statistics

| tumor_label   |   images |   mean |   standard_deviation |   minimum |   median |   maximum |
|:--------------|---------:|-------:|---------------------:|----------:|---------:|----------:|
| glioma        |     1401 | 1      |               0      |    1      |        1 |    1      |
| meningioma    |     1635 | 0.9922 |               0.0603 |    0.7282 |        1 |    1.6103 |
| pituitary     |     1757 | 1      |               0.0076 |    0.7984 |        1 |    1.2146 |
| no_tumor      |     1207 | 0.8578 |               0.1995 |    0.5068 |        1 |    1.1095 |

### Aspect-Ratio Deviation

| criterion                          |   count |   percentage |
|:-----------------------------------|--------:|-------------:|
| Aspect ratio within ±5% of square  |    5397 |        89.95 |
| Aspect ratio within ±10% of square |    5468 |        91.13 |
| Aspect ratio within ±20% of square |    5563 |        92.72 |
| Aspect ratio within ±30% of square |    5583 |        93.05 |
| Aspect ratio >30% away from square |     417 |         6.95 |

### Dimension Analysis Figures

- `outputs/figures/19_image_aspect_ratio_distribution.png`
- `outputs/figures/20_image_aspect_ratio_by_class.png`
- `outputs/figures/21_square_vs_non_square_by_class.png`
- `outputs/figures/22_image_width_height_distribution.png`


# 17. Final Cross-Feature Dataset Analysis

## 17.1 Image Representation

The classification dataset contains both RGB and grayscale image representations. Image mode was determined directly from the image files rather than inferred from metadata.

- Highest grayscale proportion: **glioma (87.58%)**
- Lowest grayscale proportion: **no_tumor (0.00%)**
- Highest RGB proportion: **no_tumor (100.00%)**

The previously calculated chi-square analysis demonstrated a strong association between image mode and tumor class (Cramer's V = 0.580998). Therefore, image representation is an important dataset characteristic that must be considered when interpreting machine-learning performance.

## 17.2 Image Geometry

The highest proportion of non-square images occurred in **no_tumor (36.70%)**. The lowest proportion occurred in **glioma (0.00%)**.

The dataset therefore contains heterogeneous image dimensions and aspect ratios. Subsequent image standardization should preserve anatomical proportions and avoid unnecessary geometric distortion.

## 17.3 Image Intensity

Based on the analyzed intensity sample, **no_tumor** had the highest mean image intensity, while **glioma** had the lowest mean image intensity.

Intensity differences were statistically evaluated using the Kruskal-Wallis test in the preceding analysis. The test demonstrated a statistically significant difference in mean image intensity between tumor classes (H = 260.358090, p = 3.761346e-56, epsilon-squared = 0.323314). The intensity analysis was based on 200 randomly sampled images per class and should therefore be reported as a sample-based analysis rather than a complete census of all 6,000 images.

## 17.4 Machine-Learning Implications

The dataset analysis identified substantial differences in image representation and measurable differences in image intensity across tumor classes. MRI plane distribution, in contrast, showed only a negligible association with tumor class based on Cramer's V.

These observations support the use of a standardized preprocessing pipeline before model development. Image pixels should be used as the predictive input, while filename-derived labels, tumor codes, plane codes, split identifiers, hashes and other metadata should not be provided as model features.

The dataset-level findings should also be considered when interpreting model performance because high performance could potentially reflect dataset-specific visual characteristics rather than exclusively disease-related features.

## 17.5 Final Analysis Outputs

- `FINAL_IMAGE_MODE_CLASS_COUNTS.csv`
- `FINAL_IMAGE_MODE_CLASS_PERCENTAGES.csv`
- `FINAL_CLASS_GEOMETRY_SUMMARY.csv`
- `FINAL_INTENSITY_CLASS_SUMMARY.csv`

