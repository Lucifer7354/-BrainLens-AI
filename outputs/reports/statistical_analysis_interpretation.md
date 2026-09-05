# BRISC2025 Statistical Association Analysis

## 1. Image Mode vs Tumor Class

- Chi-square statistic: 2025.349990
- Degrees of freedom: 3
- p-value: 0.000000e+00
- Cramer's V: 0.580998
- Effect interpretation: relatively strong association

## 2. MRI Plane vs Tumor Class

- Chi-square statistic: 26.308353
- Degrees of freedom: 6
- p-value: 1.950362e-04
- Cramer's V: 0.046823
- Effect interpretation: negligible association

## 3. Mean Pixel Intensity vs Tumor Class

- Kruskal-Wallis H statistic: 260.358090
- p-value: 3.761346e-56
- Epsilon-squared: 0.323314
- Effect interpretation: large effect

## Important methodological note

Statistical associations identified in this analysis describe properties of the BRISC2025 dataset. They should not be interpreted as evidence that these characteristics are independently diagnostic of tumor class. Image mode, intensity and anatomical plane may reflect acquisition, formatting or dataset-composition effects. These findings are therefore considered during subsequent preprocessing and model evaluation.
