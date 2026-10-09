# House Pricing Analysis using R

Mini project for **23CAP317 – R Programming** (MCA, Karpagam College of Engineering).
Author: **Vishwa A**

## What it does
Predicts house sale prices from property characteristics using the Ames Housing data
(2,930 sales, Ames, Iowa, 2006-2010). The script cleans the data, visualises it, checks
multicollinearity, splits it 80/20 and compares four models:

| Model | Test RMSE ($) | Test MAE ($) | Test R² |
|---|---|---|---|
| Linear Regression (log price) | 24,947 | 17,353 | 0.8898 |
| Random Forest (ranger, 300 trees) | 25,111 | 17,007 | 0.8883 |
| GLM (Gamma, log link) | 23,579 | 16,603 | 0.9015 |

Logistic regression (price above/below the $160,000 median): accuracy 92.3 %, kappa 0.846.

## Files
- `house_price_analysis.R` – complete script (run from the project folder)
- `data/ames_housing.csv` – dataset (exported from the `modeldata` package)
- `outputs/` – plots, `console_log.txt`, `model_comparison.csv`

## How to run
```r
install.packages(c("ggplot2", "corrplot", "caret", "ranger"))
source("house_price_analysis.R")
```

## Data source
De Cock, D. (2011). *Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester
Regression Project.* Journal of Statistics Education, 19(3).
