# SDSS Galaxy Classification: Starburst vs. Starforming

A benchmark study of machine learning models for separating **starburst** from **starforming** galaxies in the SDSS dataset, with leak-free pipelines, cross-validation, and ablations. It includes a small Flask app that classifies a galaxy from photometric measurements alone.

## Question
How well can standard ML models separate starburst from starforming galaxies, how should the class imbalance be handled, and which features does the model depend on?

## Data
- 100,000 SDSS galaxies, 43 columns (`sdss_100k_galaxy_form_burst.csv`)
- Label: `subclass`. STARFORMING = 74,993 (75%), STARBURST = 25,007 (25%)
- Removed: `objid`, `specobjid` (IDs), `ra`, `dec` (sky position), `class` (GALAXY in every row). That leaves 37 features
- The value -9999 (SDSS's "not measured" marker) was converted to missing. The most affected columns are `petroR50_u` (1,570 values) and `petroR50_z` (923). All other columns have 154 or fewer
- Split: 80/20, stratified, `random_state=42` (80,000 train / 20,000 test)

## Method
- All preprocessing (median imputation, scaling, SMOTE) runs inside pipelines, so it learns from training data only and nothing leaks from the test set
- Metrics: accuracy, macro F1, and starburst precision, recall, and F1. Starburst recall matters most because the class is rare
- Model selection used 5-fold stratified cross-validation on the training set. The test set was scored once per final model
- XGBoost settings: 300 trees, learning rate 0.1, max depth 6. No hyperparameter tuning was done

## Results

### 1. Baselines vs. models (single 80/20 split, test set, all 37 features)
| Model | Accuracy | Macro F1 | Starburst precision | Starburst recall | Starburst F1 |
|---|---|---|---|---|---|
| Always predict majority class | 0.750 | 0.429 | 0.000 | 0.000 | 0.000 |
| Logistic Regression (scaled) | 0.867 | 0.808 | 0.795 | 0.629 | 0.702 |
| Random Forest (200 trees) | 0.893 | 0.852 | 0.822 | 0.732 | 0.775 |

### 2. Handling class imbalance (Random Forest, single split, test set)
| Setup | Accuracy | Macro F1 | Starburst precision | Starburst recall | Starburst F1 |
|---|---|---|---|---|---|
| No handling | 0.893 | 0.852 | 0.822 | 0.732 | 0.775 |
| Class weights | 0.890 | 0.845 | 0.834 | 0.701 | 0.762 |
| SMOTE (training data only) | 0.882 | 0.849 | 0.732 | 0.830 | 0.778 |

SMOTE raised recall (0.73 to 0.83) but lowered precision (0.82 to 0.73) and did not clearly improve F1. The F1 differences are within what one split can produce by chance.

### 3. Model comparison (5-fold CV on training data, all 37 features, mean ± std)
| Model | Macro F1 | Starburst precision | Starburst recall | Starburst F1 |
|---|---|---|---|---|
| Logistic Regression | 0.816 ± 0.009 | 0.805 ± 0.009 | 0.643 ± 0.019 | 0.715 ± 0.015 |
| Random Forest | 0.852 ± 0.002 | 0.820 ± 0.003 | 0.734 ± 0.005 | 0.774 ± 0.003 |
| Hist. Gradient Boosting | 0.862 ± 0.002 | 0.820 ± 0.005 | 0.762 ± 0.005 | 0.790 ± 0.003 |
| XGBoost | 0.864 ± 0.003 | 0.821 ± 0.004 | 0.768 ± 0.007 | 0.794 ± 0.004 |

Gradient boosting beat Random Forest by more than the fold-to-fold spread. XGBoost and Hist. Gradient Boosting are statistically tied.

### 4. Final models on the held-out test set (XGBoost)
| Model | Features | Accuracy | Macro F1 | Starburst precision | Starburst recall | Starburst F1 |
|---|---|---|---|---|---|---|
| Full | all 37 (includes `redshift`, `redshift_err`) | 0.902 | 0.866 | 0.825 | 0.770 | 0.797 |
| Photometry-only (used in the app) | 10 photometric features | 0.875 | 0.827 | 0.773 | 0.704 | 0.737 |

### 5. Ablation: remove one feature group (full model, XGBoost, 5-fold CV)
| Removed | Starburst F1 | Change |
|---|---|---|
| Nothing (all features) | 0.794 | n/a |
| `redshift` | 0.774 | -0.020 |
| `redshift_err` | 0.757 | -0.036 |
| Magnitudes (u, g, r, i, z) | 0.794 | +0.001 |
| `psfMag_*` | 0.788 | -0.006 |
| `modelFlux_*` | 0.793 | 0.000 |
| `petroFlux_*` | 0.794 | 0.000 |
| Size (`petroRad_*`, `petroR50_*`) | 0.792 | -0.001 |
| `expAB_*` (shape) | 0.793 | -0.001 |

### 6. Do spectrum-derived features matter? (XGBoost, 5-fold CV)
| Features used | Starburst F1 | Starburst recall |
|---|---|---|
| All features | 0.794 | 0.768 |
| Without `redshift_err` | 0.757 | 0.724 |
| Photometry only, all 35 (no `redshift`, no `redshift_err`) | 0.749 | 0.718 |
| Photometry only, compact 10 | 0.744 | 0.710 |

The compact 10-feature set is: u, g, r, i, z, petroR50_r, petroR50_g, petroRad_r, petroRad_g, expAB_r. It loses about 0.005 F1 against the 35-feature photometry-only set, which is about the size of the fold-to-fold spread.

## Findings
- Every model beats the majority-class baseline, which gets 75% accuracy without finding a single starburst galaxy.
- Gradient boosting (XGBoost) is the best model. The full model reaches starburst F1 0.797 on held-out data.
- `redshift_err` is the most important single feature. Without any spectrum-derived feature, the model keeps most of its performance (F1 0.749 in cross-validation), so it can work on galaxies that have no spectrum.
- Magnitude, flux, and size features are highly correlated, so removing one group changes little because the others cover for it. Feature-importance rankings should be read with this in mind.

## Limitations
- One train/test split and one random seed. No hyperparameter tuning
- Labels come from spectra, so features derived from the spectrum (such as `redshift_err`) may reflect how the labels were assigned. This was not tested
- SMOTE and class-weight results (Table 2) come from a single split and are not confirmed by cross-validation
- The photometry-only model finds about 70% of true starbursts and is right about 77% of the time when it predicts one
- No external dataset was used

## Web app
`app/` contains a Flask app that uses the photometry-only XGBoost model. You enter 10 photometric measurements and it returns the predicted class and the model's estimated probability of "starburst".

To run it locally:
pip install -r requirements.txt
cd app
python app.py

Then open http://127.0.0.1:5000

## Repository
- `notebooks/sdss_experiments.ipynb`: all experiments in this README
- `notebooks/galaxy_classification_old.ipynb`: the first version of the project. It had issues (feature selection before the split, SMOTE not used in training, an ID column as a feature), which the new notebook fixes
- `app/`: Flask app (`app.py`), model (`galaxy_model.json`), feature list (`model_features.json`), page (`templates/index.html`)
- `sdss_100k_galaxy_form_burst.csv`: the data. The first line is a note, so load it with `pd.read_csv(..., skiprows=1)`
- The earlier demo video (Google Drive) shows the first version of the app, not this one
