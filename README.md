

## Overview

This project implements a multimodal pipeline for predicting product prices in e-commerce using image, text, and tabular data features. The approach combines OpenCLIP image embeddings, extensive text feature engineering (from both provided catalog information and optional OCR-extracted image text), and item pack/quantity extraction to handle the complex price prediction task. The solution is designed for competitive data science environments and optimized for the SMAPE evaluation metric.

## Pipeline Summary

- Load and preprocess training and test datasets.
- Extract and clean all text fields (title, catalog_content, brand).
- Download product images and compute OpenCLIP ViT-B/32 embeddings, using mean fallback and a no-image-success flag for robustness.
- (Optional & Recommended) Use OCR (such as EasyOCR) to extract text from product images for additional features and signal.
- Extract enhanced tabular/text features, including item pack quantity (IPQ), weight, volume, and advanced language cues.
- Concatenate all features into a unified feature matrix, scale numerical variables, and apply CV-safe target encoding to brand and category using StratifiedKFold cross-validation.
- Train four models per fold: neural network fusion (image, text, tabular), LightGBM, CatBoost, and XGBoost, all on the log1p price, using MAE-style loss to better align with SMAPE.
- Out-of-fold predictions and test-time ensembles are computed, with optimized linear ensembling for best overall SMAPE.
- Predictions are post-processed and clipped, targeting realistic price ranges to reduce over/underestimations.
- Metrics, feature importance, and model weights are reported at training and leaderboard (competition) stages.

## Key Features

- Robust image handling with OpenCLIP and mean embedding fallback
- Automated and validated IPQ/pack-size extraction
- Advanced text feature engineering, including OCR-derived features
- Stratified, leakage-free target encoding in all folds
- MAE-focused losses and SMAPE-based validation
- Weighted ensembling of deep and tree models

## Setup

### Dependencies

- Python 3.8+
- PyTorch
- lightgbm
- catboost
- xgboost
- scikit-learn
- numpy, pandas
- tqdm
- openclip-pytorch
- easyocr (optional, for OCR features)
- requests, pillow

Install dependencies using pip:

```sh
pip install torch lightgbm catboost xgboost scikit-learn numpy pandas tqdm openclip-pytorch easyocr pillow requests
```

### Data

Place `train.csv` and `test.csv` in a `dataset` directory in the root of the repository. Ensure that images referenced in the sample are accessible via their URLs in the appropriate column.

## Usage

Open and run the notebook `helper_final.ipynb` or `enhanced_multimodal_fusion*.ipynb`.

Steps:
1. Adjust environment and data paths as needed.
2. Execute the preprocessing and feature extraction cells.
3. To enable OCR integration, run the provided OCR extraction cells after image download.
4. Run the model training and validation blocks. Out-of-fold metrics and ensembling weights will be printed.
5. Predictions are output as a submission-ready CSV in the required format.

## Reproducibility

- The code is modular with clear separation between feature extraction, model training, and ensembling.
- Cross-validation is stratified and ensures no target encoding leakage.
- Random seeds are set for each fold for consistency.

## File Structure

- `helper_final.ipynb` — Main notebook, all core code and experiments
- `enhanced_multimodal_fusion_*.ipynb` — Additional experiments and improvements
- `amlch_refactored.py` — Additional utility code
- `dataset/` — Folder for csv data
- All outputs and models are stored in the notebook directory unless specified otherwise

## Advanced Usage

- For OCR integration on all images, see the relevant notebook cell and adjust `max_images` as required by your hardware.
- Extensive feature diagnostics and post-processing routines are provided for leaderboard calibration.
- Feature importance analysis for all tree models is available as a built-in routine.

## Results and Benchmarks

- This solution achieves SMAPE metrics in the top 30–50% on challenging e-commerce datasets when all advanced techniques are used.
- With OCR and robust IPQ extraction, expect a 5–15% SMAPE improvement over vanilla multimodal baselines.

## Contributing

Please open issues or pull requests for improvements, bug fixes, or suggested feature additions.

## License

This code is released under the MIT License. See [LICENSE](LICENSE) for details.

## References

- OpenCLIP: https://github.com/mlfoundations/open_clip
- EasyOCR: https://github.com/JaidedAI/EasyOCR
- CatBoost, XGBoost, LightGBM official documentation

