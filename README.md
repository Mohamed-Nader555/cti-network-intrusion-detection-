# CTI Network Intrusion Detection — Dataset Build & Modeling

This repository contains two Jupyter notebooks that implement a complete pipeline for **binary intrusion detection** on network flow data:

1) **Combine_dataset_Final_Project_CTI.ipynb** — builds a consolidated dataset from multiple CSV sources.  
2) **Working_with_the_dataset_Final_Project_CTI.ipynb** — performs feature pruning, visualization, model training, and evaluation.

The project targets **binary classification** of network flows into **Benign (0)** or **Attack (1)** and demonstrates a full, reproducible pipeline from raw CSVs → combined dataset → feature engineering → modeling → evaluation.

---

## Repository Structure


├── Combine_dataset_Final_Project_CTI.ipynb
├── Working_with_the_dataset_Final_Project_CTI.ipynb
├── images/
│ ├── corr_heatmap_full.png
│ ├── corr_heatmap_full_pruned.png
│ ├── Gradient Boosting Confusion and ROC.png
│ ├── Logistic Regression Confusion and ROC.png
│ ├── Random Forest Confusion and ROC.png
│ ├── Traffic Distribution Benign vs Attack.png
│ └── ANN Confusion and ROC.png
└── README.md




---

## What’s Inside (High Level)

**Notebook 1 — Combine the CSVs**

- Loads multiple source CSVs (e.g., different DoS/DDoS subsets and benign flows).
- Concatenates them into a single dataframe.
- Saves the result as **`combined_benign_dos_dataset.csv`** (used by the second notebook).

**Notebook 2 — Modeling Pipeline**

- Loads the combined dataset and sets `y = attack_flag`.
- Drops ID/meta columns from features:  
  `flow_id, timestamp, src_ip, dst_ip, label, src_port, dst_port, protocol, attack_flag`.
- **Feature pruning**
  - Remove zero-variance features (`VarianceThreshold`).
  - Correlation analysis and pruning with an upper-triangle scan using **|ρ| > 0.95**.
  - Retains ~**63** informative features after pruning.
- **Class balance**: ~**57.5% Benign** vs **42.5% Attack**.
- **Train/test split**: 80/20 (stratified).
- **Scaling**: `StandardScaler` for models that need it.
- **Models trained and evaluated**:
  - Logistic Regression (baseline)
  - Random Forest
  - Gradient Boosting
  - ANN (Keras)
- **Evaluation**: Confusion Matrix & ROC curve for each model (saved as images).

---

## Key Visualizations

**Class Balance**  
![Traffic Distribution: Benign vs Attack](images/Traffic%20Distribution%20Benign%20vs%20Attack.png)

**Correlation Heatmaps**  
- Before pruning  
  ![Correlation overview (after variance filter)](images/corr_heatmap_full.png)
- After pruning (|ρ| > 0.95, ~63 features kept)  
  ![Correlation overview (after pruning)](images/corr_heatmap_full_pruned.png)

**Model Results**  
- Logistic Regression  
  ![Logistic Regression – Confusion & ROC](images/Logistic%20Regression%20Confusion%20and%20ROC.png)
- Random Forest  
  ![Random Forest – Confusion & ROC](images/Random%20Forest%20Confusion%20and%20ROC.png)
- Gradient Boosting  
  ![Gradient Boosting – Confusion & ROC](images/Gradient%20Boosting%20Confusion%20and%20ROC.png)
- ANN (Keras)  
  ![ANN – Confusion & ROC](images/ANN%20Confusion%20and%20ROC.png)

---

## Results at a Glance

| Model                | ROC–AUC (approx.) | Notes                                      |
|----------------------|-------------------|--------------------------------------------|
| Logistic Regression  | ~0.67             | Baseline linear classifier                 |
| Random Forest        | ~1.00             | Near-perfect separation on this dataset    |
| Gradient Boosting    | ~1.00             | Near-perfect separation on this dataset    |
| ANN (Keras)          | ~0.99             | Strong performance with early stopping     |

> **Note:** Very high AUCs for tree ensembles can indicate highly separable features or potential overfitting. For robust validation, consider time-based splits or testing on traffic captured from different days/sources.

---

## How to Run (Google Colab)

1. Open each notebook in **Google Colab**.
2. Run the first cells to **mount Google Drive** (already included).
3. **Notebook 1**: Update file paths to your CSVs if needed, then run all cells to generate `combined_benign_dos_dataset.csv`.
4. **Notebook 2**: Ensure `combined_benign_dos_dataset.csv` is accessible (path may need updating), then run all cells to:
   - prune features,
   - train/evaluate models,
   - and regenerate the plots in `images/`.

**Dependencies (Colab already includes most):**
- `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `scipy`
- `tensorflow` / `keras` (for the ANN)

---

## Project Goals

- Provide a clean, end-to-end example of **intrusion detection** on flow-level data:
  - **Data consolidation** from multiple sources
  - **Feature selection** via variance and correlation pruning
  - **Model comparison** across linear, tree-based, and neural models
  - **Clear, visual evaluation** (confusion matrices + ROC curves)
- Serve as a starting point for:
  - Trying additional feature engineering (e.g., domain-driven ratios/timers)
  - Cross-validation across capture sessions/days
  - Calibration and threshold tuning for deployment scenarios

---

## Reuse & Adaptation

- Replace the input CSVs with your own captures.
- Adjust the correlation threshold (default **0.95**) if you want more/less aggressive pruning.
- Swap or extend models (e.g., XGBoost, LightGBM, SVM) and add cross-validation.

---

## License

Add your preferred license (e.g., MIT) by creating a `LICENSE` file.

---

## Contact

Questions or suggestions? Open an issue or a pull request.

