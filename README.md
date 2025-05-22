# 🫀 Cardiovascular Disease Risk Prediction in Postmenopausal Women

This project investigates whether Hormone Replacement Therapy (HRT) history can predict cardiovascular disease (CVD) in postmenopausal women using classical machine learning models. We applied stability selection with LASSO and a Random Forest classifier to UK Biobank data and identified key biomarkers and predictors of incident CVD.

🎓 Project based on my MSc work at Imperial College London (April 2025)  
📊 Dataset: 26,688 postmenopausal women from UK Biobank with prescription + hospital data  
💡 Goal: Predict CVD risk and evaluate if HRT is a meaningful predictor

---

## 🔍 Research Question

**Does a history of HRT use allow us to predict cardiovascular disease (CVD) in women?**

---

## 📚 Methodology

- **Design**: Simulated clinical trial using UK Biobank + GP prescription data  
- **Cohort**: 26,688 women (aged 40+) with no baseline CVD  
- **Exposure**: ≥6 months of HRT use within 5 years before recruitment  
- **Outcome**: Incident CVD via hospital episodes & death records  
- **Covariates**: 150+ variables (demographics, biomarkers, lifestyle)

### 🔧 Preprocessing
- 2:1 matching on age, BMI, and menopause status  
- Random forest-based imputation (MICE)  
- Oversampling to address class imbalance (~8% CVD cases)

### 🧠 Models
- **LASSO Logistic Regression** using stability selection via `sharp`  
- **Random Forest** using full predictor set and Gini variable importance  
- Threshold tuning to optimize F1-score

---

## 📊 Results

| Model               | Accuracy | AUC   | Precision | Recall | F1 Score |
|---------------------|----------|-------|-----------|--------|----------|
| Logistic Regression | **83.3%** | **0.719** | **0.204**   | 0.356  | **0.259**  |
| Random Forest       | 78.5%    | 0.699 | 0.172     | **0.427**  | 0.245    |

- LASSO identified key predictors: age, systolic BP, cystatin C, HDL, WBC count  
- HRT and menopause status were **not** predictive after adjusting for age and BMI  
- Logistic regression was more interpretable and robust

📉 Confusion Matrices, ROC Curves, SHAP plots included in `/results`

---

## 🩺 Key Findings

- Clinical biomarkers were stronger predictors than HRT use  
- Hormonal effects may be indirectly captured via markers like HDL & Cystatin C  
- Comprehensive hormone data (estradiol/progesterone) is needed in future studies  
- Highlighted gender bias in CVD research and missing hormone data in biobanks

---

## 🔬 Public Health Relevance

This work demonstrates how ML can enhance risk stratification in women's cardiovascular health. By identifying accessible and stable biomarkers, this approach could guide future screening strategies — especially when direct hormone measures are missing.

---

## 📁 Project Structure
cvd-risk-predictor/
├── data/                 # Input data (not uploaded to GitHub)
├── src/                  # Code: training scripts, modeling
├── notebooks/            # Exploratory notebooks (EDA, SHAP)
├── results/              # Confusion matrices, ROC, metrics
├── README.md             # You are here
└── requirements.txt      # Dependencies


---

## 💻 Tools Used
- Python (Pandas, NumPy, Scikit-learn, SHAP)
- R (`sharp`, `mice`, `randomForest`)
- Matplotlib & Seaborn for plotting

---

## ✨ Future Work

- Include hormonal assay data (estradiol, progesterone)  
- Validate model on more diverse populations  
- Expand to study testosterone therapy in men

---

## 📬 Contact

Made with ❤️ by Diane  
🎀 [GitHub](https://github.com/dianeandre2901) ・ [LinkedIn](https://www.linkedin.com/in/...)  
