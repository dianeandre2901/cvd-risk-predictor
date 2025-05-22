# 📦 Data Preprocessing Pipeline (R version)

This script contains the **full, clean, and rewritten pipeline** used to preprocess, recode, match, clean, and impute the UK Biobank dataset for a cardiovascular disease (CVD) prediction project in postmenopausal women. It is based entirely in **R** and includes variable selection, cohort filtering, matching, cleaning, collinearity checks, and Random Forest-based imputation.

## 📁 `scripts/preprocessing.R`

```r
# Load required libraries
library(dplyr)
library(lubridate)
library(MatchIt)
library(tableone)
library(caret)
library(corrplot)
library(mice)
library(ggplot2)

# Step 1: Load original raw dataset
raw_data <- read.csv("data/biobank_variables_eid.csv")

# Step 2: Remove irrelevant illness, cancer, and medication duplicates
filtered <- raw_data %>% 
  select(-starts_with("Non_cancer_illness_year.age_first_occurred"),
         -starts_with("age_non_cancer_illness_diagnosed"),
         -starts_with("Non_cancer_illness_code"),
         -starts_with("cancer_code"),
         -contains(".0.1"), -contains(".0.2"), -contains(".0.3"),
         -contains(".0.4"), -contains(".0.5"), -contains(".0.6"),
         -matches("Medication_for_.*_heartburn.*"),
         -matches("pulse_rate.*"),
         -matches("Number_of_self_reported.*"))

# Step 3: Clean column names
filtered <- filtered %>% rename_with(~ gsub("\\.0\\.0$", "", .)) %>%
  rename_with(~ gsub("\\.", "_", .))

# Step 4: Remove known irrelevant features
filtered <- filtered %>%
  select(-c(number_in_household, HES_data_records, reason_lost_to_follow_up, 
            date_lost_to_follow_up, Age_DVT_diagnosed, Age_pulmonary_embolism_diagnosed,
            Age_stroke_diagnosed, Age_.high_blood_pressure_diagnosed, Age_heart_attack_diagnosed,
            Age_angina_diagnosed, Age_started_HRT, Age_last_used_HRT, ever_used_HRT,
            Former_alcohol_drinker, pack_years_of_smoking_lifespan_proportion,
            Light_smokers, Current_tobacco_smoking, Past_tobacco_smoking,
            age_when_last_used_oral_contraceptive_pill, age_started_oral_contraceptive_pill))

# Step 5: Drop inconsistent entries (pregnancy, men < 40, missing menopause)
filtered <- filtered %>%
  filter(pregnant == 0, sex == 0, !is.na(had_menopause), age_at_recruitment >= 40)

# Step 6: Add HRT exposure definition
filtered <- filtered %>%
  filter(is.na(first_hrt_prescription) | first_hrt_prescription <= date_recr) %>%
  mutate(exposure_hrt_status = if_else(!is.na(hrt_within_5yrs) & hrt_within_5yrs, 1, 0))

# Step 7: Encode menopause age groups
filtered <- filtered %>%
  mutate(age_at_meno_cat = case_when(
    !is.na(age_at_menopause) & age_at_menopause < 40 ~ "<40",
    age_at_menopause < 45 ~ "40-44",
    age_at_menopause < 50 ~ "45-49",
    age_at_menopause < 55 ~ "50-54",
    age_at_menopause < 60 ~ "55-59",
    age_at_menopause >= 60 ~ "60+",
    TRUE ~ NA_character_
  ))

# Step 8: Define binary CVD outcome
filtered <- filtered %>%
  mutate(cvd_binary = if_else(incident_case == 1, 1, 0)) %>%
  filter(prevalent_case != 1) %>%
  select(-incident_case, -prevalent_case)

# Step 9: Merge BMI
bmi_data <- readRDS("data/bmi_data.rds") %>% select(eid, bmi = bmi_0_0)
filtered <- filtered %>% left_join(bmi_data, by = "eid")

# Step 10: Final variable adjustments for matching
filtered <- filtered %>%
  mutate(
    menopause_binary = if_else(menopause_status %in% c("<40", "40-44", "45-49", "50-54", "55-59", "60+"), "1", "0"),
    bmi_category = cut(bmi, breaks = c(-Inf, 18.5, 25, 30, Inf), labels = c("Underweight", "Normal", "Overweight", "Obese"))
  )

# Step 11: Matching (1:2 nearest-neighbour)
matched <- matchit(
  exposure_hrt_status ~ bmi + age_at_recruitment + menopause_status,
  data = filtered,
  method = "nearest",
  ratio = 2
)

matched_data <- match.data(matched)

# Step 12: Save matched dataset
saveRDS(matched_data, "data/matched_dataset.rds")

# Step 13: Split into train/test
set.seed(42)
train_idx <- createDataPartition(matched_data$cvd_binary, p = 0.8, list = FALSE)
train <- matched_data[train_idx, ]
test  <- matched_data[-train_idx, ]

# Step 14: Remove dates and backup
date_vars <- c("first_hrt_prescription", "last_hrt_prescription", "date_recr", "date_diagnosis")
train_dates <- train %>% select(eid, any_of(date_vars))
test_dates  <- test  %>% select(eid, any_of(date_vars))
train <- train %>% select(-any_of(date_vars))
test  <- test  %>% select(-any_of(date_vars))

saveRDS(train_dates, "data/train_dates.rds")
saveRDS(test_dates, "data/test_dates.rds")

# Step 15: Check and remove high-correlation vars
cor_matrix <- cor(train %>% select(where(is.numeric)), use = "pairwise.complete.obs")
high_corr_vars <- findCorrelation(cor_matrix, cutoff = 0.8)
train <- train[, -high_corr_vars]
test  <- test[, -high_corr_vars]

# Step 16: Impute training data with random forest
imputer <- mice(train, method = 'rf', m = 5, seed = 2025)
train_imputed <- complete(imputer, 1)
saveRDS(train_imputed, "data/train_imputed.rds")
saveRDS(imputer, "data/train_imputer_model.rds")

# Step 17: Align and impute test set
test <- test %>% select(all_of(names(imputer$data)))
test_imputed <- complete(mice(
  test,
  method = imputer$method,
  predictorMatrix = imputer$predictorMatrix,
  m = 1, maxit = 1, seed = 2025
), 1)

saveRDS(test_imputed, "data/test_imputed.rds")

# Step 18: Reattach date metadata
train_final <- left_join(train_imputed, train_dates, by = "eid")
test_final  <- left_join(test_imputed, test_dates, by = "eid")
saveRDS(train_final, "data/train_final.rds")
saveRDS(test_final, "data/test_final.rds")
```

---

**Ready for modeling** with LASSO + Random Forest. 
Let me know if you want the `train_model.R` script or SHAP-style interpretation plots next!
