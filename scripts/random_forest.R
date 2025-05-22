#  Random Forest Pipeline with Feature Importance 

This script trains and evaluates a Random Forest model for CVD risk prediction. It includes:
- Top feature selection via Gini importance
- SMOTE upsampling
- Full ROC, AUC, F1 tuning
- Confusion matrix visualizations
- SHAP-based explainability

```r
# Load packages
library(randomForest)
library(ggplot2)
library(dplyr)
library(caret)
library(pROC)
library(tidyr)
library(fastshap)

# Step 1: Load datasets
train <- readRDS("data/train_final.rds")
test  <- readRDS("data/test_final.rds")

# Step 2: Prepare training data
train_rf <- train %>%
  select(-c(first_hrt_prescription, last_hrt_prescription, date_diagnosis, 
            uk_biobank_assessment_centre))
train_rf$cvd_binary <- as.factor(train_rf$cvd_binary)

# Step 3: Train initial RF and select top features
set.seed(123)
rf_full <- randomForest(cvd_binary ~ ., data = train_rf, importance = TRUE, ntree = 500)
importance_df <- as.data.frame(importance(rf_full)) %>%
  mutate(Feature = rownames(.)) %>%
  arrange(desc(MeanDecreaseGini))
top_features <- importance_df$Feature[1:15]
train_rf_top <- train_rf %>% select(all_of(top_features), cvd_binary)

# Step 4: Apply upsampling
set.seed(123)
train_bal <- upSample(train_rf_top %>% select(-cvd_binary), train_rf_top$cvd_binary, yname = "cvd_binary")

# Step 5: Final RF model on selected features
rf_model <- randomForest(cvd_binary ~ ., data = train_bal, ntree = 300, importance = TRUE)

# Step 6: Evaluate on test set
col_keep <- colnames(train_bal)
test_rf <- test %>% select(any_of(col_keep)) %>% drop_na()
test_rf$cvd_binary <- factor(test_rf$cvd_binary, levels = levels(train_bal$cvd_binary))
preds <- predict(rf_model, newdata = test_rf)
probs <- predict(rf_model, newdata = test_rf, type = "prob")[,2]

# Step 7: Confusion matrix & AUC
truth <- test_rf$cvd_binary
cm <- confusionMatrix(preds, truth)
print(cm)
roc_obj <- roc(truth, probs)
plot(roc_obj, col = "#2C7BB6", lwd = 3, main = paste("ROC - AUC =", round(auc(roc_obj), 3)))

# Step 8: Threshold tuning for F1
thresholds <- seq(0, 1, by = 0.01)
f1_scores <- sapply(thresholds, function(t) {
  pred <- factor(ifelse(probs > t, 1, 0), levels = c(0, 1))
  ref  <- factor(as.numeric(as.character(truth)), levels = c(0, 1))
  cm <- table(pred, ref)
  if (all(dim(cm) == c(2,2))) {
    TP <- cm[2,2]; FP <- cm[2,1]; FN <- cm[1,2]
    p <- TP / (TP + FP + 1e-6); r <- TP / (TP + FN + 1e-6)
    return(2 * p * r / (p + r + 1e-6))
  } else NA
})
best_idx <- which.max(f1_scores)
best_thresh <- thresholds[best_idx]

# Step 9: Final predictions at best threshold
final_preds <- factor(ifelse(probs > best_thresh, 1, 0), levels = c(0,1))
final_truth <- factor(as.numeric(as.character(truth)), levels = c(0,1))
final_cm <- confusionMatrix(final_preds, final_truth, positive = "1")
print(final_cm)

# Step 10: SHAP Explanation
X_explain <- train_bal %>% select(-cvd_binary)
pred_fun <- function(model, newdata) predict(model, newdata = newdata, type = "prob")[, 2]
set.seed(123)
shap_vals <- fastshap::explain(rf_model, X_explain, pred_fun, nsim = 50)
shap_summary <- shap_vals %>% as.data.frame() %>%
  summarise(across(everything(), ~ mean(abs(.)))) %>%
  pivot_longer(cols = everything(), names_to = "Variable", values_to = "MeanAbsSHAP") %>%
  arrange(desc(MeanAbsSHAP)) %>%
  slice(1:15)

# Plot SHAP summary
shap_summary$Variable <- gsub("_", " ", tools::toTitleCase(shap_summary$Variable))
ggplot(shap_summary, aes(x = reorder(Variable, MeanAbsSHAP), y = MeanAbsSHAP)) +
  geom_col(fill = "#2C7BB6") +
  coord_flip() +
  labs(title = "Top 15 Variables - SHAP (Random Forest)", x = "Variable", y = "Mean |SHAP| Value") +
  theme_minimal(base_size = 14)
```
