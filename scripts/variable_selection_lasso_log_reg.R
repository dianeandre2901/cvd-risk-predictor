# 📈 Variable Selection and Logistic Regression Pipeline (R version)

This R script performs stability selection using the `sharp` package to identify stable predictors of CVD. Selected variables are then used in a logistic regression model with SMOTE upsampling. Evaluation includes ROC curve, AUC, confusion matrix, and optimal threshold tuning for F1-score.

## 📁 `scripts/stability_lasso_logistic.R`

```r
# Load required libraries
library(sharp)
library(glmnet)
library(dplyr)
library(ggplot2)
library(pheatmap)
library(RColorBrewer)
library(future)
library(smotefamily)
library(pROC)

plan(multisession, workers = 4)

# Load imputed training dataset
train_data <- readRDS("data/train_imputed.rds")

# Drop unwanted columns and NA rows
train_data <- train_data %>%
  select(-eid, -sex, -date_diagnosis, -height_cm, -age_at_recruitment,
         -menopause_status, -valid_hrt_timing, -uk_biobank_assessment_centre) %>%
  filter(!is.na(cvd_binary)) %>%
  mutate(cvd_binary = as.factor(cvd_binary)) %>%
  mutate(across(where(is.factor), droplevels)) %>%
  filter(complete.cases(.))

# Create model matrix and outcome
X <- model.matrix(~ . - cvd_binary - 1, data = train_data)
Y <- train_data$cvd_binary

# Run SHARP stability selection
set.seed(2025)
sel_out <- VariableSelection(
  xdata = X,
  ydata = Y,
  family = "binomial",
  penalty.factor = rep(1, ncol(X)),
  n_cat = 3,
  pi_list = seq(0.6, 0.9, by = 0.05),
  verbose = FALSE
)

# Selection proportions and threshold
selprop <- SelectionProportions(sel_out)
hat <- Argmax(sel_out)

# Selected variable names
selected_vars <- names(selprop)[selprop >= hat[2]]

# Plot stability selection results
sel_df <- data.frame(variable = names(selprop), selprop = selprop) %>%
  filter(selprop > 0) %>%
  arrange(desc(selprop)) %>%
  mutate(variable = factor(variable, levels = variable))

ggplot(sel_df, aes(x = variable, y = selprop)) +
  geom_col(aes(fill = selprop >= hat[2])) +
  geom_hline(yintercept = hat[2], linetype = "dashed", color = "red") +
  scale_fill_manual(values = c("grey", "firebrick")) +
  labs(title = "Stability Selection", x = "Variable", y = "Selection Proportion") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, size = 8))

# Filter to selected predictors
X_sel <- X[, selected_vars]

# Apply SMOTE to balance training data
df_train <- as.data.frame(X_sel)
df_train$cvd_binary <- Y
smote_out <- SMOTE(X = df_train[, -ncol(df_train)], target = df_train$cvd_binary)
balanced <- smote_out$data
balanced$class <- as.factor(balanced$class)

# Fit logistic regression model
model <- glm(class ~ ., data = balanced, family = "binomial")

# Predict probabilities
probs <- predict(model, newdata = as.data.frame(X_sel), type = "response")

# Evaluate model with ROC and AUC
roc_obj <- roc(Y, probs)
auc_val <- auc(roc_obj)
plot(roc_obj, col = "blue", main = paste("ROC Curve (AUC =", round(auc_val, 3), ")"))

# Confusion matrix at 0.29 threshold
class_pred <- ifelse(probs > 0.29, 1, 0)
actual <- as.numeric(as.character(Y))
conf_mat <- table(Predicted = class_pred, Actual = actual)

# Calculate metrics
TN <- conf_mat[1,1]
FP <- conf_mat[2,1]
FN <- conf_mat[1,2]
TP <- conf_mat[2,2]
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)
f1 <- 2 * precision * recall / (precision + recall)
accuracy <- (TP + TN) / sum(conf_mat)
fpr <- FP / (FP + TN)

# Print
cat(sprintf("Precision: %.3f\nRecall: %.3f\nF1: %.3f\nAccuracy: %.3f\nFPR: %.3f\n",
            precision, recall, f1, accuracy, fpr))

# Threshold tuning for F1
thresholds <- seq(0, 1, 0.01)
f1_scores <- sapply(thresholds, function(t) {
  pred <- ifelse(probs > t, 1, 0)
  cm <- table(factor(pred, levels = c(0,1)), factor(actual, levels = c(0,1)))
  if (all(dim(cm) == c(2,2))) {
    TP <- cm[2,2]
    FP <- cm[2,1]
    FN <- cm[1,2]
    p <- TP / (TP + FP + 1e-6)
    r <- TP / (TP + FN + 1e-6)
    return(2 * p * r / (p + r + 1e-6))
  } else NA
})

best_thresh <- thresholds[which.max(f1_scores)]
cat(sprintf("\nBest F1 threshold: %.2f (F1 = %.3f)\n", best_thresh, max(f1_scores, na.rm = TRUE)))
```

Let me know if you'd like to generate a clean `results/metrics.csv` writer from this script too!
