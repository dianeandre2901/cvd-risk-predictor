# Load libraries
library(pROC)
library(ggplot2)
library(caret)
library(reshape2)
library(dplyr)
library(fastshap)

# Load predictions + true labels
# You should have these stored from model output
probs <- readRDS("results/rf_probs.rds")     # predicted probabilities for class 1
truth <- readRDS("results/rf_truth.rds")     # actual labels as factor(0/1)
model <- readRDS("results/rf_model_final.rds") # trained model
X <- readRDS("results/rf_X_final.rds")         # training predictors (no outcome)

# 1. ROC Curve and AUC
roc_obj <- roc(truth, probs)
auc_val <- auc(roc_obj)

png("results/roc_rf.png", width = 600, height = 500)
plot(roc_obj, col = "#2C7BB6", lwd = 3, main = paste("ROC Curve - AUC =", round(auc_val, 3)))
legend("bottomright", legend = paste("AUC =", round(auc_val, 3)), col = "#2C7BB6", lwd = 3, bty = "n")
dev.off()

# 2. Confusion Matrix (threshold = 0.5 by default or best_threshold_rf)
best_thresh <- 0.65
preds <- factor(ifelse(probs > best_thresh, 1, 0), levels = c(0,1))
conf_mat <- confusionMatrix(preds, truth, positive = "1")

# Save metrics CSV
metrics <- data.frame(
  Model = "Random Forest",
  Threshold = best_thresh,
  AUC = round(auc_val, 3),
  Accuracy = round(conf_mat$overall["Accuracy"], 3),
  Precision = round(conf_mat$byClass["Precision"], 3),
  Recall = round(conf_mat$byClass["Recall"], 3),
  F1 = round(conf_mat$byClass["F1"], 3)
)
write.csv(metrics, "results/metrics_rf.csv", row.names = FALSE)

# Plot confusion matrix
cm_df <- as.data.frame(as.table(conf_mat$table))
colnames(cm_df) <- c("Prediction", "Truth", "Freq")

png("results/confusion_matrix_rf.png", width = 600, height = 500)
ggplot(cm_df, aes(x = Truth, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 6, fontface = "bold") +
  scale_fill_gradient(low = "white", high = "#2C7BB6") +
  labs(title = "Confusion Matrix - Random Forest", x = "Actual", y = "Predicted") +
  theme_minimal(base_size = 15)
dev.off()

# 3. SHAP Summary Plot
pred_fun <- function(model, newdata) predict(model, newdata = newdata, type = "prob")[,2]

shap_vals <- fastshap::explain(
  object = model,
  X = X,
  pred_wrapper = pred_fun,
  nsim = 50
)

shap_summary <- shap_vals %>%
  as.data.frame() %>%
  summarise(across(everything(), ~ mean(abs(.)))) %>%
  pivot_longer(cols = everything(), names_to = "Variable", values_to = "MeanAbsSHAP") %>%
  arrange(desc(MeanAbsSHAP)) %>%
  slice(1:15)

png("results/shap_rf.png", width = 800, height = 600)
ggplot(shap_summary, aes(x = reorder(Variable, MeanAbsSHAP), y = MeanAbsSHAP)) +
  geom_col(fill = "#2C7BB6") +
  coord_flip() +
  labs(title = "Top 15 SHAP Variables - Random Forest", x = "Variable", y = "Mean |SHAP| Value") +
  theme_minimal(base_size = 14)
dev.off()
