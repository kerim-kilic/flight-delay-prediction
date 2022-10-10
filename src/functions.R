generate_metrics_classification <- function(model)
{
  augment_value <- augment(model) %>% collect()
  augment_value$delay <- as.factor(augment_value$delay)
  augment_value$.predicted_label <- as.factor(augment_value$.predicted_label)
  
  model_accuracy <- augment_value %>%
    accuracy(estimate = .predicted_label, truth = delay) %>%
    pull(.estimate)
  
  model_sensitivity <- augment_value %>%
    sensitivity(estimate = .predicted_label, truth = delay) %>%
    pull(.estimate)
  
  model_specificity <- augment_value %>%
    specificity(estimate = .predicted_label, truth = delay) %>%
    pull(.estimate)
  
  model_precision <- augment_value %>%
    precision(estimate = .predicted_label, truth = delay) %>%
    pull(.estimate)
  
  model_recall <- augment_value %>%
    recall(estimate = .predicted_label, truth = delay) %>%
    pull(.estimate)
  
  model_roc_auc <- augment_value %>%
    mutate(.predicted_label = as.numeric(as.character(.predicted_label))) %>%
    roc_auc(estimate = .predicted_label, truth = delay) %>%
    pull(.estimate)
  
  metric_names <- c("Accuracy",
                    "Sensitivity",
                    "Specificity",
                    "Precision",
                    "Recall",
                    "ROC_AUC")
  
  model_metrics <- c(
    model_accuracy,
    model_sensitivity,
    model_specificity,
    model_precision,
    model_recall,
    model_roc_auc)
  
  model_metric_results <-
    data.frame(metric_names, round(model_metrics,3))
  
  colnames(model_metric_results) <- c("metric_names", "value")
  
  return(model_metric_results)
}