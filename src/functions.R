generate_metrics_classification <- function(model,type,test_data)
{
  if(type == "train")
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
  
  model_pr_auc <- augment_value %>%
    mutate(.predicted_label = as.numeric(as.character(.predicted_label))) %>%
    pr_auc(estimate = .predicted_label, truth = delay) %>%
    pull(.estimate)
  
  metric_names <- c("Accuracy",
                    "Sensitivity",
                    "Specificity",
                    "Precision",
                    "Recall",
                    "roc_auc",
                    "pr_auc")
  
  model_metrics <- c(
    model_accuracy,
    model_sensitivity,
    model_specificity,
    model_precision,
    model_recall,
    model_roc_auc,
    model_pr_auc)
  
  model_metric_results <-
    data.frame(metric_names, model_metrics) %>%
    mutate(model_metrics = round(model_metrics,3))
  
  colnames(model_metric_results) <- c("metric_names", "value")
  }
  if(type == "test")
  {
    prediction <- ml_predict(model,test_data) 
    model_accuracy <- prediction %>%
      ml_metrics_multiclass(metrics = "accuracy") %>%
      pull(.estimate)
    
    model_recall <- prediction %>%
      ml_metrics_multiclass(metrics = "recall") %>%
      pull(.estimate)
    
    model_precision <- prediction %>%
      ml_metrics_multiclass(metrics = "precision") %>%
      pull(.estimate)
    
    model_f1 <- prediction %>%
      ml_metrics_multiclass(metrics = "f_meas") %>%
      pull(.estimate)
    
    model_roc_auc <- prediction %>%
      ml_metrics_binary(metrics = "roc_auc") %>%
      pull(.estimate)
    
    model_pr_auc <- prediction %>%
      ml_metrics_binary(metrics = "pr_auc") %>%
      pull(.estimate)
    
    metric_names <- c("Accuracy",
                      "Recall",
                      "precision",
                      "f_meas",
                      "roc_auc",
                      "pr_auc")
    
    model_metrics <- c(
      model_accuracy,
      model_recall,
      model_precision,
      model_f1,
      model_roc_auc,
      model_pr_auc)
    
    model_metric_results <-
      data.frame(metric_names, model_metrics) %>%
      mutate(model_metrics = round(model_metrics,3))
    
    colnames(model_metric_results) <- c("metric_names", "value")
  }
  return(model_metric_results)
}