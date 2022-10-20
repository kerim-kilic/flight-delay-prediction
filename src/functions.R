### 
### Author: Kerim Kiliç
###

### Function to generate and return metrics of machine learning model
generate_metrics_classification <- function(model,type,test_data)
{
  # Check if the data is from the test set or train set.
  if(type == "train")
  {
    predictions <- augment(model) %>%
      mutate(predicted_label = .predicted_label) %>%
      select(-.predicted_label)
    ###
    data <- augment(glm_model) %>%
      mutate(prediction = as.numeric(.predicted_label),
             label = as.numeric(delay)) %>%
      select(-.predicted_label, -delay)
    
    model_pr_auc <- ml_metrics_binary(metrics = "pr_auc",
                      x = data,
                      estimate = prediction,
                      truth = label) %>% 
      pull(.estimate)
    
    model_roc_auc <- ml_metrics_binary(metrics = "roc_auc",
                                       x = data,
                                       estimate = prediction,
                                       truth = label) %>% 
      pull(.estimate)
  }
  if(type == "test")
  {
    predictions <- ml_predict(model,test_data)
    model_roc_auc <- predictions %>%
      ml_metrics_binary(metrics = "roc_auc") %>%
      pull(.estimate)
    
    model_pr_auc <- predictions %>%
      ml_metrics_binary(metrics = "pr_auc") %>%
      pull(.estimate)
  }
  
  # Calculate the confusion matrix with TP, TN, FP, FN
  TP <- sdf_nrow(predictions %>%
                   filter(delay == "1", 
                          predicted_label == "1"))
  
  TN <- sdf_nrow(predictions %>%
                   filter(delay == "0", 
                          predicted_label == "0"))
  
  FP <- sdf_nrow(predictions %>%
                   filter(delay == "0", 
                          predicted_label == "1"))
  
  FN <- sdf_nrow(predictions %>%
                   filter(delay == "1", 
                          predicted_label == "0"))
  
  # Calculate each model metric using confusion matrix
  accuracy <- (TP + TN) / (sdf_nrow(predictions))
  recall <- (TP) / (TP + FN)
  precision <- (TP) / (TP + FP)
  missclassification_rate <- (FP + FN) / (sdf_nrow(predictions))
  specificity <- (TN) / (TN + FP)
  f_score <- (2*precision*recall) / (precision+recall)
  
  metric_names <- c("accuracy",
                    "recall",
                    "precision",
                    "missclassification_rate",
                    "specificity",
                    "f_score",
                    "roc_auc",
                    "pr_auc")

  model_metrics <- c(
    accuracy,    
    recall,    
    precision,    
    missclassification_rate,
    specificity,
    f_score,
    model_roc_auc,
    model_pr_auc)
      
  model_metric_results <-
    data.frame(metric_names, model_metrics) %>%
    mutate(model_metrics = round(model_metrics,3))

  colnames(model_metric_results) <- c("metric_names", "value")
  return(model_metric_results)
}

# Function to create the split of the data into train and test
create_train_test_split <- function(data, sample_size, ratio, type)
{
  ## Splitted datasets for classification
  # Balanced training set using under sampling
  train_size <- sample_size * ratio
  test_size <- sample_size * (1-ratio)
  
  if(type == "numerical")
  {
    ## Splitted datasets for numerical prediction
    train_data_regr <- data %>%
      sample_frac(train_size/sdf_nrow(data))
    
    test_data_regr <- data %>%
      sample_frac(test_size/sdf_nrow(data))
    
    train_test_split <- list(train_data_regr,test_data_regr)
  }
  
  if(type == "classification")
  {
    delay_yes <- data %>%
      filter(delay == "1")
    delay_no <- data %>%
      filter(delay == "0")
    
    tmp1 <- delay_yes %>% 
      sample_n(train_size/2)
    tmp2 <- delay_no %>% 
      sample_n(train_size/2)
    train_data <- rbind(tmp1, tmp2)
    
    test_data <- data %>%
      sample_frac(test_size/sdf_nrow(data))
    train_test_split <- list(train_data,test_data)
  }
  return(train_test_split)
}

cross_validator <- function(sc, 
                            data,
                            pipeline,
                            grid,
                            type,
                            folds,
                            seed)
{

  if(type == "numerical") # how to evaluate the CV
  {
    evaluator <- ml_regression_evaluator(sc,metric_name = "r2")
  }
  if(type == "classification")
  {
    evaluator <- ml_multiclass_classification_evaluator(sc,metric_name = "accuracy")
  }
  
  # Cross validation
  cv <- ml_cross_validator(
    sc,
    estimator = pipeline, # use our pipeline to estimate the model
    estimator_param_maps = grid, # use the params in grid
    evaluator = evaluator,
    num_folds = folds, # number of CV folds
    seed = seed
  )
  
  start_time <- Sys.time()
  cv_model <- ml_fit(cv, data)
  end_time <- Sys.time()
  # Measure the time it takes to cross validate glm model
  train_time <- end_time - start_time
  
  cv_results <- cv_model$avg_metrics_df
  if(type == "classification")
  {
    best_cv_result <- cv_results[which.max(cv_results$accuracy),"accuracy"]  
  }
  if(type == "numerical")
  {
    best_cv_result <- cv_results[which.max(cv_results$r2),"r2"]  
  }
   
  return(list(cv_results,best_cv_result,train_time))
}