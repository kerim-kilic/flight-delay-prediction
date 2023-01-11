### 
### Author: Kerim Kiliç
###

#####################################################################
### Function to generate and return metrics of machine learning model
### model:      Sparklyr or H2O model to obtain metrics from.
### type:       Metrics for the train or test data.
###             For Sparklyr or H2O models
### test_data:  Supply test data for metrics on test set
### sc:         Spark connection
### Returns the metrics on the train or test data.
#####################################################################
generate_metrics_classification <- function(model,type,test_data, sc)
{
  # Check if the data is from the test set or train set.
  if(type == "train")
  {
    predictions <- augment(model) %>%
      mutate(predicted_label = .predicted_label) %>%
      select(-.predicted_label)
    ###
    data <- augment(model) %>%
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
  if(type == "h2o_classification_test")
  {
    predictions <- h2o.predict(model, test_data)
    predictions <- as.data.frame(predictions)
    actual_delays <- data.frame(as.data.frame(test_data) %>% pull(delay))
    colnames(actual_delays) <- c("actual_delay")
    
    evaluation_frame <- predictions %>%
      append(actual_delays)
  
    predictions <- evaluation_frame
    predictions <- data.frame(predictions) %>%
      select(predict,actual_delay)
    
    colnames(predictions) <- c("predicted_label", "delay")
    
    predictions <- copy_to(dest = sc,
                           df = predictions,
                           overwrite = TRUE)
    
    model_roc_auc <- h2o.auc(h2o.performance(model, newdata = test_data))
    model_pr_auc <- h2o.aucpr(h2o.performance(model, newdata = test_data))
  }
  if(type == "h2o_classification_train")
  {
    model_metrics1 <- model@model$cross_validation_metrics_summary[c(1,18,16),]
    model_metrics2 <- model@model$cross_validation_metrics_summary[c(20,6,2,15),]
    ##
    value  <- c(1 - model@model$cross_validation_metrics_summary[1,1])
    metric_names <- c("misclassification")
    miss_class <- data.frame(metric_names,value)
    ##
    metric_names <- c("accuracy",
                       "recall", 
                       "precision")
    metric_names <- data.frame(metric_names)
    
    model_metrics1 <- model_metrics1 %>% append(metric_names)
    model_metrics1 <- data.frame(model_metrics1)
    
    model_metrics1 <- model_metrics1 %>%
      select(metric_names, mean) %>%
      mutate(mean = round(mean,3))
    colnames(model_metrics1) <- c("metric_names", "value")
    ##
    metric_names <- c("specificity",
                      "f1", 
                      "roc_auc",
                      "pr_auc")
    metric_names <- data.frame(metric_names)
    
    model_metrics2 <- model_metrics2 %>% append(metric_names)
    model_metrics2 <- data.frame(model_metrics2)
    
    model_metrics2 <- model_metrics2 %>%
      select(metric_names, mean) %>%
      mutate(mean = round(mean,3))
    colnames(model_metrics2) <- c("metric_names", "value")
    
    model_metrics <- rbind(model_metrics1,miss_class,model_metrics2) %>%
      mutate(value = round(value,3))
    
    return(model_metrics)
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
                    "missclassification",
                    "specificity",
                    "f1",
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

#####################################################################
### Function to create train-test and train-validation-test split
### data:       Input data to perform split on.
### ratio:      Ratio to put into train data.
###             Function derives test and validation ratio itself.
### type:       "spark" for Sparklyr split
###             "h2o" for H2O split
### hc:         H2O connection
### Returns the data split.
#####################################################################
create_train_test_split <- function(data, ratio, type, hc)
{
  if(type == "spark")
  {
    positive_samples <- data %>% filter(delay == "1")
    negative_samples <- data %>% filter(delay == "0")
    
    positive_samples_split <- sdf_random_split(x = positive_samples,
                                               train_data = ratio,
                                               test_data = 1-ratio,
                                               seed = 1234)
    negative_samples_split <- sdf_random_split(x = negative_samples,
                                               train_data = ratio,
                                               test_data = 1-ratio,
                                               seed = 1234)
    negative_samp_t <- negative_samples_split$train_data
    negative_samples_undersampled <- negative_samp_t %>%
      sample_n(sdf_nrow(positive_samples_split$train_data))
    
    train_data <- rbind(positive_samples_split$train_data,
                        negative_samples_undersampled)
    
    test_data <- rbind(positive_samples_split$test_data,
                       negative_samples_split$test_data)
    
    train_test_split <- list(train_data = train_data,
                             test_data = test_data)
  }
  if(type == "h2o")
  {
    positive_samples <- data %>% filter(delay == "1")
    negative_samples <- data %>% filter(delay == "0")
    
    positive_samples_split <- sdf_random_split(x = positive_samples,
                                               train_data = ratio,
                                               test_data = (1-ratio)/2,
                                               valid_data = (1-ratio)/2,
                                               seed = 1234)
    negative_samples_split <- sdf_random_split(x = negative_samples,
                                               train_data = ratio,
                                               test_data = (1-ratio)/2,
                                               valid_data = (1-ratio)/2,
                                               seed = 1234)
    negative_samp_t <- negative_samples_split$train_data
    negative_samples_undersampled <- negative_samp_t %>%
      sample_n(sdf_nrow(positive_samples_split$train_data))
    
    train_data <- rbind(positive_samples_split$train_data,
                        negative_samples_undersampled)
    
    valid_data <- rbind(positive_samples_split$valid_data,
                        negative_samples_split$valid_data)
    
    test_data <- rbind(positive_samples_split$test_data,
                       negative_samples_split$test_data)
    
    
    train_test_split <- list(train_data = train_data,
                             valid_data = valid_data,
                             test_data = test_data)    
  }
  return(train_test_split)
}

#####################################################################
### Function to cross-validate Sparklyr models
### sc:         Sparklyr connection.
### data:       Training data used in cross-validation
### pipeline:   Pipeline object to use in cross-validation.
### grid:       Grid object to use in cross-validation.
### type:       "classification" for classification models
### folds:      Number of folds to perform cross-validation with.
### seed:       Seed to create repeatability.
### Returns cross-validation results.
#####################################################################
cross_validator <- function(sc, 
                            data,
                            pipeline,
                            grid,
                            type,
                            folds,
                            seed)
{

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
  # Measure the time it takes to cross validate model
  train_time <- end_time - start_time
  
  cv_results <- cv_model$avg_metrics_df
  if(type == "classification")
  {
    best_cv_result <- cv_results[which.max(cv_results$accuracy),]
  }
   
  final_cv_results <- list(all_results = cv_results,
                           best_result = best_cv_result,
                           train_time = train_time)
  class(final_cv_results) <- "final_cv_results"
  
  return(final_cv_results)
}

#####################################################################
### Function to process and prepare weather data.
### Function saves processed data locally.
### sc:         Sparklyr connection.
#####################################################################
get_weather_data <- function(sc)
{
  ### Get the correct data for each airport
  ATL_data_raw <- read.csv("./data/30320_ATL.csv") %>%
    mutate(origin = "ATL")
  DEN_data_raw <- read.csv("./data/80249_DEN.csv") %>%
    mutate(origin = "DEN")
  ORD_data_raw <- read.csv("./data/60666_ORD.csv") %>%
    mutate(origin = "ORD")
  LAX_data_raw <- read.csv("./data/90045_LAX.csv") %>%
    mutate(origin = "LAX")
  DFW_data_raw <- read.csv("./data/75261_DFW.csv") %>%
    mutate(origin = "DFW")
  SFO_data_raw <- read.csv("./data/94128_SFO.csv") %>%
    mutate(origin = "SFO")
  PHX_data_raw <- read.csv("./data/85034_PHX.csv") %>%
    mutate(origin = "PHX")
  LAS_data_raw <- read.csv("./data/89119_LAS.csv") %>%
    mutate(origin = "LAS")
  SEA_data_raw <- read.csv("./data/98158_SEA.csv") %>%
    mutate(origin = "SEA")
  MSP_data_raw <- read.csv("./data/55111_MSP.csv") %>%
    mutate(origin = "MSP")
  
  ### Create a single dataframe
  weather_data <- rbind(ATL_data_raw,DEN_data_raw,ORD_data_raw,LAX_data_raw,
                        DFW_data_raw,SFO_data_raw,PHX_data_raw,LAS_data_raw,
                        SEA_data_raw,MSP_data_raw)
  ### Define data pipeline for weather data
  weather_pipeline <- .%>%
    clean_names() %>%
    select(origin,
           date_time,
           winddir_degree,
           windspeed_kmph,
           visibility,
           precip_mm,
           cloudcover,
           total_snow_cm,
           pressure,
           temp_c,
           dew_point_c,
           wind_gust_kmph)
  
  
  ### Initial weather dataset
  weather_data <- weather_pipeline(weather_data)
  colnames(weather_data) <- c("origin", 
                              "date", 
                              "origin_wd", 
                              "origin_ws", 
                              "origin_visibility", 
                              "origin_precip", 
                              "origin_cloudcover", 
                              "origin_total_snow", 
                              "origin_atmos_pressure",
                              "origin_air_temp",
                              "origin_dew_point",
                              "origin_wind_gust")
  weather_data2 <- weather_data
  colnames(weather_data2) <- c("destination", 
                               "date", 
                               "destination_wd", 
                               "destination_ws", 
                               "destination_visibility", 
                               "destination_precip", 
                               "destination_cloudcover", 
                               "destination_total_snow", 
                               "destination_atmos_pressure",
                               "destination_air_temp",
                               "destination_dew_point",
                               "destination_wind_gust")
  
  fwrite(weather_data,file = "data/new_weather_data/origin_weather_data.csv") 
  fwrite(weather_data2,file = "data/new_weather_data/destination_weather_data.csv")
}

#####################################################################
### Function to obtain confusion matrix elements for test data.
### model:        Sparklyr or H2O model object.
### test_data:    Test data to obtain confusion matrix from.
### type:         "spark" for Sparklyr model, "h2o" for H2O model.
### Returns confusion matrix elements. 
#####################################################################
confusion_matrix_elements <- function(model,test_data,type){
  if(type == "spark")
  {
    predictions <- ml_predict(model,test_data)  
  }
  if(type == "h2o")
  {
    predictions <- h2o.predict(model, test_data)
    predictions <- as.data.frame(predictions)
    actual_delays <- data.frame(as.data.frame(test_data) %>% pull(delay))
    colnames(actual_delays) <- c("actual_delay")
    
    evaluation_frame <- predictions %>%
      append(actual_delays)
    
    predictions <- evaluation_frame
    predictions <- data.frame(predictions) %>%
      select(predict,actual_delay)
    
    colnames(predictions) <- c("predicted_label", "delay")
    
    predictions <- copy_to(dest = sc,
                           df = predictions,
                           overwrite = TRUE)
  }
  
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
  confusion_matrix_element <- list(TP = TP,
                                   TN = TN,
                                   FP = FP,
                                   FN = FN)
  return(confusion_matrix_element)
}