### 
### Author: Kerim Kiliç
###

### Function to generate and return metrics of machine learning model
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

# Function to create the split of the data into train and test
create_train_test_split <- function(data, sample_size, ratio, type, hc)
{
  ## Splitted datasets for classification
  # Balanced training set using under sampling
  train_size <- sample_size * ratio
  test_size <- sample_size * (1-ratio)
  
  if(type == "numerical")
  {
    ## Splitted datasets for numerical prediction
    data_tmp <- data %>%
      sample_frac(sample_size/sdf_nrow(data))
  
    train_test_split <- sdf_random_split(x = data_tmp,
                                            train_data = ratio,
                                            test_data = 1-ratio,
                                            seed = 1234)  
  }
  
  if(type == "classification")
  {
    delay_yes <- data %>%
      filter(delay == "1")
    delay_no <- data %>%
      filter(delay == "0")
    
    tmp1 <- delay_yes %>% 
      sample_n(sample_size/2)
    tmp2 <- delay_no %>% 
      sample_n(sample_size/2)
    data_tmp <- rbind(tmp1, tmp2)
    
    train_test_split <- sdf_random_split(x = data_tmp,
                                         train_data = ratio,
                                         test_data = 1-ratio,
                                         seed = 1234)  
  }
  
  if(type == "h2o_classification")
  {
    delay_yes <- data %>%
      filter(delay == "1")
    delay_no <- data %>%
      filter(delay == "0")
    
    tmp1 <- delay_yes %>% 
      sample_n(sample_size/2)
    tmp2 <- delay_no %>% 
      sample_n(sample_size/2)
    sample_data_classification <- rbind(tmp1, tmp2)
    sample_data_classification <- hc$asH2OFrame(sample_data_classification)
    
    train_test_split <- h2o.splitFrame(
      data = sample_data_classification,
      ratios = c(ratio,(1-ratio)/2),   ## only need to specify 2 fractions, the 3rd is implied
      destination_frames = c("train1.hex", "valid1.hex", "test1.hex"), seed = 1234
    )
  }
  
  if(type == "h2o_numerical")
  {
    sample_data_numerical <- data %>%
      sample_n(sample_size)
    sample_data_numerical <- hc$asH2OFrame(sample_data_numerical)
    train_test_split <- h2o.splitFrame(
      data = sample_data_numerical,
      ratios = c(ratio,(1-ratio)/2),   ## only need to specify 2 fractions, the 3rd is implied
      destination_frames = c("train2.hex", "valid2.hex", "test2.hex"), seed = 1234
    )
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
  # Measure the time it takes to cross validate model
  train_time <- end_time - start_time
  
  cv_results <- cv_model$avg_metrics_df
  if(type == "classification")
  {
    # best_cv_result <- cv_results[which.max(cv_results$accuracy),"accuracy"]
    best_cv_result <- cv_results[which.max(cv_results$accuracy),]
  }
  if(type == "numerical")
  {
    # best_cv_result <- cv_results[which.max(cv_results$r2),"r2"] 
    best_cv_result <- cv_results[which.max(cv_results$r2),] 
  }
   
  final_cv_results <- list(all_results = cv_results,
                           best_result = best_cv_result,
                           train_time = train_time)
  class(final_cv_results) <- "final_cv_results"
  
  return(final_cv_results)
}

### Function to get the weather data from the top 10 airports

get_weather_data <- function(sc)
{
  ### Get the correct data for each airport
  ATL_data_raw <- importNOAA(code = "722190-13874", year = 2017) %>%
    mutate(origin = "ATL")
  DEN_data_raw <- importNOAA(code = "725650-03017", year = 2017) %>%
    mutate(origin = "DEN")
  ORD_data_raw <- importNOAA(code = "725300-94846", year = 2017) %>%
    mutate(origin = "ORD")
  LAX_data_raw <- importNOAA(code = "722950-23174", year = 2017) %>%
    mutate(origin = "LAX")
  DFW_data_raw <- importNOAA(code = "722590-03927", year = 2017) %>%
    mutate(origin = "DFW")
  SFO_data_raw <- importNOAA(code = "724940-23234", year = 2017) %>%
    mutate(origin = "SFO")
  PHX_data_raw <- importNOAA(code = "722780-23183", year = 2017) %>%
    mutate(origin = "PHX")
  LAS_data_raw <- importNOAA(code = "723860-23169", year = 2017) %>%
    mutate(origin = "LAS")
  SEA_data_raw <- importNOAA(code = "727930-24233", year = 2017) %>%
    mutate(origin = "SEA")
  MSP_data_raw <- importNOAA(code = "726580-14922", year = 2017) %>%
    mutate(origin = "MSP")
  ### Create a single dataframe
  weather_data <- rbind(ATL_data_raw,DEN_data_raw,ORD_data_raw,LAX_data_raw,
                        DFW_data_raw,SFO_data_raw,PHX_data_raw,LAS_data_raw,
                        SEA_data_raw,MSP_data_raw)
  ### Define data pipeline for weather data
  weather_pipeline <-  .%>%
    mutate(date = as.Date(date,format = "%y-%m-%d")) %>%
    select(origin,date,
           # latitude,longitude,elev,
           precip_6,
           ws,wd,air_temp,atmos_pres,visibility,dew_point) 
  
  
  ### Initial weather dataset
  weather_data <- weather_pipeline(weather_data)
  weather_data <- copy_to(dest = sc,
                          df = weather_data,
                          overwrite = TRUE)
  ### Perform imputation on missing values
  input_cols_imp <- c("precip_6","ws","wd","air_temp","atmos_pres","visibility","dew_point")
  output_cols_imp <- paste0(input_cols_imp,"_imp")
  weather_data <- ft_imputer(weather_data, input_cols = input_cols_imp, output_cols = output_cols_imp, strategy = "mean")
  
  ### Calculate average value per day
  weather_data <- weather_data %>% collect()
  weather_data <- weather_data %>%
    group_by(origin, date) %>%
    summarise(across(c(ws_imp,
                       # latitude, longitude, elev, 
                       wd_imp, air_temp_imp, atmos_pres_imp,
                       visibility_imp, dew_point_imp,precip_6_imp), mean))
  
  ### One dataframe for the origin weather data and one for the destination weather data
  weather_data2 <- weather_data
  colnames(weather_data) <- c("origin", "date", "origin_ws", "origin_wd", "origin_air_temp", "origin_atmos_pres", "origin_visibility", "origin_dew_point", "origin_precip_6")
  colnames(weather_data2) <- c("destination", "date", "destination_ws", "destination_wd", 
                               "destination_air_temp", "destination_atmos_pres", "destination_visibility", "destination_dew_point","destination_precip_6")
  fwrite(weather_data,file = "data/origin_weather_data.csv") 
  fwrite(weather_data2,file = "data/destination_weather_data.csv")
}