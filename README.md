# Flight delay prediction in the US airport network

The **flight-delay-prediction.Rmd** markdown in this repository analyzes US flight data from 2017 to train several machine learning models to make delay predictions. The markdown uses **Spark** through the **sparklyr** library and **h2o** through the **h2o** and **sparkling water** libraries to efficiently process large quantities of data and train the machine learning models. The markdown calls custom functions in **src/functions.R** to create train, test, and validation data, to train and evaluate several machine learning models.

## Requirements

The R script requires the following packages:

```r
library(data.table)
library(tidymodels)
library(sparklyr)
library(kableExtra)
library(h2o)
library(rsparkling)
library(worldmet)
library(janitor)
```

## Models

The following models are trained and evaluated in this repository:

| Model                               | Tool                          |
| ----------------------------------- | ----------------------------- | 
| Logistic regression                 | Spark                         | 
| Random forest                       | Spark                         |
| Gradient boosting machine           | h2o                           |
| Feed-forward neural network         | h2o                           |
