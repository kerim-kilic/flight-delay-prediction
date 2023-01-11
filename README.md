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

During development of the code **Spark version 3.3.0** was used, with the latest version of h2o and sparkling water.

## Data used

The data in **data/flights_2017.RDS** contains data of approximately 5.5 million domestic flights of the US. The predictive models are trained to make predictions for the top 10 airports in the US with the most flights in 2017. The airports are given in the table below:

| Symbol                               | Airport                                            | State                            |
| ------------------------------------ | -------------------------------------------------- | -------------------------------- |
| ATL                                  | Hartsfield-Jackson Atlanta International Airport   | Georgia                          |
| DEN                                  | Denver International Airport                       | Colorado                         |
| DFW                                  | Dallas/Fort Worth International Airport            | Texas                            |
| LAS                                  | Harry Reid International Airport                   | Nevada                           |
| LAX                                  | Los Angeles International Airport                  | California                       |
| MSP                                  | Minneapolis-Saint Paul International Airport       | Minnesota                        |
| ORD                                  | Chicago O’Hare International Airport               | Illinois                         |
| PHX                                  | Phoenix Sky Harbor International Airport           | Arizona                          |
| SEA                                  | Seattle-Tacoma International Airport               | Washington                       |
| SFO                                  | San Francisco International Airport                | California                       |

The folder **data/** contains the weather data for each day for each airport, the files are all indexed using the ZIP code and the airport symbol.

The remaining files in **data/** contain

## Models

The following models are trained and evaluated in this repository:

| Model                               | Tool                          |
| ----------------------------------- | ----------------------------- | 
| Logistic regression                 | Spark                         | 
| Random forest                       | Spark                         |
| Gradient boosting machine           | h2o                           |
| Feed-forward neural network         | h2o                           |
