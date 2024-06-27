require(ProbCast)
require(rstudioapi)
require(data.table)
require(mboost)
require(lubridate)
require(mgcv)
require(quantreg)
require(evgam)
require(ggplot2)
require(xtable)
allNs <- c("A","B","C","D","E","F","G","H","J","K","L","M","N","P")

Version <- "9_2"

setwd(dirname(getActiveDocumentContext()$path))

# This file contains the forecasts obtained with the static and dynamic Kalman GAM 
# adaptations of the GAM-Point model

## Reading dataset
NodeData <- readRDS("data_processed.Rda")



## List of Model Formulas
{Model_list <- list()

## Vanilla
Model_list[["Vanilla-T"]] <- node_n ~
  t + moy + dow_RpH + clock_fac + clock_fac:dow_RpH +
  x2T_weighted.mean_p_max_point + I(x2T_weighted.mean_p_max_point^2) +
  I(x2T_weighted.mean_p_max_point^3) +
  x2T_weighted.mean_p_max_point:moy + I(x2T_weighted.mean_p_max_point^2):moy +
  I(x2T_weighted.mean_p_max_point^3):moy +
  x2T_weighted.mean_p_max_point:clock_fac + I(x2T_weighted.mean_p_max_point^2):clock_fac +
  I(x2T_weighted.mean_p_max_point^3):clock_fac

Model_list[["Vanilla-T_qr"]] <- ~ gam_pred + I(gam_pred^2) +
  clock_fac +
  dow_RphG +
  x2T_weighted.mean_p_max_point

## Vanilla Plus
Model_list[["Vanilla-Point"]] <- node_n ~
  t + moy + dow_RpH + clock_fac + clock_fac:dow_RpH +
  x2T_weighted.mean_p_max_point + I(x2T_weighted.mean_p_max_point^2) +
  I(x2T_weighted.mean_p_max_point^3) +
  x2T_weighted.mean_p_max_point:moy + I(x2T_weighted.mean_p_max_point^2):moy +
  I(x2T_weighted.mean_p_max_point^3):moy +
  x2T_weighted.mean_p_max_point:clock_fac + I(x2T_weighted.mean_p_max_point^2):clock_fac +
  I(x2T_weighted.mean_p_max_point^3):clock_fac +
  WindSpd100_weighted.mean_cell +
  SSRD_mean_2_Cap


Model_list[["Vanilla-Point_qr"]] <- ~ gam_pred + I(gam_pred^2) +
  clock_fac +
  dow_RphG +
  x2T_weighted.mean_p_max_point +
  WindSpd100_weighted.mean_cell +
  SSRD_mean_2_Cap

## GAM-T: GAM with Temperature only (same features as Vanilla)
Model_list[["GAM-T"]] <- node_n~
  ## Linear/poly trends
  doy_s + doy_c + doy_s2 + doy_c2 + # Fourier annual seasonality
  t+I(t^2)+
  node_n_sm_L1 +
  #
  s(clock_hour,k=35,bs="cr")+
  dow_RpH+School_Hol+
  s(clock_hour,by=dow_RpH,k=30,bs="cr")+ # factor smooth
  s(clock_hour,by=School_Hol,k=20,bs="cr")+
  s(x2T_weighted.mean_p_max_point,k=35,bs="cr")+
  #s(x2Tsm_point,k=35,bs="cr")+
  ti(x2T_weighted.mean_p_max_point,clock_hour) # tensor interactions

Model_list[["GAM-T_qr"]] <- ~ gam_pred + I(gam_pred^2) +
  clock_fac +
  dow_RphG +
  x2T_weighted.mean_p_max_point

## GAM-Point: GAM with simple weather features for embedded generation 
Model_list[["GAM-Point"]] <- node ~ #node_n ~ 
  ## Linear/poly trends
  doy_s + doy_c + doy_s2 + doy_c2 + # Fourier annual seasonality
  t+I(t^2)+
  #
  node_n_sm_L1 +
  s(clock_hour,k=35,bs="cr")+
  dow_RpH+School_Hol+
  s(clock_hour,by=dow_RpH,k=30,bs="cr")+ # factor smooth
  s(clock_hour,by=School_Hol,k=20,bs="cr")+
  s(x2T_weighted.mean_p_max_point,k=35,bs="cr")+
  s(x2Tsm_point,k=35,bs="cr")+
  s(SSRD_mean_2_Cap,k=5,bs="cr")+
  s(WindSpd100_weighted.mean_cell,k=20,by=EMBEDDED_WIND_CAPACITY,bs="cr")+
  WindSpd10_weighted.mean_cell + 
  s(TP_weighted.mean_cell,bs="cr") +
  #ti(n2ex,clock_hour)+
  ti(x2T_weighted.mean_p_max_point,clock_hour)+
  ti(TP_weighted.mean_cell,clock_hour)

Model_list[["GAM-Point_qr"]] <- ~ gam_pred + I(gam_pred^2) +
  clock_fac +
  dow_RphG +
  SSRD_mean_2_Cap +
  WindSpd100_weighted.mean_cell +
  x2T_weighted.mean_p_max_point

## GAM-Grid: GAM with weather features derived from grid
Model_list[["GAM-Grid"]] <- node_n~
  ## Linear/poly trends
  doy_s + doy_c + doy_s2 + doy_c2 + # Fourier annual seasonality
  t+ I(t^2)+ 
  #
  node_n_sm_L1 +
  s(clock_hour,k=35,bs="cr")+
  dow_RpH+School_Hol+
  s(clock_hour,by=dow_RpH,k=30,bs="cr")+ # factor smooth
  s(clock_hour,by=School_Hol,k=20,bs="cr")+
  s(x2T_weighted.mean_pcell,k=35,bs="cr")+
  #s(x2Tsm,k=35,bs="cr")+
  s(SSRD_mean_2_Cap,k=5,bs="cr")+
  s(SSRD_max_2,by=SolarCap,bs="cr",k=5)+
  s(SSRD_weighted.sd_cell,by=SolarCap,bs="cr",k=5)+
  #s(MaxCC,by=SolarCap,bs="cr",k=5)+
  s(WindSpd100_weighted.mean_cell,k=20,by=EMBEDDED_WIND_CAPACITY,bs="cr")+
  WindSpd10_weighted.sd_cell +
  s(TP_weighted.mean_cell,bs="cr",k=5)+
  s(TP_weighted.sd_cell,k=5,bs="cr")+
  #ti(n2ex,clock_hour)+
  ti(x2T_weighted.mean_pcell,clock_hour)+
  ti(TP_weighted.mean_cell,clock_hour)+
  ti(WindSpd10_weighted.mean_cell,SSRD_mean_2_Cap)


Model_list[["GAM-Grid_qr"]] <- ~ gam_pred + I(gam_pred^2) + 
  clock_fac +
  dow_RphG +
  SSRD_mean_2_Cap +
  SSRD_weighted.sd_cell +
  WindSpd100_weighted.mean_cell +
  WindSpd100_weighted.sd_cell +
  x2T_weighted.mean_pcell
}

for(N in allNs)
{
  NodeData[[N]] <- data.table(NodeData[[N]])
}  


NodeTrain <- list()
NodeTest <- list()
GAM_models <- list()
y_GAM <- list()
y_pred <- list()

# Split dataset in train and test set
for(N in allNs){
  NodeTrain[[N]] <- NodeData[[N]][NodeData[[N]]$"targetTime" < "2019-01-02", ][1:(nrow(NodeData[[N]][NodeData[[N]]$"targetTime" < "2019-01-02", ])-46),]
  NodeTest[[N]] <- NodeData[[N]][NodeData[[N]]$"targetTime" >= "2019-01-01", ][3:nrow(NodeData[[N]][NodeData[[N]]$"targetTime" >= "2019-01-01", ]),]
  
  NodeTrain[[N]] <- data.table(NodeTrain[[N]])
  NodeTest[[N]] <- data.table(NodeTest[[N]])
}  

# GAM-Point model training (using bam function of mgcv)
for(N in allNs){
  print(N)
  GAM_models[[N]] <- bam(formula=Model_list[7]$`GAM-Point`,
                              data=NodeTrain[[N]])
}

# GAM-Point model predictions
for(N in allNs){
  y_pred[[N]] <- predict(GAM_models[[N]],newdata=NodeTest[[N]])
  y_GAM[[N]] <- data.frame(y_pred[[N]])
}

# Static KF
y_kf_static_delay <- list()
std_theta_static <- list() 

source("utils.R")

# Static Kalman GAM
for(N in allNs){
  print(N)
  kf <- Kalman_Filter(GAM_models[[N]])
  # Kalman filter training
  kf <- kf$fit(kf,NodeTrain[[N]][721:nrow(NodeTrain[[N]])], NodeTrain[[N]][["node"]][721:nrow(NodeTrain[[N]]) ])

  # Predictions which take into account the 48h delay in receiving data
  y_kf_static_delay[[N]] <- kf$predict(kf, NodeTest[[N]], Y_test=NodeTest[[N]],target="node", delay=TRUE)
  # Vector of standard deviations for each parameter in theta for the Reduced Grid Search method
  # (to be used for the optimization of the variances of the Dynamic KF)
  std_theta_static[[N]] <- apply(kf$theta_mat[15000:nrow(kf$theta_mat),], 2, sd)
}

# Saving prediction results to file to be read with Python
for (N in allNs) {
  file_path <- file.path(file.path("..", "Python/Other_data/KF R"), sprintf("Group%s_KF_static_delay.Rda", N))
  saveRDS(y_kf_static_delay[[N]][["y_mean"]], file=file_path)
}

##################################################################
# Dynamic Kalman GAM

# Hyperparameters (Q, sigma, P_1|0 and theta_1|0) Optimization 

# Iterative Grid Search Method

# Function which maps the model matrix of the GAM model (for its 624 coefficients) to its terms 
# (20 terms in the GAM_Point model + Intercept) which are the only ones to be adapted

f <- function(df, GAM_model) {
  model_matrix <- predict(GAM_model, newdata=df, type = "lpmatrix")
  coefficients <- coef(GAM_model)
  coeff_list <- list(coefficients)
  term_names <- names(coeff_list[[1]])
  #Term indexes for GAM-Point model (the difference between every two indices represents the number
  #of coefficients required to represent the first term)
  indexes <- c(1,2,3,4,5,6,7,8,9,21,23,24,58,435,492,526,560,564,584,593,609,625)
  n_terms <- length(indexes)-1
  
  # Initialization of a matrix to hold the feature evaluations
  features_eval <- matrix(0, nrow = nrow(model_matrix), ncol = n_terms)
  coef_vector <- unlist(coef(GAM_model))
  # First column is the intercept
  features_eval[, 1] <- 1
  # Loop over each term
  for(f in 1:n_terms) {
    col_range <- indexes[f]:(indexes[f+1] - 1)
    if(length(col_range) == 1) {
      # Scalar case: Directly multiply each row by the scalar coefficient
      features_eval[, f] <- model_matrix[, col_range] * rep(coef_vector[col_range], nrow(model_matrix))
    } else {
      # Vector case: Perform matrix multiplication
      features_eval[, f] <- model_matrix[, col_range] %*% coef_vector[col_range]
    }
  }
  
  return(as.matrix(features_eval[,2:ncol(features_eval)]))
}


# Hyperparameter optimization using the Iterative Grid Search method (viking package)

# We are computing results only for GSP Groups C, H and P due to high computational costs
source("utils.R")
require(viking)
# Original q* list for the Iterative Grid Search
# q_list <- c(1, 2^-1, 2^-2, 2^-3, 2^-4, 2^-5, 2^-6, 2^-7, 2^-8, 2^-9, 2^-10, 
#            2^-11, 2^-12, 2^-13, 2^-14, 2^-15, 2^-16, 2^-17, 2^-18, 2^-19,2^-20, 
#            2^-21, 2^-22, 2^-23, 2^-24, 2^-25, 2^-26, 2^-27, 2^-28, 2^-29,2^-30)

# q* list employed when setting net-load as target variable (Section 7.2)
q_list <- c(2^-15,2^-20,2^-25,2^-28,2^-30,2^-32,2^-34,2^-36,2^-38,2^-40,2^-45,2^-50)

for (N in c("C", "H", "P")) {
  # Train set: Feature evaluation (X) and target variable (y)
  X <- f(NodeTrain[[N]][721:nrow(NodeTrain[[N]]) ], GAM_models[[N]])
  y <- NodeTrain[[N]][["node"]][721:nrow(NodeTrain[[N]]) ]
  # Iterative grid search (viking package) considering a delay of 48h in receiving data (96 data points)
  # This function takes around 24h per region to compute
  res <- viking::iterative_grid_search(X,y,q_list,delay=96,p1=1)
  
  # Saving matrix to file
  file_path <- file.path("DynamicKF_Matrices", sprintf("dynKF_matrices_%s_delay_node.Rda", N))
  saveRDS(res, file=file_path)
}


# Dynamic KF predictions with the hyperparameters obtained with Iterative Grid Search (for Groups C, H and P)
y_kf_dynamic_delay <- list()

source("utils.R")

for(N in c("C", "H", "P")){
  print(N)
  file_path <- file.path("DynamicKF_Matrices", sprintf("dynKF_matrices_%s_delay_node.Rda", N))
  res <- readRDS(file=file_path)
  
  # Dynamic Kalman GAM model is trained
  kf <- Kalman_Filter(GAM_models[[N]], Q=diag(c(0,diag(res$Q))), sigma=res$sig, P=diag(c(res$sig^2,diag(res$P))), theta1=c(1,res$theta)) 
  kf <- kf$fit(kf,NodeTrain[[N]][721:nrow(NodeTrain[[N]])], NodeTrain[[N]][["node"]][721:nrow(NodeTrain[[N]]) ])
  # Forecasts on the test set with a 48h delay
  y_kf_dynamic_delay[[N]] <- kf$predict(kf, NodeTest[[N]], Y_test=NodeTest[[N]], delay=TRUE)
  
  # Saving forecasts to file for python 
  file_path <- file.path(file.path("..", "Python/Other_data/KF R"), sprintf("Group%s_KF_dynamic_delay_node.Rda", N))
  saveRDS(y_kf_dynamic_delay[[N]][["y_mean"]], file=file_path)
}

###################################################################
# Log-likelihood comparison for the Iterative Grid Search method and the Reduced Grid Search methods
# for GSP Groups C, H and P

q_list <- c(1e-10,1e-11,1e-12,1e-13,1e-14,1e-15,1e-16,1e-17,1e-18,1e-19)

for(N in c("C", "H", "P")){
  print(N)
  # Iterative Grid Search
  file_path <- file.path("DynamicKF_Matrices", sprintf("dynKF_matrices_%s_delay_node.Rda", N))
  res <- readRDS(file=file_path)
  n_train <- nrow(NodeTrain[[N]][721:nrow(NodeTrain[[N]])])
  print(paste("Log-likelihood Iterative Grid Search", res$LOGLIK[length(res$LOGLIK)] * n_train))
  
  # Reduced Grid Search (I)
  optim <- QOptimization$new(gam_model=GAM_models[[N]],NodeTrain[[N]][721:nrow(NodeTrain[[N]])], NodeTrain[[N]][721:nrow(NodeTrain[[N]]) ])
  res <- optim$grid_search_reduced_likelihood(q_list=q_list) 
  print(paste("Log-likelihood Reduced Grid Search (I)", res$loglik))
  
  # Reduced Grid Search (II)
  optim <- QOptimization$new(gam_model=GAM_models[[N]],NodeTrain[[N]][721:nrow(NodeTrain[[N]])], NodeTrain[[N]][721:nrow(NodeTrain[[N]]) ])
  res <- optim$grid_search_reduced_likelihood(q_list=q_list, std_static=std_theta_static[[N]])
  print(paste("Log-likelihood Reduced Grid Search (II)", res$loglik))
  
}

###################################################################
# Reduced Grid Search Hyperparameter Optimization method

y_kf_dynamic_delay <- list()

source("utils.R")

q_list <- c(1e-10,1e-11,1e-12,1e-13,1e-14,1e-15,1e-16,1e-17,1e-18,1e-19)

# Reduced Grid Search (I) method for all regions
for(N in allNs){
  print(N)
  # Initialization of the optimization object with the training set
  optim <- QOptimization$new(gam_model=GAM_models[[N]],NodeTrain[[N]][721:nrow(NodeTrain[[N]])], NodeTrain[[N]][721:nrow(NodeTrain[[N]]) ])
  res <- optim$grid_search_reduced_likelihood(q_list=q_list)
  
  # Dynamic Kalman GAM object initialization and training 
  kf <- Kalman_Filter(GAM_models[[N]], Q=res$Q, sigma=res$sig, P=res$P, theta1=res$theta1)
  kf <- kf$fit(kf,NodeTrain[[N]][721:nrow(NodeTrain[[N]])], NodeTrain[[N]][["node"]][721:nrow(NodeTrain[[N]]) ])
  
  # Forecasts computed on the test set
  y_kf_dynamic_delay[[N]] <- kf$predict(kf, NodeTest[[N]], Y_test=NodeTest[[N]], delay=TRUE)
  
  # Saving forecasts to file
  file_path <- file.path(file.path("..", "Python/Other_data/KF R"), sprintf("Group%s_KF_dynamic_delay.Rda", N))
  saveRDS(y_kf_dynamic_delay[[N]][["y_mean"]], file=file_path)
}




