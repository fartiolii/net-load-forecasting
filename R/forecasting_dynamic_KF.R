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


## Load Dataset ####
NodeData <- readRDS("data_processed.Rda")


## List of Model Formulae ####
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
Model_list[["GAM-Point"]] <- node ~  
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


for(N in allNs){
  NodeTrain[[N]] <- NodeData[[N]][NodeData[[N]]$"targetTime" < "2019-01-02", ][1:(nrow(NodeData[[N]][NodeData[[N]]$"targetTime" < "2019-01-02", ])-46),]
  NodeTest[[N]] <- NodeData[[N]][NodeData[[N]]$"targetTime" >= "2019-01-01", ][3:nrow(NodeData[[N]][NodeData[[N]]$"targetTime" >= "2019-01-01", ]),]
  
  NodeTrain[[N]] <- data.table(NodeTrain[[N]])
  NodeTest[[N]] <- data.table(NodeTest[[N]])
}  

# Compute predictions with offline GAM model
# for(N in allNs){
#   # GAM model
#   print(N)
#   GAM_models[[N]] <- bam(formula=Model_list[7]$`GAM-Point`,
#                               data=NodeTrain[[N]])
# }
# 
# #Prediction with no data update on test set
# for(N in allNs){
#   y_pred[[N]] <- predict(GAM_models[[N]],newdata=NodeTest[[N]])
#   y_GAM[[N]] <- data.frame(y_pred[[N]])
# }
# 
#saveRDS(GAM_models, file="GAM_models_node")

# Import offline GAM models from file
GAM_models <- readRDS(file="GAM_models_node")



# Dynamic KF - Optimization of Q, sigma, P_1|0 and theta_1|0

# Function which maps the matrix of features and the coefficients of the GAM model (more than 600) 
# to its terms (20 terms in the GAM_Point model + Intercept) which are the only ones to be adapted

f <- function(df, GAM_model) {
  model_matrix <- predict(GAM_model, newdata=df, type = "lpmatrix")
  coefficients <- coef(GAM_model)
  coeff_list <- list(coefficients)
  term_names <- names(coeff_list[[1]])
  #Term indexes for GAM_Point model
  indexes <- c(1,2,3,4,5,6,7,8,9,21,23,24,58,435,492,526,560,564,584,593,609,625)
  n_terms <- length(indexes)-1

  # Initialization of a matrix to hold the feature evaluations
  features_eval <- matrix(0, nrow = nrow(model_matrix), ncol = n_terms)
  coef_vector <- unlist(coef(GAM_model))
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
  # First column is the intercept
  features_eval[, 1] <- 1
  
  return(as.matrix(features_eval))
}


# Variances optimization using the viking package
source("utils.R")
require(viking)
q_list <- c(1, 2^-1, 2^-2, 2^-3, 2^-4, 2^-5, 2^-6, 2^-7, 2^-8, 2^-9, 2^-10, 
            2^-11, 2^-12, 2^-13, 2^-14, 2^-15, 2^-16, 2^-17, 2^-18, 2^-19,2^-20, 
            2^-21, 2^-22, 2^-23, 2^-24, 2^-25, 2^-26, 2^-27, 2^-28, 2^-29,2^-30)

for (N in allNs) {
  # Train set: Feature evaluation (X) and target variable (y)
  X <- f(NodeTrain[[N]][721:nrow(NodeTrain[[N]]) ], GAM_models[[N]])
  y <- NodeTrain[[N]][["node"]][721:nrow(NodeTrain[[N]]) ]
  # Iterative grid search (viking package) considering a delay of 48h in receiving data
  # This function takes around 1d and 10h per region to compute
  res <- viking::iterative_grid_search(X,y,q_list,delay=96)
  
  # Saving matrix to file
  file_path <- file.path("DynamicKF_Matrices", sprintf("dynKF_matrices_%s_delay.Rda", N))
  saveRDS(res, file=file_path)
}




# Dynamic KF predictions with the obtained matrices
y_kf_dynamic <- list()
y_kf_dynamic_delay <- list()

source("utils.R")

for(N in allNs){
  print(N)
  file_path <- file.path("DynamicKF_Matrices", sprintf("dynKF_matrices_%s_delay.Rda", N))
  res <- readRDS(file=file_path)
  kf <- Kalman_Filter(GAM_models[[N]], Q=diag(diag(res$Q)), sigma=res$sig, P=res$P) #, theta1=res$theta) 
  
  kf <- kf$fit(kf,NodeTrain[[N]][721:nrow(NodeTrain[[N]])], NodeTrain[[N]][["node"]][721:nrow(NodeTrain[[N]]) ])
  y_kf_dynamic[[N]] <- kf$predict(kf, NodeTest[[N]], Y_test=NodeTest[[N]][["node"]])
  y_kf_dynamic_delay[[N]] <- kf$predict(kf, NodeTest[[N]], Y_test=NodeTest[[N]], delay=TRUE)
  
  
  file_path <- file.path(file.path("..", "Python/Other_data/KF R"), sprintf("Group%s_KF_dynamic.Rda", N))
  saveRDS(y_kf_dynamic[[N]][["y_mean"]], file=file_path)
  file_path <- file.path(file.path("..", "Python/Other_data/KF R"), sprintf("Group%s_KF_dynamic_delay.Rda", N))
  saveRDS(y_kf_dynamic_delay[[N]][["y_mean"]], file=file_path)
}



# Dynamic KF prediction with matrices obtained by optimizing the Likelihood on a grid of q* values

y_kf_dynamic <- list()
y_kf_dynamic_delay <- list()

source("utils.R")

q_list <- c(1e-14,1e-15,1e-16, 1e-17,1e-18)


for(N in allNs){
  print(N)
  # Initialization of the optimization object with the training set
  optim <- QOptimization$new(gam_model=GAM_models[[N]],NodeTrain[[N]][721:nrow(NodeTrain[[N]])], NodeTrain[[N]][721:nrow(NodeTrain[[N]]) ])
  
  # Maximization of the likelihood on the grid of q*
  res <- optim$grid_search_reduced_likelihood(q_list=q_list) #, std_static=std_theta_static[[N]])
  
  kf <- Kalman_Filter(GAM_models[[N]], Q=res$Q, sigma=res$sig, P=res$P, theta1=res$theta1)
  
  kf <- kf$fit(kf,NodeTrain[[N]][721:nrow(NodeTrain[[N]])], NodeTrain[[N]][["node"]][721:nrow(NodeTrain[[N]]) ])
  
  y_kf_dynamic_delay[[N]] <- kf$predict(kf, NodeTest[[N]], Y_test=NodeTest[[N]], delay=TRUE)
  
  
  file_path <- file.path(file.path("..", "Python/Other_data/KF R"), sprintf("Group%s_KF_dynamic_delay.Rda", N))
  saveRDS(y_kf_dynamic_delay[[N]][["y_mean"]], file=file_path)
}
print(RMSE(y_kf_dynamic_delay[[N]][["y_mean"]], NodeTest[[N]][["node"]]))
print(MAE(y_kf_dynamic_delay[[N]][["y_mean"]], NodeTest[[N]][["node"]]))




# The best prediction results to date (for the 48h delay case) have been obtained with 
# the same sigma and Q for all regions: q=1e-13 and sigma=0.08
sigma <- 0.08
q <- 1e-13
for(N in allNs){
  print(N)
  kf <- Kalman_Filter(GAM_models[[N]], Q=diag(rep(q, 21)), 
                                  sigma=sigma,
                                  P=diag(rep(1, 21)*sigma^2))
  
  kf <- kf$fit(kf,NodeTrain[[N]][721:nrow(NodeTrain[[N]])], NodeTrain[[N]][["node"]][721:nrow(NodeTrain[[N]]) ])
  y_kf_dynamic_delay[[N]] <- kf$predict(kf, NodeTest[[N]], Y_test=NodeTest[[N]], delay=TRUE, const_delay=FALSE)

  file_path <- file.path(file.path("..", "Python - to send/Other_data/KF R"), sprintf("Group%s_KF_dynamic_delay_fxd.Rda", N))
  saveRDS(y_kf_dynamic_delay[[N]][["y_mean"]], file=file_path)
}
print(RMSE(y_kf_dynamic_delay[[N]][["y_mean"]], NodeTest[[N]][["node"]]))
print(MAE(y_kf_dynamic_delay[[N]][["y_mean"]], NodeTest[[N]][["node"]]))




