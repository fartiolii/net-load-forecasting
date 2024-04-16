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


## Load and Prep Dataset ####
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
  WindSpd10_weighted.mean_cell + 
  dow_RpH+School_Hol+
  
  s(clock_hour,k=35,bs="cr")+
  s(clock_hour,by=dow_RpH,k=30,bs="cr")+ # factor smooth
  s(clock_hour,by=School_Hol,k=20,bs="cr")+
  s(x2T_weighted.mean_p_max_point,k=35,bs="cr")+
  s(x2Tsm_point,k=35,bs="cr")+
  s(SSRD_mean_2_Cap,k=5,bs="cr")+
  s(WindSpd100_weighted.mean_cell,k=20,by=EMBEDDED_WIND_CAPACITY,bs="cr")+
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

# Splitting the dataset in train and test set for every region
for(N in allNs){
  NodeTrain[[N]] <- NodeData[[N]][NodeData[[N]]$"targetTime" < "2019-01-02", ][1:(nrow(NodeData[[N]][NodeData[[N]]$"targetTime" < "2019-01-02", ])-46),]
  NodeTest[[N]] <- NodeData[[N]][NodeData[[N]]$"targetTime" >= "2019-01-01", ][3:nrow(NodeData[[N]][NodeData[[N]]$"targetTime" >= "2019-01-01", ]),]
  
  NodeTrain[[N]] <- data.table(NodeTrain[[N]])
  NodeTest[[N]] <- data.table(NodeTest[[N]])
}  

for(N in allNs){
  # GAM model (bam function of mgcv)
  print(N)
  GAM_models[[N]] <- bam(formula=Model_list[7]$`GAM-Point`,
                              data=NodeTrain[[N]])
}  


idx_2020 = 17516
idx_2021 = 35104

# Offline GAM 
for(N in allNs){
  y_pred[[N]] <- predict(GAM_models[[N]],newdata=NodeTest[[N]])
  y_GAM[[N]] <- data.frame(y_pred[[N]])
  
  # Save predictions to file to be read with Python
  file_path <- file.path(file.path("..", "Code/Other_data/Results R"), sprintf("Group%s_GAM_R_bam.Rda", N))
  saveRDS(y_GAM[[N]], file=file_path)
}


# GAM with yearly udpate 
for(N in allNs){  
    y_pred[[N]] <- predict(GAM_models[[N]],newdata=NodeTest[[N]][1:idx_2020, ])
    
    # Update model with data from 2019
    first_data_batch <- NodeTest[[N]][1:idx_2020, ]
    
    temp <- bam.update(GAM_models[[N]], first_data_batch)
    
    # Predict data for 2020
    y_pred[[N]] <- c(y_pred[[N]],predict(temp,newdata=NodeTest[[N]][(idx_2020+1):idx_2021, ]))
    
    
    # Update model with data from 2020
    second_data_batch <- NodeTest[[N]][(idx_2020+1):idx_2021, ]
    
    temp <- bam.update(temp, second_data_batch)
    
    # Predict data for 2021
    y_pred[[N]] <- c(y_pred[[N]],predict(temp,newdata=NodeTest[[N]][(idx_2021+1):nrow(NodeTest[[N]]),]))
    
    
    y_GAM[[N]] <- data.frame(y_pred[[N]])
    
    # Save predictions to file to be read with Python
    file_path <- file.path(file.path("..", "Code/Other_data/Results R"), sprintf("Group%s_bam_yearly_updated.Rda", N))
    saveRDS(y_GAM[[N]], file=file_path)
}









 
