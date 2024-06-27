# Kalman Filter and QOptimization classes (see the Python classes for further details)

Kalman_Filter <- function(gam_model, theta1 = NULL, theta = NULL, P = NULL, sigma = NULL, Q = NULL) {
  # Kalman Filter class used to adapt the coefficients of the terms of a GAM model trained with
  # the mgcv package
  obj <- list(
    GAM_model = gam_model,
    theta1 = theta1, # Initial vector of parameters (theta_1|0)
    theta = theta,   
    theta_mat = NULL,
    P = P,           # P_1|0 matrix
    sigma = sigma,   # Variance of the target variable measurement
    Q = Q            # Variance Covariance matrix of the error on the state theta
  )

  
  # Function which maps the model matrix of the GAM model (for its 624 coefficients) to its terms 
  # (20 terms in the GAM_Point model + Intercept) which are the only ones to be adapted
  
  obj$f <- function(obj,df,train=FALSE) {
    model_matrix <- predict(obj$GAM_model, newdata=df, type = "lpmatrix")
    coefficients <- coef(obj$GAM_model)
    coeff_list <- list(coefficients)
    term_names <- names(coeff_list[[1]])
    #Term indexes for GAM-Point model (the difference between every two indices represents the number
    #of coefficients required to represent the first term)
    indexes <- c(1,2,3,4,5,6,7,8,9,21,23,24,58,435,492,526,560,564,584,593,609,625)
    n_terms <- length(indexes)-1
    
    # Initialization of a matrix to store the feature evaluations
    features_eval <- matrix(0, nrow = nrow(model_matrix), ncol = n_terms)
    coef_vector <- unlist(coef(obj$GAM_model))
    # First column is the intercept
    features_eval[, 1] <- 1
    # For cycle over each term
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
    
    return(as.matrix(features_eval))
  }

  # Update of the theta vector with the latest measurement yt (Equation (5.8), first equation)
  obj$theta_update <- function(obj,f_Xt, yt) {
    obj$theta <- obj$theta + obj$P %*% f_Xt / (obj$sigma^2) * as.numeric(yt - t(obj$theta) %*% f_Xt)
    return(obj)
  }

  # Update of matrix P (Equation (5.7))
  obj$P_update <- function(obj,f_Xt, yt) {
    num <- obj$P %*% (f_Xt %*% (t(f_Xt) %*% obj$P))
    obj$P <- obj$P - num / as.numeric(t(f_Xt) %*% obj$P %*% f_Xt + obj$sigma^2)
    return(obj)
  }

  # Function that fits the KF on the train set in input
  obj$fit <- function(obj,X_train, Y_train) {
    Yt <- as.matrix(Y_train)
    f_Xt <- obj$f(obj,X_train)
      
    n <- nrow(f_Xt)
    d <- ncol(f_Xt)
    obj$d <- d

    # Matrix and vector initializations (default values are for the static Kalman GAM model)
    if (is.null(obj$theta1)) {
      # If the initial state vector is not given we initialize it to a vector of ones
      obj$theta1 <- matrix(1, nrow = d, ncol = 1)
    }
    obj$theta <- obj$theta1

    if (is.null(obj$P)) {
      obj$P <- diag(d)
    }
    if (is.null(obj$Q)) {
      obj$Q <- matrix(0, nrow = d, ncol = d)
    }
    if (is.null(obj$sigma)) {
      obj$sigma <- 1
    }
    
    # Matrix which saves the different states computed during the training phase
    theta_mat <- matrix(0, nrow = n, ncol = d)
    
    P <- obj$P
    theta <- obj$theta
    
    # For cycle that loops over all the points of the training set and updates theta and P accordingly
    for (i in seq_len(n)){
      theta_mat[i, ] <- t(theta)
      ft <- t(f_Xt[i, , drop = FALSE])
      P <- P - tcrossprod(P %*% ft) / ((obj$sigma^2) + (t(ft) %*% P %*% ft)[1])
      theta <- theta + P %*% ft / (obj$sigma^2) * (Yt[i] - (t(theta) %*% ft)[1])
      P <- P + obj$Q
    }
    obj$P <- P
    obj$theta <- theta
    obj$theta_mat <- theta_mat
    
    return(obj)
  }

  # Function that, given the fitted KF, returns the predictions of the test set in input
  # by considering data up to 30 min before (delay=FALSE) or considering a 48h delay
  # in receiving data (delay=TRUE)
  # const_delay: Bool, True if we incorporate past data in the model with a constant delay of 48h, False if instead we take into 
  # account the fact that at midnight of each day, we get all data regarding the past 48 to 24 hours
  # The setting considered in the thesis is delay=TRUE and const_delay=TRUE
  
  obj$predict <- function(obj, X_test, Y_test, target="node", delay = FALSE, const_delay=TRUE) {
    
    # No delay version
    if (!delay) {
        if (is.null(Y_test)) {
          stop("Target variable must be given for the online version")
        }

        Y_test <- as.matrix(Y_test)
        f_Xtest <- obj$f(obj,X_test)
        n_test <- nrow(f_Xtest)
        y_mean <- numeric(n_test)
        y_std <- numeric(n_test)

        theta_init <- obj$theta
        P_init <- obj$P
        
        # For every target time in the test set, data up to the 30 min before is used
        for (i in seq_len(n_test)) {
          ft <- t(f_Xtest[i, , drop = FALSE])
          y_mean[i] <- t(obj$theta) %*% ft
          y_std[i] <- sqrt(obj$sigma^2 + t(ft) %*% (obj$P %*% ft))
          obj <- obj$P_update(obj,ft, Y_test[i])
          obj <- obj$theta_update(obj,ft, Y_test[i])
          obj$P <- obj$P + obj$Q
        }

        obj$theta <- theta_init
        obj$P <- P_init
      
    } else {
        # Delay is considered
        Y_target <- Y_test$targetTime
        Y_test <- as.matrix(Y_test[[target]]) 
        
        f_Xtest <- obj$f(obj,X_test)
        n_test <- nrow(f_Xtest)
        y_mean <- numeric(n_test)
        y_std <- numeric(n_test)
        
        if(const_delay == FALSE)
        {
          # In this case, to predict the value of every 30 min throughout a single day, we are using all measurements up to the midnight before 
          # (which are measurements that go from 48h to 24h before the midnight). e.g. the prediction of values from 25/01 at 00:00 to 25/01 at 23:30                 
          # occurs the 25/01 at 00:00 with new measurement data incorporated in the model which goes from 23/01 at 00:00 to 23/01 at 23:30.
          
          theta_init <- obj$theta
          P_init <- obj$P

          for (t in seq_len(n_test)) {
            # We check if the target time of the prediction corresponds to midnight
            if (hour(Y_target[t]) == 0 && minute(Y_target[t]) == 0) {
              # We check if data from 48h to 24h before is available (just a condition to check at the beginning of the training phase)
              if (t - 96 >= 0) {
                # KF is updated with measurements from 48h to 24h before (which correspond to indexes -96 to -48 since we have half an hour data)
                ft <- f_Xtest[(t - 96):(t - 48 -1), ]
                yt <- Y_test[(t - 96):(t - 48 -1), ]
                # For cycle that updates the KF with 48 measurements (which correspond to 24h data)
                for (j in seq_len(nrow(ft))) {
                  f_t <- t(ft[j, , drop = FALSE])
                  y_t <- yt[j]
                  obj <- obj$P_update(obj, f_t, y_t)
                  obj <- obj$theta_update(obj, f_t, y_t)
                  obj$P <- obj$P + obj$Q
                }
              }

              # Prediction of the next 24h are computed
              ft <- f_Xtest[t:(t + 48-1), ]
              y_mean[t:(t + 48-1)] <- as.vector(ft %*% obj$theta)

              # The prediction of the standard deviation requires to sum k-times Q to P, where k represents the number of steps ahead of the
              # prediction that we are doing (See Section 5.5)
              P <- obj$P+48*obj$Q # Equation (5.22)
              for (idx in t:(t + 47)) {
                f_t <- t(f_Xtest[idx, , drop = FALSE])
                P <- P + obj$Q
                y_std[idx] <- sqrt(obj$sigma^2 + t(f_t) %*% P %*% f_t)
              }
            }
          }
        }else  
        {
          # In this case we use the data to update the model as if data is constantly arriving with a 48h delay throughout the day. At each time                     
          # instant we update the model with the measurement of 48h before and then predict the new value of yt.
          # This is the setting considered for the results reported in the thesis.
          P <- obj$P
          theta <- obj$theta
          delay <- 96
          
          for(t in seq_len(n_test))
          {
            if (t > delay) {
              ft <- t(f_Xtest[t-delay, , drop = FALSE])
              P <- P - tcrossprod(P %*% ft) / (obj$sigma^2 + (t(ft) %*% P %*% ft)[1])
              theta <- theta + P %*% ft / (obj$sigma^2) * (Y_test[t-delay] - (t(theta) %*% ft)[1])
              P <- P + obj$Q
            }
            
            ft <- t(f_Xtest[t, , drop = FALSE])
            y_mean[t] <- crossprod(theta, ft)[1]
            y_std[t] <- sqrt(obj$sigma^2 + t(ft) %*% (P + 95*obj$Q)%*% ft)
          }
        }
      
    }
    return(list(y_mean = y_mean, y_std = y_std))
  }
  
  
  # The following two methods are used for the Reduced Grid Search method for hyperparameter
  # selection for the dynamic Kalman Filter only
  
  
  # Method that computes the likelihood (assuming delay=True) as in Equation (5.26) on the training set in input, given Q*,
  # and evaluates sigma (Equation (5.14)).
  obj$predict_reduced_likelihood <- function(obj,X_test, Y_test, target="node", delay=FALSE, const_delay=TRUE, fit_theta1=FALSE) {
    Y_target <- Y_test[['targetTime']]
    Y_test <- as.matrix(Y_test[[target]])
    f_Xtest <- obj$f(obj,X_test)
    n_test <- nrow(f_Xtest)
    obj$d <- ncol(f_Xtest)
    y_mean <- numeric(n_test)
    y_std <- numeric(n_test)
    
    # Since Q*:=Q/sigma^2 and P*:=P/sigma^2, if we set sigma=1 we can continue to use the previous notation (therefore attributes Q and P to indicate Q* and P*) 
    obj$sigma <- 1
    # Optimization of theta_1|0 (prior) (Formula (5.16)). It is not the optimal theta_1|0 chosen for the Reduced Grid Search method but
    # we allow the user to choose it. 
    if (fit_theta1){
      obj <- obj$optimize_theta1(obj, f_Xtest,Y_target,Y_test)
    }
    if (is.null(obj$theta1)) {
      # If the initial state is not given we set it to a vector of ones.
      obj$theta1 <- matrix(1, nrow = obj$d, ncol = 1)
    }
    obj$theta <- obj$theta1
    
    # Computation of the likelihood (Equation (5.26))
    loglik <- 0
    err_norm_sigma <- 0
    
    if (const_delay == FALSE){
      for (t in 1:n_test) {
        # We check if the target time of the prediction corresponds to midnight
        if (hour(Y_target[t]) == 0 && minute(Y_target[t]) == 0) {
          # We check if data from 48h to 24h before is available (just a condition to check at the beginning of the training phase)
          if (t - 96 >= 1) {
            # KF is updated with measurements from 48h to 24h before (which correspond to indexes -96 to -48 since we have half an hour data)
            ft <- f_Xtest[(t - 96):(t - 48-1), , drop = FALSE]
            yt <- Y_test[(t - 96):(t - 48-1), , drop = FALSE]
            for (j in 1:nrow(ft)) {
              f_t <- t(ft[j, , drop = FALSE])
              y_t <- yt[j, ]
              obj <- obj$P_update(obj, f_t, y_t)
              obj <- obj$theta_update(obj, f_t, y_t)
              obj$P <- obj$P + obj$Q
            }
          }
          # Predictions for the next 24h are computed
          ft <- f_Xtest[t:(t + 47), , drop = FALSE]
          y_mean[t:(t + 47)] <- as.vector(ft %*% obj$theta)

          # The prediction of the standard deviation requires to sum k-times Q to P, where k represents the number of steps ahead of the
          # prediction that we are doing (See Section 5.5)
          P <- obj$P+48*obj$Q # Equation (5.22)
          corr_val <- numeric(48)
          for (idx in t:(t + 47)) {
            ft <- t(f_Xtest[idx, , drop = FALSE])
            P <- P + obj$Q
            y_std[idx] <- sqrt(obj$sigma^2 + t(ft) %*% P %*% ft)
            corr_val[idx - t + 1] <- obj$sigma^2 + t(ft) %*% P %*% ft
          }
          
          # Terms required for the computation of the likelihood
          err <- (Y_test[t:(t + 47)] - y_mean[t:(t + 47)])^2
          loglik <- loglik + sum(log(corr_val))
          err_norm_sigma <- err_norm_sigma + sum(err / corr_val)
        }
      }
    }else{
      # In this case we use the data to update the model as if data is constantly arriving with a 48h delay throughout the day. At each time                       
      # instant we update the model with the measurement of 48h before and then predict the new value of yt.
      # This setting is chosen for the results reported in the thesis.
      P <- obj$P
      theta <- obj$theta
      delay <- 96
      
      for(t in seq_len(n_test))
      {
        if (t > delay) {
          ft <- t(f_Xtest[t-delay, , drop = FALSE])
          P <- P - tcrossprod(P %*% ft) / (obj$sigma^2 + (t(ft) %*% P %*% ft)[1])
          theta <- theta + P %*% ft / (obj$sigma^2) * (Y_test[t-delay] - (t(theta) %*% ft)[1])
          P <- P + obj$Q
        }
        
        ft <- t(f_Xtest[t, , drop = FALSE])
        y_mean[t] <- crossprod(theta, ft)[1]
        # Standard deviation of the prediction, matrix P_t|t-k is computed as in Equation (5.22)
        y_std[t] <- sqrt(obj$sigma^2 + t(ft) %*% (P + 95*obj$Q)%*% ft)
        
        # Terms required for the computation of the likelihood.
        err <- (Y_test[t] - y_mean[t])^2
        loglik <- loglik + sum(log(y_std[t]^2))
        err_norm_sigma <- err_norm_sigma + sum(err / y_std[t]^2)
      }
    }
    
    # Computation of sigma (Equation (5.14))
    sigma <- sqrt(err_norm_sigma / n_test)
    loglik <- loglik + n_test * log(err_norm_sigma / n_test)
   
    return(list(y_mean = y_mean, y_std = y_std, loglik = -0.5*(n_test*(log(2*pi)+1) + loglik), sigma=sigma, theta1=obj$theta1))
  }
  
  
  obj$optimize_theta1 <- function(obj,f_Xtest, Y_target, Y_test, const_delay=TRUE) {
    # Method which computes the optimal theta_1|0 given Q* and P_1|0=I by using Equation (5.16). It is not how the hyperparameter
    # is chosen for the Reduced Grid Search method, but we allow the user to chose it. 
    
    n_test <- nrow(f_Xtest)
    y_mean <- numeric(n_test)
    y_std <- numeric(n_test)
    
    # Initialization of theta*_1|0 (for Equation (5.16)) and P
    if (is.null(obj$theta1)) {
      obj$theta1 <- matrix(0, nrow = obj$d, ncol = 1)
    }
    obj$theta <- obj$theta1
    if (is.null(obj$P)) {
      obj$P <- diag(obj$d)
    }
    theta_init <- obj$theta
    P_init <- obj$P
    
    # Initialization of matrix C_1|0
    C_matrix <- diag(rep(1, obj$d))
    
    if (const_delay==FALSE){
      
      P_new <- obj$P
      theta <- obj$theta

      # We initialize the numerator and the matrix to invert to obtain theta_1|0 according to the Equation (5.16)
      num <- matrix(0, nrow = obj$d, ncol = 1)
      inv <- matrix(0, nrow = obj$d, ncol = obj$d)
      for (t in 1:n_test) {
        if (hour(Y_target[t]) == 0 && minute(Y_target[t]) == 0) {
          if (t - 96 >= 1) {
            ft <- f_Xtest[(t - 96):(t - 48-1), , drop = FALSE]
            yt <- Y_test[(t - 96):(t - 48-1), , drop = FALSE]
            for (j in 1:nrow(ft)) {
              f_t <- t(ft[j, , drop = FALSE])
              y_t <- yt[j, ]
              P_new <- P_new - (P_new%*%f_t%*%t(f_t)%*%P_new)/as.numeric(obj$sigma^2 + t(f_t) %*% P_new %*% f_t)
              # Iterative computation of matrix C according to Equation (5.17)
              C_matrix <- (diag(obj$d) - (P_new %*% f_t) %*% t(f_t)) %*% C_matrix
              theta <- theta + P_new %*% f_t * as.numeric(y_t - t(theta) %*% f_t) / (obj$sigma^2)
              P_new <- P_new + obj$Q
            }
          }
          f_t <- f_Xtest[t:(t + 47), , drop = FALSE]
          y_mean[t:(t + 47)] <- as.vector(f_t %*% obj$theta)
          # Matrix P_t|t-k is computed according to Equation (5.22)
          P <- P_new+48*obj$Q
          for (idx in t:(t + 47)) {
            ft <- t(f_Xtest[idx, , drop = FALSE])
            P <- P + obj$Q
            # Numerator and the matrix to invert are updated (Equation (5.16))
            num <- num + (Y_test[idx] - y_mean[idx])/as.numeric(t(ft) %*% P %*% ft + obj$sigma^2)* t(C_matrix) %*% ft
            inv <- inv + t(C_matrix) %*% ft %*% t(ft) %*% C_matrix / as.numeric(obj$sigma^2 + t(ft) %*% P %*% ft)
          }
        }
      }
      
    }else{
      
      P <- obj$P
      theta <- obj$theta
      delay <- 96
      
      num <- matrix(0, nrow = obj$d, ncol = 1)
      inv <- matrix(0, nrow = obj$d, ncol = obj$d)
      
      for(t in seq_len(n_test))
      {
        if (t > delay) {
          ft <- t(f_Xtest[t-delay, , drop = FALSE])
          P <- P - tcrossprod(P %*% ft) / (obj$sigma^2 + (t(ft) %*% P %*% ft)[1])
          # Iterative computation of matrix C according to Equation (5.17)
          C_matrix <- (diag(obj$d) - (P %*% ft) %*% t(ft)) %*% C_matrix
          theta <- theta + P %*% ft / (obj$sigma^2) * (Y_test[t-delay] - (t(theta) %*% ft)[1])
          P <- P + obj$Q
        }
        
        ft <- t(f_Xtest[t, , drop = FALSE])
        y_mean[t] <- crossprod(theta, ft)[1]
        y_std[t] <- sqrt(obj$sigma^2 + t(ft) %*% (P + 95*obj$Q)%*% ft)
        
        # Numerator and the matrix to invert are updated (Equation (5.16))
        num <- num + (Y_test[t] - y_mean[t])/y_std[t]^2* t(C_matrix) %*% ft 
        inv <- inv + t(C_matrix) %*% ft %*% t(ft) %*% C_matrix / y_std[t]^2
        
      }
    }
    
    
    # Computation of vector theta1 according to Equation (5.16)
    obj$theta1 <- solve(inv) %*% num
    
    return(obj)
  }

  return(obj)
}

############################################################################

library(R6)

# Class which implements the Reduced Grid Search method for hyperparameter selection for the dynamic Kalman filter. It performs a grid search to       
# maximize the likelihood (Equation (5.26)).
QOptimization <- R6Class(
  "QOptimization",
  public = list(
    X_train = NULL, 
    y_train = NULL,
    gam_model = NULL,
    method = NULL,
    
    # The class has to be initialized with the train set and the trained GAM model
    initialize = function(gam_model, X_train, y_train) {
      self$X_train <- X_train
      self$y_train <- y_train
      self$gam_model <- gam_model
    },


    # This method returns the optimal matrix Q, the optimal sigma, P_1|0=sigma^2*I_d and theta_1|0 by performing a grid search on a single grid 
    # of q* values according to the Reduced Grid Search method. 
    # std_static is an optional input vector which represents the standard deviation of the components of vector theta, obtained on the training set               by the static Kalman filter. It is used for the State Variance Initialization matrix.
    # This method assumes that Q* = (q*)*I for the Identity Intialization matrix or Q = (q*)*diag(std_static^2) for the State Variance Initialization               matrix. The optimal variances are obtained by maximizing the likelihood (Equation (5.26)) on the training set.
    grid_search_reduced_likelihood = function(q_list=c(1,1e-5,1e-10), target="node", std_static=NULL, const_delay=TRUE) {
      max_loglik <- -Inf
      # Since Q*:=Q/sigma^2 and P*:=P/sigma^2, if we set sigma=1 we can continue to use the previous notation (therefore attributes Q and P to indicate Q* and P*)
      sigma <- 1
      optimal_q <- NULL
      
      # For each q* we compute the likelihood. At the end we choose the value of q* maximizing the likelihood
      for (q in q_list) {
        cat(q, "\n")

        n_terms <- 21 #20 terms of GAM Point + Intercept
        
        if (is.null(std_static)) {
          # Identity Initialization matrix
          Q <- diag(rep(q, n_terms))
          P <- diag(rep(1, n_terms))
        } else {
          # State Variance Initialization matrix
          Q <- diag(std_static^2) * q
          P <- diag(rep(1, n_terms))
        }
        # Kalman filter object is initialized according to the current q* 
        kf <- Kalman_Filter(gam_model=self$gam_model, Q=Q, sigma=sigma, P=P)
        # Likelihood computation for the current q*
        res <- kf$predict_reduced_likelihood(kf,self$X_train, self$y_train, target=target, delay=TRUE, const_delay=const_delay)
        loglik <- res$loglik
        print(loglik)
        # We check if the likelihood is improved and update the optimal hyperparameters accordingly
        if (loglik > max_loglik) {
          max_loglik <- loglik
          optimal_sigma <- res$sigma
          optimal_q <- q * optimal_sigma^2
          optimal_theta1 <- res$theta1
        }
      }

      cat("Optimal values of q and sigma: ", optimal_q, optimal_sigma, "\n")
      cat("Maximum likelihood: ", max_loglik, "\n")

      # Returning Q, sigma, P_1|0 and theta_1|0 according to the initialization matrix
      n_terms <- 21
      if (is.null(std_static)) {
        return(list(Q=diag(rep(optimal_q, n_terms)), sigma=optimal_sigma, P=diag(rep(optimal_sigma^2, n_terms)), theta1=optimal_theta1, loglik=max_loglik))
      } else {
        return(list(Q=diag(std_static^2) * optimal_q, sigma=optimal_sigma, P=diag(rep(optimal_sigma^2, n_terms)), theta1=optimal_theta1, loglik=max_loglik))
      }

    }
    
  )
)


