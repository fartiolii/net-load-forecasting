# Kalman Filter and QOptimization classes (similar to the Python classes)

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
    Q = Q            # Variance Covariance matrix of the error on theta
  )

  
  # Function which maps the matrix of features and the coefficients of the GAM model (more than 600) 
  # to its terms (20 terms in the GAM_Point model) which are the only ones to be adapted
  
  obj$f <- function(obj,df,train=FALSE) {
    model_matrix <- predict(obj$GAM_model, newdata=df, type = "lpmatrix")
    coefficients <- coef(obj$GAM_model)
    coeff_list <- list(coefficients)
    term_names <- names(coeff_list[[1]])
    #Term indexes for the GAM_Point model
    indexes <- c(1,2,3,4,5,6,7,8,9,21,23,24,58,435,492,526,560,564,584,593,609,625)
    n_terms <- length(indexes)-1
    
    # Initialization of a matrix to store the feature evaluations
    features_eval <- matrix(0, nrow = nrow(model_matrix), ncol = n_terms)
    coef_vector <- unlist(coef(obj$GAM_model))
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
    
    return(as.matrix(features_eval[,2:ncol(features_eval)]))
    
  }

  # Update of the theta vector with the latest measurement yt
  obj$theta_update <- function(obj,f_Xt, yt) {
    obj$theta <- obj$theta + obj$P %*% f_Xt / (obj$sigma^2) * as.numeric(yt - t(obj$theta) %*% f_Xt)
    return(obj)
  }

  # Update of matrix P 
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

    # Matrix and vector initializations
    if (is.null(obj$theta1)) {
      obj$theta1 <- matrix(0, nrow = d, ncol = 1)
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
  obj$predict <- function(obj, X_test, Y_test = NULL, delay = FALSE) {
    
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
      
        #Delay of 48h is considered
        if (is.null(Y_test)) {
          stop("Target variable must be given for the online version")
        }
        Y_target <- Y_test$targetTime
        Y_test <- as.matrix(Y_test$node) 
        
        f_Xtest <- obj$f(obj,X_test)
        n_test <- nrow(f_Xtest)
        y_mean <- numeric(n_test)
        y_std <- numeric(n_test)
        
        
        theta_init <- obj$theta
        P_init <- obj$P

        for (t in seq_len(n_test)) {
          # We check if the target time of the prediction corresponds to midnight 
          if (hour(Y_target[t]) == 0 && minute(Y_target[t]) == 0) {
            # We check if data from 48h to 24h before is available (condition not true just at the beginning of the dataset)
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
            # prediction that we are doing
            P <- obj$P+48*obj$Q
            for (idx in t:(t + 47)) {
              f_t <- t(f_Xtest[idx, , drop = FALSE])
              P <- P + obj$Q
              y_std[idx] <- sqrt(obj$sigma^2 + t(f_t) %*% P %*% f_t)
            }
          }
        }
    }
    return(list(y_mean = y_mean, y_std = y_std))
  }
  
  
  # The following three methods are used for the optimization of the variances of the dynamic Kalman Filter only
  obj$predict_likelihood <- function(obj,X_test, Y_test, delay=FALSE) {
    Y_target <- Y_test[['targetTime']]
    Y_test <- as.matrix(Y_test[['node']])
    f_Xtest <- obj$f(obj,X_test)
    n_test <- nrow(f_Xtest)
    obj$d <- ncol(f_Xtest)
    y_mean <- numeric(n_test)
    y_std <- numeric(n_test)
    
    # Inizialization of the matrices and of the likelihood
    loglik <- 0
    if (is.null(obj$theta1)) {
      obj$theta1 <- matrix(0, nrow = obj$d, ncol = 1)
    }
    obj$theta <- obj$theta1
    if (is.null(obj$P)) {
      obj$P <- diag(obj$d)
    }
    
    
    theta_init <- obj$theta
    P_init <- obj$P
    for (t in 1:n_test) {
      # We check if the target time of the prediction corresponds to midnight 
      if (hour(Y_target[t]) == 0 && minute(Y_target[t]) == 0) {
        # We check if data from 48h to 24h before is available (condition not true just at the beginning of the dataset)
        if (t - 96 >= 1) {
          # KF is updated with measurements from 48h to 24h before (which correspond to indexes -96 to -48 since we have half an hour data)
          ft <- f_Xtest[(t - 96):(t - 48-1), , drop = FALSE]
          yt <- Y_test[(t - 96):(t - 48-1), , drop = FALSE]
          # For cycle that updates the KF with 48 measurements (which correspond to 24h data)
          for (j in 1:nrow(ft)) {
            f_t <- t(ft[j, , drop = FALSE])
            y_t <- yt[j, ]
            obj <- obj$P_update(obj,f_t, y_t)
            obj <- obj$theta_update(obj,f_t, y_t)
            obj$P <- obj$P + obj$Q
          }
        }
        # Prediction of the next 24h are computed
        ft <- f_Xtest[t:(t + 47), , drop = FALSE]
        y_mean[t:(t + 47)] <- as.vector(ft %*% obj$theta)
        
        # The prediction of the standard deviation requires to sum k-times Q to P, where k represents the number of steps ahead of the                               
        # prediction that we are doing
        P <- obj$P+48*obj$Q
        corr_val <- numeric(48)
        for (idx in t:(t + 47)) {
          ft <- t(f_Xtest[idx, , drop = FALSE])
          P <- P + obj$Q
          y_std[idx] <- sqrt(obj$sigma^2 + t(ft) %*% P %*% ft)
          corr_val[idx - t + 1] <- obj$sigma^2 + t(ft) %*% P %*% ft
        }
        # Incremental computation of the loglikelihood
        err <- (Y_test[t:(t + 47)] - y_mean[t:(t + 47)])^2
        loglik <- loglik + sum(log(corr_val)) + sum(err / corr_val)
      }
    }
    obj$theta <- theta_init
    obj$P <- P_init
    return(list(y_mean = y_mean, y_std = y_std, loglik = -0.5 * loglik))
  }
  
  #Method that computes the likelihood (assuming delayed data) on the training set in input, given Q*.
  #It computes sigma by minimizing the likelihood (given Q*) and therefore it's used for optimization of 
  #variances on a unique grid of q*.
  
  obj$predict_reduced_likelihood <- function(obj,X_test, Y_test, delay=FALSE) {
    Y_target <- Y_test[['targetTime']]
    Y_test <- as.matrix(Y_test[['node']])
    f_Xtest <- obj$f(obj,X_test)
    n_test <- nrow(f_Xtest)
    obj$d <- ncol(f_Xtest)
    y_mean <- numeric(n_test)
    y_std <- numeric(n_test)
    
    # Since Q*=Q/sigma^2 and P*=P/sigma^2 we can continue to use the previous notation (therefore Q and P) if we set sigma=1
    obj$sigma <- 1
    # Optimization of theta_1|0 (prior) which seems to perform worse in most cases than simply initializing to 0 the vector
    #obj <- obj$optimize_theta1(obj, f_Xtest,Y_target,Y_test)
    if (is.null(obj$theta1)) {
      obj$theta1 <- matrix(0, nrow = obj$d, ncol = 1)
    }
    obj$theta <- obj$theta1
    theta_init <- obj$theta
    P_init <- obj$P
    loglik <- 0
    loglik_2 <- 0
    err_norm_sigma <- 0
    
    for (t in 1:n_test) {
      # We check if the target time of the prediction corresponds to midnight 
      if (hour(Y_target[t]) == 0 && minute(Y_target[t]) == 0) {
        # We check if data from 48h to 24h before is available (condition not true just at the beginning of the dataset)
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
        # Predictions of the next 24h are computed
        ft <- f_Xtest[t:(t + 47), , drop = FALSE]
        y_mean[t:(t + 47)] <- as.vector(ft %*% obj$theta)
        
        # The prediction of the standard deviation requires to sum k-times Q to P, where k represents the number of steps ahead of the                               
        # prediction that we are doing
        P <- obj$P+48*obj$Q
        corr_val <- numeric(48)
        for (idx in t:(t + 47)) {
          ft <- t(f_Xtest[idx, , drop = FALSE])
          P <- P + obj$Q
          y_std[idx] <- sqrt(obj$sigma^2 + t(ft) %*% P %*% ft)
          corr_val[idx - t + 1] <- obj$sigma^2 + t(ft) %*% P %*% ft
        }
        err <- (Y_test[t:(t + 47)] - y_mean[t:(t + 47)])^2
        loglik <- loglik + sum(log(corr_val))
        loglik_2 <- loglik_2 + sum(log(corr_val) + err / corr_val)
        err_norm_sigma <- err_norm_sigma + sum(err / corr_val)
      }
    }
    obj$theta <- theta_init
    obj$P <- P_init
    # Estimation of sigma
    sigma <- sqrt(err_norm_sigma / n_test)
    loglik_2 <- n_test * log(sigma^2) + loglik + 1 / sigma^2 * err_norm_sigma
    loglik <- loglik + n_test * log(err_norm_sigma / n_test)
    return(list(y_mean = y_mean, y_std = y_std, loglik = -0.5 * loglik_2, sigma=sigma, theta1=obj$theta1))
  }
  
  obj$optimize_theta1 <- function(obj,f_Xtest, Y_target, Y_test) {
    # Method which computes the optimized theta_1|0 given Q* and P_1|0=I
    
    n_test <- nrow(f_Xtest)
    y_mean <- numeric(n_test)
    if (is.null(obj$theta1)) {
      obj$theta1 <- matrix(0, nrow = obj$d, ncol = 1)
    }
    obj$theta <- obj$theta1
    if (is.null(obj$P)) {
      obj$P <- diag(obj$d)
    }
    theta_init <- obj$theta
    P_init <- obj$P
    C_matrix <- diag(rep(1, obj$d))
    P_new <- obj$P
    theta <- obj$theta
    
    # We initialize the numerator and the matrix to invert to obtain theta_1|0 according to the formula
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
            C_matrix <- (diag(obj$d) - (P_new %*% f_t) %*% t(f_t)) %*% C_matrix
            theta <- theta + P_new %*% f_t * as.numeric(y_t - t(theta) %*% f_t) / (obj$sigma^2)
            P_new <- P_new + obj$Q
          }
        }
        f_t <- f_Xtest[t:(t + 47), , drop = FALSE]
        y_mean[t:(t + 47)] <- as.vector(f_t %*% obj$theta)
        P <- P_new+48*obj$Q
        for (idx in t:(t + 47)) {
          ft <- t(f_Xtest[idx, , drop = FALSE])
          P <- P + obj$Q
          num <- num + (Y_test[idx] - y_mean[idx])/as.numeric(t(ft) %*% P %*% ft + obj$sigma^2)* t(C_matrix) %*% ft 
          inv <- inv + t(C_matrix) %*% ft %*% t(ft) %*% C_matrix / as.numeric(obj$sigma^2 + t(ft) %*% P %*% ft)
        }
      }
    }
    obj$theta <- theta_init
    obj$P <- P_init
    # Computation of vector theta1 according to the formula 
    obj$theta1 <- solve(inv) %*% num
    
    return(obj)
  }

  return(obj)
}

############################################################################

library(R6)

#Class which optimizes the variances (Q and sigma) of the dynamic Kalman Filter (based on GAM) by performing a grid search and by 
#using different metrics (RMSE and Likelihood).
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

    # This method returns the optimal matrix Q, the optimal sigma and P_1|0=sigma^2*I_d by performing a grid search on a grid 
    # of q values and on a grid of sigma values. 
    # std_static is an optional input vector which represents the standard deviation of each parameter in the vector theta, obtained on the training set           by the Static Kalman Filter. 
    # This method assumes that Q = q*I_d where I_d is an identity matrix if std_static is None otherwise Q = q*diag(std_static^2).
    # The optimal matrices can be obtained by either minimizing the RMSE or by maximizing the likelihood on the training set.
    grid_search = function(q_list=c(1,1e-5,1e-10), sigma_list=c(1,0.5,0.1), method="RMSE", std_static=NULL) {
      optimal_q <- NULL
      optimal_sigma <- NULL
      self$method <- method
      
      if (self$method == "RMSE") {
        min_RMSE <- Inf
      } else if (self$method == "Likelihood") {
        max_loglik <- -Inf
      } else {
        cat("Method not supported.\n")
        next
      }

      # For each q and for each sigma we compute the RMSE or the likelihood and choose the 
      # combination of (q, sigma) minimizing the RMSE or maximing the Likelihood
      for (q in q_list) {
        if (self$method == "RMSE") {
          q_metric <- Inf
        } else if (self$method == "Likelihood") {
          q_metric <- -Inf
        }
        for (sigma in sigma_list) {
          cat(q, sigma, "\n")
          
          # Initialization of the matrices and the KF according to the GAM model 
          n_terms <- 20
          if (is.null(std_static)) {
            Q <- diag(rep(q, n_terms))
            P <- diag(rep(1, n_terms))
          } else {
            Q <- diag(std_static^2) * q
            P <- diag(rep(1, n_terms))
          }
          kf <- Kalman_Filter(gam_model=self$gam_model, Q=Q, sigma=sigma, P=P)


          if (self$method == "RMSE") {
            # Fit the KF on the train set and compute predictions to compute the RMSE on the train set
            kf <- kf$fit(kf, self$X_train, self$y_train[['node']])
            res <- kf$predict(kf, self$X_train, self$y_train, delay=TRUE)
            new_RMSE <- self$RMSE(self$y_train[['node']], res$y_mean)
            print(new_RMSE)

            if (new_RMSE < min_RMSE) {
              min_RMSE <- new_RMSE
              optimal_q <- q
              optimal_sigma <- sigma
            }
            # Sigma in the list are ordered decreasingly. If decreasing sigma increases the RMSE we skip to the 
            # next q (decreasing further won't improve the metric)
            if (new_RMSE > q_metric) {
              break
            }
            q_metric <- new_RMSE
            sigma_list <- sigma_list[which(sigma_list == sigma):length(sigma_list)]

          } else if (self$method == "Likelihood") {

            y_hat <- kf$predict_likelihood(kf,self$X_train, self$y_train,delay=True)
            loglik <- y_hat$loglik
            print(loglik)
            if (loglik > max_loglik){
              max_loglik <- loglik
              optimal_q <- q
              optimal_sigma <- sigma
            }
            # Sigma in the list are ordered decreasingly. If decreasing sigma decreases the Likelihood we skip to the 
            # next q (decreasing further won't improve the metric)
            if (loglik < q_metric) {
              break
            }
            q_metric <- loglik
            sigma_list <- sigma_list[which(sigma_list == sigma):length(sigma_list)]

          }
        }
      }
      cat("Optimal values q and sigma: ", optimal_q, optimal_sigma, "\n")


      n_terms <- 20
      if (is.null(std_static)) {
        return(list(Q=diag(rep(optimal_q, n_terms)), sigma=optimal_sigma, P=diag(rep(optimal_sigma^2, n_terms))))
      } else {
        return(list(Q=diag(std_static^2) * optimal_q, sigma=optimal_sigma, P=diag(rep(optimal_sigma^2, n_terms))))
      }

    },

    # This method returns the optimal matrix Q, the optimal sigma, P_1|0=sigma^2*I_d and theta_1|0 by performing a grid search on a single grid 
    # of q* values. 
    # std_static is an optional input vector which represents the standard deviation of each parameter in the vector theta, obtained on the training set           
    # by the Static Kalman Filter. 
    # This method assumes that Q* = (q*)*I_d where I_d is an identity matrix if std_static is None otherwise Q = (q*)*diag(std_static^2).
    # The optimal matrices are obtained by maximizing the likelihood on the training set.
    grid_search_reduced_likelihood = function(q_list=c(1,1e-5,1e-10), std_static=NULL) {
      max_loglik <- -Inf
      sigma <- 1
      optimal_q <- NULL

      for (q in q_list) {
        cat(q, "\n")

        n_terms <- 20
        if (is.null(std_static)) {
          Q <- diag(rep(q, n_terms))
          P <- diag(rep(1, n_terms))
        } else {
          Q <- diag(std_static^2) * q
          P <- diag(rep(1, n_terms))
        }
        kf <- Kalman_Filter(gam_model=self$gam_model, Q=Q, sigma=sigma, P=P)

        res <- kf$predict_reduced_likelihood(kf,self$X_train, self$y_train, delay=TRUE)
        loglik <- res$loglik
        print(loglik)
        if (loglik > max_loglik) {
          max_loglik <- loglik
          optimal_sigma <- res$sigma
          optimal_q <- q * optimal_sigma^2
          optimal_theta1 <- res$theta1
        }
      }

      cat("Optimal values q and sigma: ", optimal_q, optimal_sigma, "\n")

      n_terms <- 20
      if (is.null(std_static)) {
        return(list(Q=diag(rep(optimal_q, n_terms)), sigma=optimal_sigma, P=diag(rep(optimal_sigma^2, n_terms)), theta1=optimal_theta1))
      } else {
        return(list(Q=diag(std_static^2) * optimal_q, sigma=optimal_sigma, P=diag(rep(optimal_sigma^2, n_terms)), theta1=optimal_theta1))
      }

    },

    RMSE = function(y, yhat) {
      return(sqrt(sum((y - yhat)^2) / length(y)))
    }

  )
)

MAPE = function(y, yhat){
  return(sum(abs(y - yhat)/y) / length(y)) 
}
  
RMSE = function(y, yhat) {
  return(sqrt(sum((y - yhat)^2) / length(y)))
}

MAE = function(y, yhat) {
  return(sum(abs(y - yhat)) / length(y))
}


