import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.optimize import minimize
import time
from scipy.stats import norm
import copy
import warnings


######## Kalman Filter Class

class Kalman_Filter(object):
    """This class represents a Kalman Filter model which adapts the coefficients of either a GAM model or a linear regression model.
    
    INPUT:
    - GAM: bool, True if the coefficients to be adapted are the ones of a GAM model, False if the coefficients are the ones of a linear regression model
    - gam_model: the fitted GAM model object of the pygam package
    - params: used only for linear regression models. It contains the optimal linear regression parameters required to initialize the state theta.
    - theta1: Initial vector of parameters theta to adapt (theta_1|0)
    - P: Initial variance-covariance matrix of the vector theta (P_1|0)
    - sigma: Variance of the measurement error on y
    - Q: Variance-Covariance matrix of the error on the state theta
    """
    def __init__(self, gam_model=None, GAM=True, params=None,theta1=None, P=None, sigma=None, Q=None):
        self.GAM = GAM
        self.GAM_model = gam_model
        self.static = True if Q is None else False # Bool that indicates if we are considering the static or the dynamic Kalman filter
        self.theta1 = theta1 
        self.theta = self.theta1  # State vector (contains the parameters to adapt)
        self.theta_mat = None     # Matrix that contains all the different theta vectors computed progressively on the train set 
        self.P = P                # variance-covariance matrix of theta 
        self.sigma = sigma        # variance of the target variable measurement
        self.Q = Q                # variance-covariance matrix of the error on theta 
        self.params = None if params is None else np.array(params)
        
        
    def f(self, df, train=False):
        """Method that, given the feature matrix n_obs*n_features, returns:
           - a matrix n_obs*n_groups if the number of parameters to adapt is lower than the number of features used (in the case of GAM)
           - the matrix itself if all parameters are adapted in the linear regression case"""
        
        if self.GAM == True:
            n_terms = len(list(self.GAM_model.terms)) #number of terms (groups) i.e. actual features in the model (GAM fits multiple paramters for each feature)
            features_eval = np.zeros((df.shape[0],n_terms))
            it = 0
            for f in range(n_terms):
                feature_mat = self.GAM_model._modelmat(np.array(df), term=f) # matrix evaluation for this term
                n_coeff_feature = feature_mat.shape[1] # number of parameters for this term
                features_eval[:, f] = feature_mat@self.GAM_model.coef_[it:it+n_coeff_feature] #summing the contributions for this term
                it += n_coeff_feature
                
            return features_eval
        
        else:
            # For linear regression models, the model matrix itself is returned 
            return df.copy() 
        
        
        
    def theta_update(self, f_Xt, yt):
        # Implementation of Equation (5.8) (first equation)
        # update of theta with new data yt
        self.theta += self.P@f_Xt / (self.sigma**2) * (yt - self.theta.T@f_Xt)

        
    def P_update(self, f_Xt, yt):
        # Implementation of Equation (5.7) 
        # update of matrix P
        self.P += -(self.P@f_Xt@f_Xt.T@self.P)/(f_Xt.T@self.P@f_Xt + self.sigma**2)

        
    def fit(self, X_train, Y_train):
        """This method fits the Kalman filter with the model matrix X_train in input and the target variable Y_train in input """
        Yt = np.array(Y_train)
        f_Xt = self.f(X_train) # Model matrix of the test set, according to which parameters to adapt (depending on GAM or linear model)
        n, d = f_Xt.shape
        self.d = d # Length of the vector theta (that contains the parameters to adapt)
        
        # Parameters are initialized 
        if not self.params is None:
            self.theta1 = self.params.reshape(-1,1) # For linear regression models, we initialize the state with the optimal parameters 
        if self.theta1 is None:
            self.theta1 = np.ones((self.d,1)) # For GAM models, we initialize the state with a vector of ones
        self.theta = self.theta1.copy()
        
        # For the dynamic Kalman filter, hyperparameters are provided in input to the object, here we initialize by default the ones of the static case
        if self.P is None:
            self.P = np.eye(self.d) # Static Kalman filter case
        if self.Q is None:
            self.Q = np.zeros(self.d) # Static Kalman filter case
        if self.sigma is None:
            self.sigma = 1 # Static Kalman filter case
            
        self.theta_mat = np.zeros((n,self.d))
        
        # This for cycle uses all data in the training set to update theta and its variance-covariance matrix P
        for t in range(n):
            self.theta_mat[t, :] = self.theta.flatten() # Save current theta vector in the matrix
            self.P_update(f_Xt[t,:].reshape(-1,1), Yt[t]) # Obtain P_t|t (Equation (5.7))
            self.theta_update(f_Xt[t,:].reshape(-1,1), Yt[t]) # Update theta (Equation (5.8), first equation)
            self.P += self.Q #Obtain P_t+1|t (Equation (5.8))
        
            
    
    def predict(self, X_test, Y_test, delay=False, const_delay=True):
        """After the Kalman filter is trained with the fit method, this method computes the predictions on the test set in input
           by either using the measurement from 30 minutes before (delay=False) or by considering the fact that, every day, data is 
           obtained at midnight of the day after (delay=True)
           const_delay: Bool, True if we incorporate past data in the model with a constant delay of 48h, False if instead we take into 
                        account the fact that at midnight of each day, we get all data regarding the past 48 to 24 hours 
        """
        if self.theta_mat is None:
            print("Kalman Filter must be trained first")
        if not delay:
            # In this case, to predict the value of the next 30 min we are using the measurements up to 30 min before. This is not the setting adopted in the             thesis but we allow the user to select it.
            Y_test = np.array(Y_test)
            f_Xtest = self.f(X_test)
            n_test = f_Xtest.shape[0]
            y_mean = np.zeros(n_test,) # vector of mean predictions is initialized
            y_std = np.zeros(n_test,)  # vector of the standard deviation of the predictions is initialized
            self.theta_mat_test = np.zeros((n_test,self.d)) # Matrix in which all vectors theta computed on the test set are saved

            theta_init = self.theta.copy() # Initial value of theta is saved (as the prediction should not change theta/P)
            P_init = self.P.copy()         # Initial value of P is saved (as the prediction should not change theta/P)
            
            # For cycle that computes predictions for all the test set in input
            for t in range(n_test):
                ft = f_Xtest[t,:].reshape(-1,1)
                y_mean[t] = self.theta.T@ft                             # Mean prediction
                y_std[t] = np.sqrt(self.sigma**2 + ft.T@self.P@ft)      # Standard deviation of the prediction
                self.theta_mat_test[t, :] = self.theta.flatten()        # Save current theta vector in the matrix
                self.P_update(ft, Y_test[t])                            # Compute P_t|t (Equation (5.7))
                self.theta_update(ft, Y_test[t])                        # Update theta (Equation (5.8), first equation)
                self.P += self.Q                                        # Compute P_t+1|t (Equation (5.8), second equation)

            self.theta = theta_init        # Initial value of theta is copied back in the state vector of the object KF
            self.P = P_init                # Initial value of P is copied back in the variance-covariance matrix of the object KF
        else:
            Y_target = Y_test['targetTime'] # Target time of the prediction to make 
            Y_test = np.array(Y_test[Y_test.columns[1]]) # Target value to predict given in input (can be net-load, 7 days differenced net-load etc.)
            f_Xtest = self.f(X_test) # Model matrix of the test set, according to which parameters to adapt (depending on GAM or linear model)
            n_test = f_Xtest.shape[0]
            y_mean = np.zeros(n_test,) # vector of mean predictions is initialized
            y_std = np.zeros(n_test,)  # vector of the standard deviation of the predictions is initialized
            
            if const_delay is False:
                # In this case, to predict the value of every 30 min throughout a single day, we are using all measurements up to the midnight before                       # (which are measurements that go from 48h to 24h before the midnight). e.g. the prediction of values from 25/01 at 00:00 to 25/01 at 23:30                 # occurs the 25/01 at 00:00 with new measurement data incorporated in the model which goes from 23/01 at 00:00 to 23/01 at 23:30. This is not the             setting considered for the results reported in the thesis but we allow the user to select it.
   
                theta_init = self.theta.copy() # Initial value of theta is saved (as the prediction should not change theta/P)
                P_init = self.P.copy()         # Initial value of P is saved (as the prediction should not change theta/P)

                for t in range(n_test):
                    # We check if the target time of the prediction corresponds to midnight 
                    if Y_target.iloc[t].hour == 0 and Y_target.iloc[t].minute == 0:
                        # We check if data from 48h to 24h before is available (just a condition to check at initialization)
                        if t-96 >= 0:
                            # KF is updated with measurements from 48h to 24h before (which correspond to indexes -96 to -48 since we have half an hour                                 # data)
                            ft = f_Xtest[t-96:t-48,:]
                            yt = Y_test[t-96:t-48]
                            # For cycle that updates the KF with 48 measurements (which correspond to 24h data)
                            for j in range(ft.shape[0]):
                                f_t = ft[j,:].reshape(-1,1)
                                y_t = yt[j]
                                self.P_update(f_t,y_t)
                                self.theta_update(f_t,y_t)
                                self.P += self.Q

                        # Predictions for the next 24h are computed 
                        ft = f_Xtest[t:t+48,:]
                        y_mean[t:t+48] = (ft@self.theta).flatten()

                        # The prediction of the standard deviation requires to sum k-times Q to P, where k represents the number of steps ahead of the                               # prediction that we are doing. See Section (5.5)
                        P = self.P + 48*self.Q # Equation (5.22)
                        for idx in range(t, t+48):
                            f_t = f_Xtest[idx,:].reshape(-1,1)
                            P += self.Q
                            y_std[idx] = np.sqrt(self.sigma**2 + f_t.T@P@f_t)


                self.theta = theta_init      # Initial value of theta is copied back in the state vector of the object KF
                self.P = P_init              # Initial value of P is copied back in the variance-covariance matrix of the object KF

            else:
                # In this case we use the data to update the model as if data is constantly arriving with a 48h delay throughout the day. At each time                       # instant we update the model with the measurement of 48h before and then predict the new value of yt.
                # This is the setting considered for the results reported in the thesis.
                
                theta_init = self.theta.copy() # Initial value of theta is saved (as the prediction should not change theta/P)
                P_init = self.P.copy()         # Initial value of P is saved (as the prediction should not change theta/P)

                # For cycle that computes predictions for all the test set in input
                for t in range(n_test):
                    if t >= 96:
                        # Model update with the measurement from 48h before
                        ft = f_Xtest[t-96,:].reshape(-1,1)
                        self.P_update(ft, Y_test[t-96])                            # Compute P_t|t (Equation (5.7))
                        self.theta_update(ft, Y_test[t-96])                        # Update theta (Equation (5.8), first equation)
                        self.P += self.Q                                           # Compute P_t+1|t (Equation (5.8), second equation)
                    
                    # Prediction
                    ft = f_Xtest[t,:].reshape(-1,1)
                    y_mean[t] = self.theta.T@ft                                         # Mean prediction
                    y_std[t] = np.sqrt(self.sigma**2 + ft.T@(self.P+95*self.Q)@ft)      # Standard deviation of the prediction, matrix P_t|t-k is computed as in Equation (5.22)


                self.theta = theta_init        # Initial value of theta is copied back in the state vector of the object KF
                self.P = P_init                # Initial value of P is copied back in the variance-covariance matrix of the object KF
                
        return y_mean, y_std
    
    
    
    
    # The following two methods are used for the Reduced Grid Search hyperparameter selection method for the dynamic Kalman Filter only
    
    def compute_reduced_likelihood(self, X_test, Y_test, const_delay=True, fit_theta1=False, params=None):
        """Method that computes the likelihood (assuming delay=True) as in Equation (5.26) on the training set in input, given Q*,
        and evaluates sigma (Equation (5.14))."""
        
        Y_target = Y_test['targetTime'] # Target time of the prediction to make 
        Y_test = np.array(Y_test[Y_test.columns[1]]) # Target value to predict given in input (can be net-load, 7 days differences etc.)
        f_Xtest = self.f(X_test) # Model matrix of the test set, according to which parameters to adapt (depending on GAM or linear model)
        n_test, d = f_Xtest.shape
        self.d = d
        y_mean = np.zeros(n_test,) # vector of mean predictions is initialized
        y_std = np.zeros(n_test,)  # vector of the standard deviation of the predictions is initialized
        
        # Since Q*:=Q/sigma^2 and P*:=P/sigma^2, if we set sigma=1 we can continue to use the previous notation (therefore attributes Q and P to indicate Q* and P*) 
        self.sigma = 1
        if fit_theta1:
            # Optimization of theta_1|0 (prior) (Formula (5.16)). It is not the optimal theta_1|0 chosen for the Reduced Grid Search method but
            # we allow the user to choose it. 
            self.theta1 = self.optimize_theta1(f_Xtest, Y_target, Y_test, const_delay=const_delay)
        else:
            if not self.params is None:
                # For linear regression models, the initial state coincides with the optimized parameters obtained from the regression.
                self.theta1 = self.params.reshape(-1,1) 
            if self.theta1 is None:
                self.theta1 = np.zeros((self.d,1))
        self.theta = self.theta1.copy()
        
        if const_delay is False:
        
            theta_init = self.theta.copy()
            P_init = self.P.copy()

            # Computation of the likelihood (Equation (5.26))
            loglik = 0
            err_norm_sigma = 0

            for t in range(n_test):
                # We check if the target time of the prediction corresponds to midnight 
                if Y_target.iloc[t].hour == 0 and Y_target.iloc[t].minute == 0:
                    # We check if data from 48h to 24h before is available (just a condition to check at initialization)
                    if t-96 >= 0:
                        # KF is updated with measurements from 48h to 24h before (which correspond to indexes -96 to -48 since we have half an hour data)
                        ft = f_Xtest[t-96:t-48,:]
                        yt = Y_test[t-96:t-48]
                        # For cycle that updates the KF with 48 measurements (which correspond to 24h data)
                        for j in range(ft.shape[0]):
                            f_t = ft[j,:].reshape(-1,1)
                            y_t = yt[j]
                            self.P_update(f_t,y_t)     # Compute P_t|t (Equation (5.7))
                            self.theta_update(f_t,y_t) # Update theta (Equation (5.8), first equation)
                            self.P += self.Q           # Compute P_t+1|t (Equation (5.8), second equation)

                    # Predictions of the next 24h are computed 
                    ft = f_Xtest[t:t+48,:]
                    y_mean[t:t+48] = (ft@self.theta).flatten()

                    # The prediction of the standard deviation requires to sum k-times Q to P, where k represents the number of steps ahead of the                               # prediction that we are doing. See Section 5.5.
                    P = self.P.copy()+ 48*self.Q        # Equation (5.22)
                    corr_val = np.zeros((48,))
                    i=0
                    for idx in range(t, t+48):
                        ft = f_Xtest[idx,:].reshape(-1,1)
                        P += self.Q

                        corr_val[i] = self.sigma**2 + ft.T@P@ft
                        i += 1
                    
                    # Terms required for the computation of the likelihood
                    err = (Y_test[t:t+48] - y_mean[t:t+48])**2
                    loglik += np.sum(np.log(corr_val)) 
                    err_norm_sigma += np.sum(err/corr_val)   


            self.theta = theta_init
            self.P = P_init
        else:
            # In this case we use the data to update the model as if data is constantly arriving with a 48h delay throughout the day. At each time                       # instant we update the model with the measurement of 48h before and then predict the new value of yt.
            # This setting is chosen for the results reported in the thesis.
        
            theta_init = self.theta.copy() # Initial value of theta is saved (as the prediction should not change theta/P)
            P_init = self.P.copy()         # Initial value of P is saved (as the prediction should not change theta/P)

            loglik = 0
            err_norm_sigma = 0

            # For cycle that computes predictions for all the test set in input
            for t in range(n_test):
                if t >= 96:
                    ft = f_Xtest[t-96,:].reshape(-1,1)
                    self.P_update(ft, Y_test[t-96])                            # Compute P_t|t (Equation (5.7))
                    self.theta_update(ft, Y_test[t-96])                        # Update theta (Equation (5.8), first equation)
                    self.P += self.Q                                           # Compute P_t+1|t (Equation (5.8), second equation)

                ft = f_Xtest[t,:].reshape(-1,1)
                y_mean[t] = self.theta.T@ft                             # Mean prediction
                # Standard deviation of the prediction, matrix P_t|t-k is computed as in Equation (5.22)
                y_std[t] = np.sqrt(self.sigma**2 + ft.T@(self.P+95*self.Q)@ft) if self.sigma**2 + ft.T@(self.P+95*self.Q)@ft > 0 else 1
                
                # Terms required for the computation of the likelihood.
                err = (Y_test[t] - y_mean[t])**2
                loglik += np.log(y_std[t]**2) 
                err_norm_sigma += err/y_std[t]**2


            self.theta = theta_init        # Initial value of theta is copied back in the state vector of the object KF
            self.P = P_init                # Initial value of P is copied back in the variance-covariance matrix of the object KF


        # Computation of sigma (Equation (5.14))
        self.sigma = np.sqrt(err_norm_sigma/n_test)

        loglik += n_test*np.log(err_norm_sigma/n_test)
        
        return y_mean, y_std, -0.5*(n_test*(np.log(2*np.pi)+1) + loglik)
    
    
    def optimize_theta1(self, f_Xtest, Y_target, Y_test, const_delay=True):
        """Method which computes the optimal theta_1|0 given Q* and P_1|0=I by using Equation (5.16). It is not how the hyperparameter
           is chosen for the Reduced Grid Search method, but we allow the user to chose it. """
        n_test, d = f_Xtest.shape
        y_mean = np.zeros(n_test,)
        
        # Initialization of theta*_1|0 (for Equation (5.16)) and P
        self.theta1 = np.zeros((d,1))
        self.theta = self.theta1.copy()
        if self.P is None:
            self.P = np.eye(d) 
        
        # Initialization of matrix C_1|0
        C_matrix = np.diag(np.ones(d))
        P_new = self.P.copy()
        theta = self.theta.copy()
        
        # We initialize the numerator and the matrix to invert to obtain theta_1|0 according to the Equation (5.16)
        num = np.zeros((d,1))
        inv = np.zeros((d,d))
        
        if const_delay is False:
            for t in range(n_test):
                # We check if the target time of the prediction corresponds to midnight 
                if Y_target.iloc[t].hour == 0 and Y_target.iloc[t].minute == 0:
                    # We check if data from 48h to 24h before is available (just a condition to check at initialization)
                    if t-96 >= 0:
                        # KF is updated with measurements from 48h to 24h before (which correspond to indexes -96 to -48 since we have half an hour data)
                        ft = f_Xtest[t-96:t-48,:]
                        yt = Y_test[t-96:t-48]
                        # For cycle that updates the KF with 48 measurements (which correspond to 24h data) and computes matrix C
                        for j in range(ft.shape[0]):
                            f_t = ft[j,:].reshape(-1,1)
                            y_t = yt[j]

                            P_new += -(P_new@f_t@f_t.T@P_new)/(f_t.T@P_new@f_t + self.sigma**2)
                            # Iterative computation of matrix C according to Equation (5.17)
                            C_matrix = (np.eye(d) - (P_new@f_t)@f_t.T)@C_matrix
                            theta += P_new@f_t * (y_t - theta.T@f_t)/ (self.sigma**2) 
                            P_new += self.Q

                    # Predictions of the next 24h are computed 
                    ft = f_Xtest[t:t+48,:]
                    y_mean[t:t+48] = (ft@self.theta).flatten()
                    
                    # Matrix P_t|t-k is computed according to Equation (5.22)
                    P = P_new.copy() + 48*self.Q
                    for idx in range(t, t+48):
                        ft = f_Xtest[idx,:].reshape(-1,1)
                        P += self.Q
                        
                        # Numerator and the matrix to invert are updated (Equation (5.16))
                        num += (Y_test[idx] - y_mean[idx])/(self.sigma**2 + ft.T@P@ft)*C_matrix.T@ft
                        inv += C_matrix.T@ft@ft.T@C_matrix/(self.sigma**2 + ft.T@P@ft)
                        
        else:
            for t in range(n_test):
                if t >= 96:
                    ft = f_Xtest[t-96,:].reshape(-1,1)
                    yt = Y_test[t-96]
                    
                    P_new += -(P_new@ft@ft.T@P_new)/(ft.T@P_new@ft + self.sigma**2)
                    # Iterative computation of matrix C according to Equation (5.17)
                    C_matrix = (np.eye(d) - (P_new@ft)@ft.T)@C_matrix
                    theta += P_new@ft * (yt - theta.T@ft)/ (self.sigma**2) 
                    P_new += self.Q
  
                ft = f_Xtest[t,:].reshape(-1,1)
                y_mean[t] = self.theta.T@ft      # Mean prediction
                
                # Numerator and the matrix to invert are updated (Equation (5.16))
                num += (Y_test[t] - y_mean[t])/(self.sigma**2 + ft.T@(P_new+95*self.Q)@ft)*C_matrix.T@ft
                inv += C_matrix.T@ft@ft.T@C_matrix/(self.sigma**2 + ft.T@(P_new+95*self.Q)@ft)
                
                 
        # Computation of vector theta1 according to Equation (5.16)
        theta1 = np.linalg.inv(inv)@num
        return theta1
    
###################### Dynamic Kalman filter hyperparameters optimization ##############################
class Q_optimization(object):
    """Class which implements the Reduced Grid Search method for hyperparameter selection for the dynamic Kalman filter. It performs a grid search to       maximize the likelihood (Equation (5.26))."""
    def __init__(self, X_train, y_train, gam_model=None, GAM=False, params=None):
        self.X_train = X_train # Features of the training set
        self.y_train = y_train # Target variable of the training set
        # Same as parameters of the Kalman Filter class
        self.GAM = GAM        
        self.gam_model = gam_model
        self.params = None if params is None else np.array(params)
        
     
    def grid_search_reduced_likelihood(self, q_list=[1,1e-5,1e-10], std_static=None, const_delay=True,fit_theta1=False):
        """This method returns the optimal matrix Q, the optimal sigma, P_1|0=sigma^2*I_d and theta_1|0 by performing a grid search on a single grid 
        of q* values according to the Reduced Grid Search method. 
        std_static is an optional input vector which represents the standard deviation of the components of vector theta, obtained on the training set               by the static Kalman filter. It is used for the State Variance Initialization matrix.
        This method assumes that Q* = (q*)*I for the Identity Intialization matrix or Q = (q*)*diag(std_static^2) for the State Variance Initialization               matrix. The optimal variances are obtained by maximizing the likelihood (Equation (5.26)) on the training set."""
        
        # Since Q*:=Q/sigma^2 and P*:=P/sigma^2, if we set sigma=1 we can continue to use the previous notation (therefore attributes Q and P to indicate Q* and P*)
        max_loglik = -np.inf
        sigma = 1
        optimal_q = None
        
        # For each q* we compute the likelihood. At the end we choose the value of q* maximizing the likelihood
        for q in q_list:
            print(q)
               
            # Initialization of the KF according to the model (GAM or linear regression) and to the current q*        
            kf = self.get_kf(q=q, std_static=std_static)
            # Likelihood computation for the current q*
            y_hat, _, loglik = kf.compute_reduced_likelihood(self.X_train, self.y_train, const_delay=const_delay, fit_theta1=fit_theta1,                                                                            params=self.params)
            
            print(loglik)
            # We check if the likelihood is improved and update the optimal hyperparameters accordingly
            if loglik > max_loglik:
                max_loglik = loglik
                optimal_sigma = kf.sigma   
                optimal_q = q*optimal_sigma**2
                optimal_theta1 = kf.theta1

        print("Optimal q: ", optimal_q)
        print("Optimal sigma: ", optimal_sigma)
        print("Maximum likelihood achieved: ", max_loglik)
        
         # Returning Q, sigma, P_1|0 and theta_1|0 according to the model (GAM or linear regression) and to the initialization matrix
        if self.GAM:
            n_terms = len(list(self.gam_model.terms))
            if std_static is None:
                return np.eye(n_terms)*optimal_q, optimal_sigma, np.eye(n_terms)*optimal_sigma**2, optimal_theta1
            else:
                return np.diag(std_static**2)*optimal_q, optimal_sigma, np.eye(n_terms)*optimal_sigma**2, optimal_theta1
        else:
            if std_static is None:
                return np.eye(np.shape(self.X_train)[1])*optimal_q, optimal_sigma, np.eye(np.shape(self.X_train)[1])*optimal_sigma**2,optimal_theta1
            else:
                return np.diag(std_static**2)*optimal_q, optimal_sigma, np.eye(np.shape(self.X_train)[1])*optimal_sigma**2, optimal_theta1
                        
   
    def get_kf(self, q=None, std_static=None):
        """Method which returns a Kalman filter object by initializing the hyperparameters based on the model (GAM or linear regression) and the       initialization matrix for the Reduced Grid Search Method."""
        sigma = 1
        
        if self.GAM:
            n_terms = len(list(self.gam_model.terms))
            if std_static is None:
                # Identity Initialization matrix
                Q = np.eye(n_terms)*q
            else:
                # State Variance Initialization matrix
                Q = np.diag(std_static**2)*q
            
            P = np.eye(n_terms)

            kf = Kalman_Filter(gam_model=self.gam_model, Q=Q, sigma=sigma, P=P)
        else:
            if std_static is None:
                # Identity Initialization matrix
                Q = np.eye(np.shape(self.X_train)[1])*q
            else:
                # State Variance Initialization matrix
                Q = np.diag(std_static**2)*q

            P = np.eye(np.shape(self.X_train)[1])

            kf = Kalman_Filter(GAM=False, Q=Q, sigma=sigma, P=P)
        
        return kf
                
    
######## Persistence Benchmark
class Persistence_Benchmark(object):
    """This class implements the persistence benchmark (n of days in input) used to compare the predictions of every other model.
       The prediction for each target time coincides with the actual value of the target variable the given number of days before."""
    def __init__(self, Y_df):
        self.Y_df = Y_df
    
    def predict(self, targetTime, delay_days=7):
        y_pred = np.zeros((len(targetTime),))
        
        for t in range(len(targetTime)):
            # At midnight the predictions are computed for the following 24h (i.e. 48 data points)
            if targetTime.iloc[t].hour == 0 and targetTime.iloc[t].minute == 0:
                dt = targetTime.iloc[t] - pd.Timedelta(days=delay_days) # The predictions coincide with the values x days before
                idx = (self.Y_df['targetTime'] == dt).idxmax() 
                y_pred[t:t+48] = self.Y_df['node'].iloc[idx:idx+48]
        return y_pred
        
    
######## Mean Performance Evaluation metrics

def RMSE(y, yhat):
    """This method computes the Root Mean Square Error of the prediction yhat with respect to the vector y """
    return np.sqrt(np.sum((y-yhat)**2)/len(y))

def MAE(y, yhat):
    """This method computes the Mean Absolute Error of the prediction yhat with respect to the vector y """
    return 1/len(y)*np.sum(np.abs(y-yhat))

def MAPE(y, yhat):
    """This method computes the Mean Absolute Percentage Error of the prediction yhat with respect to the vector y """
    return 1/len(y)*np.sum(np.abs((y-yhat)/y))

def nRMSE(y_mat, yhat_mat):
    "y_mat and yhat_mat are matrices of dimension nxn_regions"
    n, n_regions = y_mat.shape
    partial_sum = 0
    
    for i in range(n_regions):
        partial_sum += np.sum((y_mat[:,i]-yhat_mat[:, i])**2)/np.sum((y_mat[:,i]-y_mat[:, i].mean())**2)
    return np.sqrt(partial_sum/n_regions)

def nMAE(y_mat, yhat_mat):
    "y_mat and yhat_mat are matrices of dimension nxn_regions"
    n, n_regions = y_mat.shape
    partial_sum = 0
    
    for i in range(n_regions):
        partial_sum += np.sum(np.abs(y_mat[:,i]-yhat_mat[:, i]))/np.sum(np.abs(y_mat[:,i]-y_mat[:, i].mean()))
    return partial_sum/n_regions
        

    
######## Iterative Grid Search for Dynamic Kalman Filter (viking package in R)

# The following methods are the translation to Python of the viking methods in R

def parameters_star(X, y, Qstar, p1=0):
    n, d = X.shape
    thetastar = np.zeros((d, 1))
    Pstar = np.diag(np.ones(d) * p1)
    C = np.diag(np.ones(d))
    thetastar_arr = np.zeros((n + 1, d))
    Pstar_arr = np.zeros((n + 1, d, d))
    C_arr = np.zeros((n + 1, d, d))
    C_arr[0, :, :] = C

    for t in range(n):
        Xt = X[t, :].reshape(-1,1)
        err = y[t] - np.dot(thetastar.T, Xt)[0]
        inv = 1 / (1 + np.dot(Xt.T, np.dot(Pstar, Xt))[0])

        thetastar_new = thetastar + np.dot(Pstar, Xt) * inv * err
        Pstar_new = Pstar + Qstar - np.dot(Pstar, Xt) @ (Xt.T @ Pstar) * inv
        C_new = (np.eye(d) - np.dot(np.dot(Pstar, Xt), Xt.T) * inv) @ C
        

        thetastar = thetastar_new
        Pstar = Pstar_new
        C = C_new

        thetastar_arr[t + 1, :] = thetastar.flatten()
        Pstar_arr[t + 1, :, :] = Pstar
        C_arr[t + 1, :, :] = C

    return {"thetastar_arr": thetastar_arr, "Pstar_arr": Pstar_arr, "C_arr": C_arr}


def get_theta1(X, y, par, Qstar, use, mode='gaussian'):
    n, d = X.shape
    A = np.diag(np.zeros(d))
    b = np.zeros((d, 1))

    for t in range(n):
        Xt = X[t, :].reshape(-1, 1)
        err = y[t] - np.dot(par['thetastar_arr'][use[t], :], Xt)[0]

        inv = 1
        if mode == 'gaussian':
            Pstar_t = par['Pstar_arr'][use[t], :, :] + max(0, t - use[t]) * Qstar
            inv = 1 / (1 + np.dot(Xt.T, np.dot(Pstar_t, Xt))[0])

        A += np.dot(par['C_arr'][use[t], :, :].T, Xt)@(par['C_arr'][use[t], :, :].T@Xt).T * inv
        b += np.dot(par['C_arr'][use[t], :, :].T, Xt) * err * inv
    
    return np.linalg.solve(A, b)


def get_sig(X, y, par, Qstar, use):
    n, d = X.shape
    theta1 = get_theta1(X, y, par, Qstar, use)
    sig2 = 0

    for t in range(n):
        Xt = X[t, :].reshape(-1, 1)
        P = par['Pstar_arr'][use[t], :, :] + max(t - use[t], 0) * Qstar
        err = y[t] - ((par['thetastar_arr'][use[t], :].reshape(-1,1) + np.dot(par['C_arr'][use[t], :, :],theta1)).T @ Xt)[0]
        sig2 += err**2 / (1 + (Xt.T@ np.dot(P,Xt))[0]) / n

    return np.sqrt(sig2)

def filter_null(x, y):
    return y if x is None else x



def iterative_grid_search(X, y, q_list, Q_init=None, max_iter=0, delay=1, use=None,
                           restrict=None, mode="gaussian", p1=0, ncores=1,
                           train_theta1=None, train_Q=None, verbose=True):
    n = X.shape[0]
    d = X.shape[1]
    train_theta1 = filter_null(train_theta1, np.array(range(n)))
    train_Q = filter_null(train_Q, np.array(range(n)))
    
    if Q_init is None:
        Qstar = np.diag(np.zeros(d))
    else:
        Qstar = np.diag(np.diag(Q_init))
    
    use = filter_null(use, np.array([max(x-delay, 0) for x in range(n)]))
    search_dimensions = filter_null(restrict, np.array(range(d)))
    b = True
    l_opt = loglik(X, y, Qstar, use, p1, train_theta1, train_Q, mode=mode)
    n_iter = 0
    LOGLIK = []
    init_time = time.time()
    
    while b:
        n_iter += 1
        b = n_iter < max_iter or max_iter == 0
        i = -1
        q_i = 0
        
        
        for k in search_dimensions:
            q_prev = Qstar[k, k]
            l_arr = []
            
            for q in q_list:
                Qstar[k, k] = q
                l_arr.append(loglik(X, y, Qstar, use, p1, train_theta1, train_Q, mode=mode))
            
            l_arr = np.array(l_arr)
            if np.max(l_arr) > l_opt:
                l_opt = np.max(l_arr)
                i = k
                q_i = q_list[np.argmax(l_arr)]
                
            Qstar[k, k] = q_prev
        
        LOGLIK.append(l_opt)
        
        if i == -1:
            b = False
        else:
            Qstar[i, i] = q_i
            
            if verbose:
                if q_i in [min(q_list), max(q_list)]:
                    warnings.warn("Values may not be suited: {}".format(q_i))
                    
                print("Iteration: {} | Log-likelihood: {} | Diagonal of Q/sigma^2:".format(n_iter, round(l_opt, 6)))
                print(np.diag(Qstar))
    
    par = parameters_star(X, y, Qstar, p1)
    sig = get_sig(X, y, par, Qstar, use)
    
    if verbose:
        print("Computation time of the iterative grid search: {}".format(time.time() - init_time))
    
    return {"theta": get_theta1(X, y, par, Qstar, use, mode=mode),
            "P": np.diag(np.ones(d) * p1 * sig**2),
            "sig": sig,
            "Q": Qstar * sig**2,
            "LOGLIK": LOGLIK}




def loglik(X, y, Qstar, use, p1, train_theta1, train_Q, mode="gaussian"):
    par = parameters_star(X, y, Qstar, p1)
    theta1 = get_theta1(X[train_theta1, :], y[train_theta1], par, Qstar, use, mode=mode)
    
    sum1 = 0
    sum2 = 0
    n = train_Q.shape[0]
    
    for t in train_Q:
        Xt = X[t, :].reshape(-1,1)
        P = par['Pstar_arr'][use[t], :, :] + max(0, t - use[t]) * Qstar
        sum1 += np.log(1 + (Xt.T@np.dot(P,Xt))[0])/n
        
        err2 = (y[t] - ((par['thetastar_arr'][use[t], :].reshape(-1,1) + np.dot(par['C_arr'][use[t], :, :],theta1)).T @ Xt)[0])**2
        
        sum2 += err2 / (1 + (Xt.T@np.dot(P,Xt))[0]) / n   

    return -0.5 * sum1 - 0.5 - 0.5 * np.log(2 * np.pi * sum2)

    
    
    
