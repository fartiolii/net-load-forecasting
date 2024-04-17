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


groups = [(0, 1), (1, 2), (2, 10), (10, 14), (14, 18), (18,22), (22, 28),
          (28, 36), (36, 41), (41,47),(47, 52), (52,55),(55, 58), (58, 59), (59, 60), (60, 61)] 


######## Kalman Filter Class

class Kalman_Filter(object):
    """This class represents a Kalman Filter model which adapts the coefficients of either a GAM model or a linear regression model.
    
    INPUT:
    - GAM: bool, True if the coefficients to be adapted are the ones of a GAM model, False if the coefficients are the ones of a linear regression model
    - gam_model: the fitted GAM model object of the pygam package
    - total_mat: bool, True if, when using the linear regression model, every single parameter is adapted. False if instead the contributions of different                    parameters*feature_evaluation are grouped in a certain number of groups and the KF adapts just one paramtere per group (e.g. what happens                    when adapting the coefficients of the GAM model)
    - params: used only for linear regression models if total_mat is False. It contains the vector of all the parameters and it is used to compute the                     "freezed" contributions to the different groups
    - theta1: Initial vector of parameters theta to adapt (theta_1|0)
    - P: Initial variance-covariance matrix of the vector theta (P_1|0)
    - sigma: Variance of the measurement error on y
    - Q: Variance-Covariance matrix of the error on the state theta
    """
    def __init__(self, gam_model=None, GAM=True, total_mat=True, params=None,theta1=None, P=None, sigma=None, Q=None):
        self.GAM=GAM
        self.GAM_model = gam_model
        self.static = True if Q is None else False
        self.theta1 = theta1 
        self.theta = self.theta1  # State vector (contains the parameters to adapt)
        self.theta_mat = None     # Matrix that contains all the different theta vectors computed progressively on the train set 
        self.P = P                # variance-covariance matrix of theta 
        self.sigma = sigma        # variance of the target variable measurement
        self.Q = Q                # variance-covariance matrix of the error on theta 
        self.total_mat = total_mat
        self.params = None if params is None else np.array(params)
        
        
    def f(self, df, train=False):
        """Method that given the feature matrix n_obs*n_features returns:
           - a matrix n_obs*n_groups if the number of parameters to adapt is lower than the number of features used (in the case of GAM or linear                        regression if total_mat is False)
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
        
        elif self.total_mat == True:
            return df.copy()
        else:
            # Multiplying the feature matrix by the parameters optimized with linear regression on the train set
            weighted_df = df*self.params
            # Summing the contributions for each feature group
            sums = [weighted_df[:, start:end].sum(axis=1) for start, end in groups]

            # Combining the sums into a new array
            features_eval = np.column_stack(sums)
            
            return features_eval
        
        
        
    def theta_update(self, f_Xt, yt):
        # Implementation formula (3.5)
        # update of theta with new data yt
        self.theta += self.P@f_Xt / (self.sigma**2) * (yt - self.theta.T@f_Xt)

        
    def P_update(self, f_Xt, yt):
        # Implementation formula (3.4)
        # update of matrix P
        self.P += -(self.P@f_Xt@f_Xt.T@self.P)/(f_Xt.T@self.P@f_Xt + self.sigma**2)

        
    def fit(self, X_train, Y_train):
        """This method fits the Kalman Filter with the feature matrix in input X_train and the target variable in input Y_train"""
        Yt = np.array(Y_train)
        f_Xt = self.f(X_train) # Features are evaluated based on which parameters will be adapted
        n, d = f_Xt.shape
        self.d = d # Length of the vector theta (that contains the parameters to adapt)
        
        # Parameters are initialized 
        if self.theta1 is None:
            self.theta1 = np.zeros((self.d,1))
        self.theta = self.theta1.copy()
        
        if self.P is None:
            self.P = np.eye(self.d)
        if self.Q is None:
            self.Q = np.zeros(self.d)
        if self.sigma is None:
            self.sigma = 1
            
        self.theta_mat = np.zeros((n,self.d))
        
        # This for cycle uses all data in the training set to update theta and its variance-covariance matrix P
        for t in range(n):
            self.theta_mat[t, :] = self.theta.flatten() # Save current theta vector in the matrix
            self.P_update(f_Xt[t,:].reshape(-1,1), Yt[t]) # Obtain P_t|t
            self.theta_update(f_Xt[t,:].reshape(-1,1), Yt[t]) # Update theta
            self.P += self.Q #Obtain P_t+1|t
        
            
    
    def predict(self, X_test, Y_test, delay=False, const_delay=True):
        """After having trained the KF with the method fit, this method computes the predictions on the test set given in input
           by either using the measurement from 30 minutes before (delay=False) or by considering the fact that, every day, data is 
           obtained at midnight of the day after (delay=True)
           const_delay: Bool, True if we incorporate past data in the model with a constant delay of 48h, False if instead we take into 
                        account the fact that at midnight of each day, we get all data regarding the past 48 to 24 hours 
        """
        if self.theta_mat is None:
            print("Kalman Filter must be trained first")
        if not delay:
            # In this case, to predict the value of the next 30 min we are using the measurement of the 30 min before
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
                self.P_update(ft, Y_test[t])                            # Compute P_t|t
                self.theta_update(ft, Y_test[t])                        # Update theta
                self.P += self.Q                                        # Compute P_t+1|t

            self.theta = theta_init        # Initial value of theta is copied back in the state vector of the object KF
            self.P = P_init                # Initial value of P is copied back in the variance-covariance matrix of the object KF
        else:
            Y_target = Y_test['targetTime'] # Target time of the prediction to make 
            Y_test = np.array(Y_test[Y_test.columns[1]]) # Target value to predict given in input (can be net-load, 7 days differences etc.)
            f_Xtest = self.f(X_test) # Evaluation matrix of the features of the test set, according to which parameters to adapt
            n_test = f_Xtest.shape[0]
            y_mean = np.zeros(n_test,) # vector of mean predictions is initialized
            y_std = np.zeros(n_test,)  # vector of the standard deviation of the predictions is initialized
            
            if const_delay is False:
                # In this case, to predict the value of every 30 min throughout a single day, we are using all measurements up to the midnight before                       # (which are measurements that go from 48h to 24h before the midnight). e.g. the prediction of values from 25/01 at 00:00 to 25/01 at 23:30                 # occurs the 25/01 at 00:00 with new measurement data incorporated in the model which goes from 23/01 at 00:00 to 23/01 at 23:30.
   
                theta_init = self.theta.copy() # Initial value of theta is saved (as the prediction should not change theta/P)
                P_init = self.P.copy()         # Initial value of P is saved (as the prediction should not change theta/P)

                for t in range(n_test):
                    # We check if the target time of the prediction corresponds to midnight 
                    if Y_target.iloc[t].hour == 0 and Y_target.iloc[t].minute == 0:
                        # We check if data from 48h to 24h before is available (condition not true just at the beginning of the dataset)
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

                        # Prediction of the next 24h are computed 
                        ft = f_Xtest[t:t+48,:]
                        y_mean[t:t+48] = (ft@self.theta).flatten()

                        # The prediction of the standard deviation requires to sum k-times Q to P, where k represents the number of steps ahead of the                               # prediction that we are doing
                        P = self.P + 48*self.Q
                        for idx in range(t, t+48):
                            f_t = f_Xtest[idx,:].reshape(-1,1)
                            P += self.Q
                            y_std[idx] = np.sqrt(self.sigma**2 + f_t.T@P@f_t)


                self.theta = theta_init      # Initial value of theta is copied back in the state vector of the object KF
                self.P = P_init              # Initial value of P is copied back in the variance-covariance matrix of the object KF

            else:
                # In this case we use the data to update the model as if data is constantly arriving with a 48h delay throughout the day. At each time                     # instant we update the model with the measurement of 48h before and then predict the new value of yt.
                
                theta_init = self.theta.copy() # Initial value of theta is saved (as the prediction should not change theta/P)
                P_init = self.P.copy()         # Initial value of P is saved (as the prediction should not change theta/P)

                # For cycle that computes predictions for all the test set in input
                for t in range(n_test):
                    if t >= 96:
                        # Model update with the measurement from 48h before
                        ft = f_Xtest[t-96,:].reshape(-1,1)
                        self.P_update(ft, Y_test[t-96])                            # Compute P_t|t
                        self.theta_update(ft, Y_test[t-96])                        # Update theta
                        self.P += self.Q                                        # Compute P_t+1|t
                    
                    # Prediction
                    ft = f_Xtest[t,:].reshape(-1,1)
                    y_mean[t] = self.theta.T@ft                                         # Mean prediction
                    y_std[t] = np.sqrt(self.sigma**2 + ft.T@(self.P+95*self.Q)@ft)      # Standard deviation of the prediction


                self.theta = theta_init        # Initial value of theta is copied back in the state vector of the object KF
                self.P = P_init                # Initial value of P is copied back in the variance-covariance matrix of the object KF
                
        return y_mean, y_std
    
    
    
    
    # The following two methods are used for the optimization of the variances of the dynamic Kalman Filter only
    
    def compute_reduced_likelihood(self, X_test, Y_test, const_delay=True, fit_theta1=False):
        """Method that computes the likelihood (assuming delayed data) as in Formula (3.13) on the training set in input, given Q*.
        It computes sigma (Formula 3.9) by maximizing the likelihood (given Q*) and therefore it's used for 
        optimization of variances on a unique grid of q*."""
        
        Y_target = Y_test['targetTime'] # Target time of the prediction to make 
        Y_test = np.array(Y_test[Y_test.columns[1]]) # Target value to predict given in input (can be net-load, 7 days differences etc.)
        f_Xtest = self.f(X_test) # Evaluation matrix of the features of the test set, according to which parameters to adapt
        n_test, d = f_Xtest.shape
        self.d = d
        y_mean = np.zeros(n_test,) # vector of mean predictions is initialized
        y_std = np.zeros(n_test,)  # vector of the standard deviation of the predictions is initialized
        
        # Since Q*=Q/sigma^2 and P*=P/sigma^2 we can continue to use the previous notation (therefore Q and P) if we set sigma=1
        self.sigma = 1
        if fit_theta1:
            # Optimization of theta_1|0 (prior) (Formula (3.11)) which in some cases perform worse than simply initializing to 0 the vector
            self.theta1 = self.optimize_theta1(f_Xtest, Y_target, Y_test, const_delay=const_delay)
        else:
            self.theta1 = np.zeros((self.d,1))
        self.theta = self.theta1.copy()
        
        if const_delay is False:
        
            theta_init = self.theta.copy()
            P_init = self.P.copy()

            # Computation of the likelihood (Fomula (3.13))
            loglik = 0
            err_norm_sigma = 0

            for t in range(n_test):
                # We check if the target time of the prediction corresponds to midnight 
                if Y_target.iloc[t].hour == 0 and Y_target.iloc[t].minute == 0:
                    # We check if data from 48h to 24h before is available (condition not true just at the beginning of the dataset)
                    if t-96 >= 0:
                        # KF is updated with measurements from 48h to 24h before (which correspond to indexes -96 to -48 since we have half an hour data)
                        ft = f_Xtest[t-96:t-48,:]
                        yt = Y_test[t-96:t-48]
                        # For cycle that updates the KF with 48 measurements (which correspond to 24h data)
                        for j in range(ft.shape[0]):
                            f_t = ft[j,:].reshape(-1,1)
                            y_t = yt[j]
                            self.P_update(f_t,y_t)
                            self.theta_update(f_t,y_t)
                            self.P += self.Q

                    # Predictions of the next 24h are computed 
                    ft = f_Xtest[t:t+48,:]
                    y_mean[t:t+48] = (ft@self.theta).flatten()

                    # The prediction of the standard deviation requires to sum k-times Q to P, where k represents the number of steps ahead of the                               # prediction that we are doing
                    P = self.P.copy()+ 48*self.Q
                    corr_val = np.zeros((48,))
                    i=0
                    for idx in range(t, t+48):
                        ft = f_Xtest[idx,:].reshape(-1,1)
                        P += self.Q
                        y_std[idx] = np.sqrt(self.sigma**2 + ft.T@P@ft)

                        corr_val[i] = self.sigma**2 + ft.T@P@ft
                        i += 1

                    err = (Y_test[t:t+48] - y_mean[t:t+48])**2
                    loglik += np.sum(np.log(corr_val)) 
                    err_norm_sigma += np.sum(err/corr_val)   


            self.theta = theta_init
            self.P = P_init
        else:
            # In this case we use the data to update the model as if data is constantly arriving with a 48h delay throughout the day. At each time                     # instant we update the model with the measurement of 48h before and then predict the new value of yt.
        
            theta_init = self.theta.copy() # Initial value of theta is saved (as the prediction should not change theta/P)
            P_init = self.P.copy()         # Initial value of P is saved (as the prediction should not change theta/P)

            loglik = 0
            err_norm_sigma = 0

            # For cycle that computes predictions for all the test set in input
            for t in range(n_test):
                if t >= 96:
                    ft = f_Xtest[t-96,:].reshape(-1,1)
                    self.P_update(ft, Y_test[t-96])                            # Compute P_t|t
                    self.theta_update(ft, Y_test[t-96])                        # Update theta
                    self.P += self.Q                                        # Compute P_t+1|t

                ft = f_Xtest[t,:].reshape(-1,1)
                y_mean[t] = self.theta.T@ft                             # Mean prediction
                y_std[t] = np.sqrt(self.sigma**2 + ft.T@(self.P+95*self.Q)@ft)      # Standard deviation of the prediction

                err = (Y_test[t] - y_mean[t])**2
                loglik += np.log(y_std[t]**2)
                err_norm_sigma += err/y_std[t]**2


            self.theta = theta_init        # Initial value of theta is copied back in the state vector of the object KF
            self.P = P_init                # Initial value of P is copied back in the variance-covariance matrix of the object KF


        # Estimation of sigma (Formula (3.9))
        self.sigma = np.sqrt(err_norm_sigma/n_test)

        loglik += n_test*np.log(err_norm_sigma/n_test)
        
        return y_mean, y_std, -0.5*loglik 
    
    
    def optimize_theta1(self, f_Xtest, Y_target, Y_test, const_delay=True):
        """Method which computes the optimized theta_1|0 given Q* and P_1|0=I by using Formula (3.11)"""
        n_test, d = f_Xtest.shape
        y_mean = np.zeros(n_test,)
        
        self.theta1 = np.zeros((d,1))
        self.theta = self.theta1.copy()
        if self.P is None:
            self.P = np.eye(d) 
        
        C_matrix = np.diag(np.ones(d))
        P_new = self.P.copy()
        theta = self.theta.copy()
        
        # We initialize the numerator and the matrix to invert to obtain theta_1|0 according to the formula
        num = np.zeros((d,1))
        inv = np.zeros((d,d))
        
        if const_delay is False:
            for t in range(n_test):
                # We check if the target time of the prediction corresponds to midnight 
                if Y_target.iloc[t].hour == 0 and Y_target.iloc[t].minute == 0:
                    # We check if data from 48h to 24h before is available (condition not true just at the beginning of the dataset)
                    if t-96 >= 0:
                        # KF is updated with measurements from 48h to 24h before (which correspond to indexes -96 to -48 since we have half an hour data)
                        ft = f_Xtest[t-96:t-48,:]
                        yt = Y_test[t-96:t-48]
                        # For cycle that updates the KF with 48 measurements (which correspond to 24h data) and computes matrix C
                        for j in range(ft.shape[0]):
                            f_t = ft[j,:].reshape(-1,1)
                            y_t = yt[j]

                            P_new += -(P_new@f_t@f_t.T@P_new)/(f_t.T@P_new@f_t + self.sigma**2)
                            # Iterative computation of matrix C according to Formula (3.12)
                            C_matrix = (np.eye(d) - (P_new@f_t)@f_t.T)@C_matrix
                            theta += P_new@f_t * (y_t - theta.T@f_t)/ (self.sigma**2) 
                            P_new += self.Q

                    # Predictions of the next 24h are computed 
                    ft = f_Xtest[t:t+48,:]
                    y_mean[t:t+48] = (ft@self.theta).flatten()

                    P = P_new.copy() + 48*self.Q
                    for idx in range(t, t+48):
                        ft = f_Xtest[idx,:].reshape(-1,1)
                        P += self.Q

                        num += (Y_test[idx] - y_mean[idx])/(self.sigma**2 + ft.T@P@ft)*C_matrix.T@ft
                        inv += C_matrix.T@ft@ft.T@C_matrix/(self.sigma**2 + ft.T@P@ft)
                        
        else:
            for t in range(n_test):
                if t >= 96:
                    ft = f_Xtest[t-96,:].reshape(-1,1)
                    yt = Y_test[t-96]
                    
                    P_new += -(P_new@ft@ft.T@P_new)/(ft.T@P_new@ft + self.sigma**2)
                    # Iterative computation of matrix C according to formula (5.7) of [1]
                    C_matrix = (np.eye(d) - (P_new@ft)@ft.T)@C_matrix
                    theta += P_new@ft * (yt - theta.T@ft)/ (self.sigma**2) 
                    P_new += self.Q

                ft = f_Xtest[t,:].reshape(-1,1)
                y_mean[t] = self.theta.T@ft                             # Mean prediction
                y_std[t] = np.sqrt(self.sigma**2 + ft.T@(P_new+95*self.Q)@ft)      # Standard deviation of the prediction
                
                
                num += (Y_test[t] - y_mean[t])/(self.sigma**2 + ft.T@(P_new+95*self.Q)@ft)*C_matrix.T@ft
                inv += C_matrix.T@ft@ft.T@C_matrix/(self.sigma**2 + ft.T@(P_new+95*self.Q)@ft)
                
                 
        # Computation of vector theta1 according to the Formula (3.11)
        theta1 = np.linalg.inv(inv)@num
        return theta1
    
###################### Kalman Filter variance-covariance optimization ##############################
class Q_optimization(object):
    """Class which optimizes the variances (Q and sigma) of the dynamic Kalman Filter by performing a grid search and by 
       maximizing the likelihood.
       """
    def __init__(self, X_train, y_train, gam_model=None, GAM=False, total_mat=True, params=None):
        self.X_train = X_train # Features of the training set
        self.y_train = y_train # Target variable of the training set
        # Same as parameters of the Kalman Filter class
        self.GAM = GAM        
        self.gam_model = gam_model
        self.total_mat = total_mat 
        self.params = None if params is None else np.array(params)
        
     
    def grid_search_reduced_likelihood(self, q_list=[1,1e-5,1e-10], std_static=None, const_delay=True,fit_theta1=False):
        """This method returns the optimal matrix Q, the optimal sigma, P_1|0=sigma^2*I_d and theta_1|0 by performing a grid search on a single grid 
        of q* values. 
        std_static is an optional input vector which represents the standard deviation of each parameter in the vector theta, obtained on the training set           by the Static Kalman Filter. 
        This method assumes that Q* = (q*)*I_d where I_d is an identity matrix if std_static is None otherwise Q = (q*)*diag(std_static^2).
        The optimal matrices are obtained by maximizing the likelihood (Formula (3.13)) on the training set."""
        
        # Since Q*=Q/sigma^2 and P*=P/sigma^2 we can continue to use the previous notation (i.e. Q and P) if we set sigma=1
        max_loglik = -np.inf
        sigma = 1
        optimal_q = None
        
        # For each q* we compute the likelihood and choose the value of q* maximizing the likelihood
        for q in q_list:
            print(q)
               
            # Initialization of the matrices and the KF according to the model (GAM or linear regression)
            if self.GAM:
                n_terms = len(list(self.gam_model.terms))
                if std_static is None:
                    Q = np.eye(n_terms)*q
                    P = np.eye(n_terms)
                else:
                    Q = np.diag(std_static**2)*q
                    P = np.eye(n_terms)

                kf = Kalman_Filter(gam_model=self.gam_model, Q=Q, sigma=sigma, P=P)
            else:
                if self.total_mat:
                    if std_static is None:
                        Q = np.eye(np.shape(self.X_train)[1])*q
                        P = np.eye(np.shape(self.X_train)[1])
                    else:
                        Q = np.diag(std_static**2)*q
                        P = np.eye(np.shape(self.X_train)[1])

                    kf = Kalman_Filter(GAM=False, Q=Q, sigma=sigma, P=P)
                else:
                    n_terms = len(groups)
                    if std_static is None:
                        Q = np.eye(n_terms)*q
                        P = np.eye(n_terms)
                    else:
                        Q = np.diag(std_static**2)*q
                        P = np.eye(n_terms)

                    kf = Kalman_Filter(GAM=False, total_mat=False,params=self.params, Q=Q, sigma=sigma, P=P)
            
            y_hat, _, loglik = kf.compute_reduced_likelihood(self.X_train, self.y_train, const_delay=const_delay, fit_theta1=fit_theta1)
            print(loglik)
            if loglik > max_loglik:
                max_loglik = loglik
                optimal_sigma = kf.sigma   
                optimal_q = q*optimal_sigma**2
                optimal_theta1 = kf.theta1

        print("Optimal values q and sigma: ", optimal_q, optimal_sigma)
        
         # Returning Q, sigma, P_1|0 and theta_1|0 according to the model
        if self.GAM:
            n_terms = len(list(self.gam_model.terms))
            if std_static is None:
                return np.eye(n_terms)*optimal_q, optimal_sigma, np.eye(n_terms)*optimal_sigma**2, optimal_theta1
            else:
                return np.diag(std_static**2)*optimal_q, optimal_sigma, np.eye(n_terms)*optimal_sigma**2, optimal_theta1
        else:
            if self.total_mat:
                if std_static is None:
                    return np.eye(np.shape(self.X_train)[1])*optimal_q, optimal_sigma, np.eye(np.shape(self.X_train)[1])*optimal_sigma**2,optimal_theta1
                else:
                    return np.diag(std_static**2)*optimal_q, optimal_sigma, np.eye(np.shape(self.X_train)[1])*optimal_sigma**2, optimal_theta1
                        
            else:
                n_terms = len(groups)
                if std_static is None:
                    return np.eye(n_terms)*optimal_q, optimal_sigma, np.eye(n_terms)*optimal_sigma**2,optimal_theta1
                else:
                    return np.diag(std_static**2)*optimal_q, optimal_sigma, np.eye(n_terms)*optimal_sigma**2,optimal_theta1

        
        
######## Persistence Benchmark
class Persistence_Benchmark(object):
    """This class represents the persistence benchmark (2 days) used to compare the predictions of every other model.
       The prediction for each target time coincides with the actual value of 2 days before."""
    def __init__(self, Y_df):
        self.Y_df = Y_df
    
    def predict(self, targetTime):
        y_pred = np.zeros((len(targetTime),))
        
        for t in range(len(targetTime)):
            # At midnight the predictions are computed for the following 24h (i.e. 48 points)
            if targetTime.iloc[t].hour == 0 and targetTime.iloc[t].minute == 0:
                dt = targetTime.iloc[t] - pd.Timedelta(days=2) # The predictions coincide with the values 2 days before
                idx = (self.Y_df['targetTime'] == dt).idxmax() 
                y_pred[t:t+48] = self.Y_df['node'].iloc[idx:idx+48]
        return y_pred
        
    
######## Mean Performance Evaluation metrics

def RMSE(y, yhat):
    """This method computes the Root Mean Square Error of the prediction yhat wrt to the vector y """
    return np.sqrt(np.sum((y-yhat)**2)/len(y))

def MAE(y, yhat):
    """This method computes the Mean Absolute Error of the prediction yhat wrt to the vector y """
    return 1/len(y)*np.sum(np.abs(y-yhat))

def MAPE(y, yhat):
    """This method computes the Mean Absolute Percentage Error of the prediction yhat wrt to the vector y """
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
        
    
######## Quantile Regression Class

class Quantile_Regression(object):
    def __init__(self, X_train, Y_train, y_pred):
        self.X_train = X_train
        self.Y_train = Y_train
        self.y_pred = y_pred
        self.beta_opt_dict = {}
        self.z_train = self.dataset_preprocessing_qr(self.X_train, self.y_pred)
    
    def rho(self, y, yhat, q):
        return (1*(y<yhat)-q)*(yhat-y)
    
    def loss_fun_qr(self, beta, q):
        return np.sum(self.rho(self.Y_train-self.y_pred,self.z_train@beta,q))
    
    def beta_opt(self, q):
        if q not in self.beta_opt_dict.keys():
            beta_0 = np.zeros(self.z_train.shape[1])
            res = minimize(self.loss_fun_qr, beta_0, args=(q))
            self.beta_opt_dict[q] = res.x
        return self.beta_opt_dict[q]

    
    def dataset_preprocessing_qr(self, df, y_pred):
        y_pred_std = (y_pred - y_pred.mean())/y_pred.std()
        features_to_standardize = ['SSRD_mean_2_Cap','WindSpd100_weighted.mean_cell','x2T_weighted.mean_p_max_point']
        df_std = standardize_dataframe(df[features_to_standardize])
        features_qr = np.c_[y_pred_std, y_pred_std**2,    
                            1*(df['clock_hour']>=0)*(df['clock_hour']<1), 1*(df['clock_hour']>=1)*(df['clock_hour']<2),
                            1*(df['clock_hour']>=2)*(df['clock_hour']<3), 1*(df['clock_hour']>=3)*(df['clock_hour']<4),
                            1*(df['clock_hour']>=4)*(df['clock_hour']<5), 1*(df['clock_hour']>=5)*(df['clock_hour']<6),
                            1*(df['clock_hour']>=6)*(df['clock_hour']<7), 1*(df['clock_hour']>=7)*(df['clock_hour']<8),
                            1*(df['clock_hour']>=8)*(df['clock_hour']<9), 1*(df['clock_hour']>=9)*(df['clock_hour']<10),
                            1*(df['clock_hour']>=10)*(df['clock_hour']<11), 1*(df['clock_hour']>=11)*(df['clock_hour']<12), 
                            1*(df['clock_hour']>=12)*(df['clock_hour']<13), 1*(df['clock_hour']>=13)*(df['clock_hour']<14),
                            1*(df['clock_hour']>=14)*(df['clock_hour']<15), 1*(df['clock_hour']>=15)*(df['clock_hour']<16),
                            1*(df['clock_hour']>=16)*(df['clock_hour']<17), 1*(df['clock_hour']>=17)*(df['clock_hour']<18),
                            1*(df['clock_hour']>=18)*(df['clock_hour']<19), 1*(df['clock_hour']>=19)*(df['clock_hour']<20), 
                            1*(df['clock_hour']>=20)*(df['clock_hour']<21), 1*(df['clock_hour']>=21)*(df['clock_hour']<22),
                            1*(df['clock_hour']>=22)*(df['clock_hour']<23), 1*(df['clock_hour']>=23),
                            1*(df['dow_Rph']=='Lun'),1*(df['dow_Rph']=='Mar'),1*(df['dow_Rph']=='Mer'),
                            1*(df['dow_Rph']=='Jeu'),1*(df['dow_Rph']=='Ven'),1*(df['dow_Rph']=='Sam'),
                            1*(df['dow_Rph']=='Dim'),1*(df['dow_Rph']=='Hol'), 
                            df_std['SSRD_mean_2_Cap'], df_std['WindSpd100_weighted.mean_cell'], df_std['x2T_weighted.mean_p_max_point']]
        return features_qr
    
    def predict(self, q, df_test=None, y_pred_test=None):
        if df_test is None:
            df_test = self.X_train
        if y_pred_test is None:
            y_pred_test = self.y_pred
            
        beta_opt_q = self.beta_opt(q)

        z_test = self.dataset_preprocessing_qr(df_test, y_pred_test)
        return y_pred_test + z_test@beta_opt_q
    
class Online_Gradient_Descent(Quantile_Regression):
    def __init__(self,X_train, Y_train, y_pred):
        super().__init__(X_train, Y_train, y_pred)
        self.alpha = None
        
    def predict(self, q, df_test, y_test, y_pred_test, alpha):
        self.alpha = alpha
        beta = super().beta_opt(q).reshape(-1,1)
        z_test = super().dataset_preprocessing_qr(df_test, y_pred_test)
        pred_quantile = np.zeros((len(y_pred_test),1))
        y_test = np.array(y_test)
        
        for t in range(len(y_pred_test)):
            zt = z_test[t,:].reshape(1,-1)
            pred_quantile[t] = zt@beta
            beta = self.beta_update(beta,y_test[t],y_pred_test[t],zt,q)
        
        return (y_pred_test.reshape(-1,1) + pred_quantile).flatten()
        
    def beta_update(self,beta,y,yhat,z,q):
        beta += -self.alpha*(1*(y-yhat<z@beta)-q)*z.T
        return beta
            
        
    
    


######## Probability Distribution evaluation metrics

def pinball_score_alpha(estimated_quantile,y,alpha):
    return np.sum((estimated_quantile - y)*(1*(y <= estimated_quantile)-alpha))/len(y)


def pinball_score(estimated_quantile_mat, y, alpha_vec):
    score = 0
    for i in range(alpha_vec.shape[0]):
        score += pinball_score_alpha(estimated_quantile_mat[:,i],y,alpha_vec[i])
    return score/alpha_vec.shape[0]

def RPS(estimated_quantile_mat, y, alpha_vec):
    score = 0
    for i in range(alpha_vec.shape[0]):
        if i == 0:
            qi = 0
            qi_1 = alpha_vec[i+1]
        elif i == alpha_vec.shape[0]-1:
            qi = alpha_vec[i-1]
            qi_1 = 1
        else:
            qi = alpha_vec[i-1]
            qi_1 = alpha_vec[i+1]

        score += pinball_score_alpha(estimated_quantile_mat[:,i],y,alpha_vec[i])*(qi_1-qi)
    return score/alpha_vec.shape[0]


def standardize_dataframe(df):
    return (df-df.mean())/df.std()
    
    
######## Iterative Grid Search for Dynamic Kalman Filter (viking package in R)

# The following methods coincide with the translation to Python of the viking methods in R

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

    
    
    
