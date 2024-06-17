from utility_functions import * 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge,Lasso
import mkl

mkl.set_num_threads(1)

def generate_data_interactive(noise_low,noise_high,B_true, n, p, v, polykernel_degree = 3, 
                              noise_half_width = 0.01, constant = 3):
    (d, _) = B_true.shape
    X = np.random.normal(size=(p, n))
    #X = np.random.randn(p, n)
    #X = np.random.uniform(0,2,size=(p,n))
    poly = PolynomialFeatures(degree = polykernel_degree, interaction_only = True, include_bias = False)
    X_new = np.transpose(poly.fit_transform(np.transpose(X)))
    
    c_expected = np.matmul(B_true, X_new) + constant 
    epsilon =  np.random.uniform(noise_low,noise_high,size=(d,n))
    c_observed = c_expected + epsilon

    X_false = np.concatenate((np.reshape(np.ones(n), (1, -1)), X), axis = 0)
    X_true = np.concatenate((np.reshape(np.ones(n), (1, -1)), X_new), axis = 0)
    
    Y_transpose_Z_observed=np.zeros((1,n))
    Y_transpose_Z_expected=np.zeros((1,n))
    Z=np.zeros((d,n))
    
    for i in range(n):
        idxs = np.random.randint(0, 70, size=1)
        temp_Y_transpose_Z_observed=np.dot(c_observed[:,i],feasible_vector[idxs[0]])
        temp_Y_transpose_Z_expected=np.dot(c_expected[:,i],feasible_vector[idxs[0]])
        Y_transpose_Z_observed[0,i]=temp_Y_transpose_Z_observed
        Y_transpose_Z_expected[0,i]=temp_Y_transpose_Z_expected
        Z[:,i]=feasible_vector[idxs[0]]
    
    return [X_false, X_true, c_observed, c_expected,Y_transpose_Z_observed,Y_transpose_Z_expected,Z]

def Double_robust(X_train_False,X_train, Y_transpose_Z_observed_train, Y_transpose_Z_expected_train,Z_train,
                  X_val_False,X_val, Y_transpose_Z_observed_val, Y_transpose_Z_expected_val,Z_val,
                  X_train_False_2,X_train_2, Y_transpose_Z_observed_train_2, Y_transpose_Z_expected_train_2,Z_train_2,
                  X_val_False_2,X_val_2, Y_transpose_Z_observed_val_2, Y_transpose_Z_expected_val_2,Z_val_2,
                  Train_expected_edge_cost_2,Val_expected_edge_cost_2,Sigma_matrix=False):
    X=np.hstack((X_train,X_val))
    Z=np.hstack((Z_train,Z_val))
    Y=np.transpose(np.hstack((Y_transpose_Z_observed_train,Y_transpose_Z_observed_val)))
    X_False=np.hstack((X_train_False,X_val_False))
    
    X_2=np.hstack((X_train_2,X_val_2))
    Z_2=np.hstack((Z_train_2,Z_val_2))
    Y_2=np.transpose(np.hstack((Y_transpose_Z_observed_train_2,Y_transpose_Z_observed_val_2)))
    Y_2_expected=np.transpose(np.hstack((Y_transpose_Z_expected_train_2,Y_transpose_Z_expected_val_2)))
    Y_2_edge_expected=np.hstack((Train_expected_edge_cost_2,Val_expected_edge_cost_2))
    X_2_False=np.hstack((X_train_False_2,X_val_False_2))
    
    #print(X)
    (p,m)=X.shape
    (d,m)=Z.shape
    (p_False,m)=X_False.shape
    new_X=[]
    new_X_False=[]
    new_X_2=[]
    new_X_2_False=[]
    
    for i in range(m):
        temp_X=[]
        temp_X_2=[]
        #temp_X_False=[]
        #temp_X_2_False=[]
        for j in range(p):
           for k in range(d):
               temp_X.append(X[j,i]*Z[k,i])
               temp_X_2.append(X_2[j,i]*Z_2[k,i])
               #temp_X_False.append(X_False[j,i]*Z[k,i])
               #temp_X_2_False.append(X_2_False[j,i]*Z[k,i])
        new_X.append(temp_X)
        new_X_2.append(temp_X_2)
        #new_X_False.append(temp_X_False)
        #new_X_2_False.append(temp_X_2_False)
    
    for i in range(m):
        #temp_X=[]
        #temp_X_2=[]
        temp_X_False=[]
        temp_X_2_False=[]
        for j in range(p_False):
           for k in range(d):
               #temp_X.append(X[j,i]*Z[k,i])
               #temp_X_2.append(X_2[j,i]*Z_2[k,i])
               temp_X_False.append(X_False[j,i]*Z[k,i])
               temp_X_2_False.append(X_2_False[j,i]*Z[k,i])
        #new_X.append(temp_X)
        #new_X_2.append(temp_X_2)
        new_X_False.append(temp_X_False)
        new_X_2_False.append(temp_X_2_False)
    
    #linear model
    data_X = np.zeros((m,p*d))
    data_X_2 = np.zeros((m,p*d))
    data_X_False = np.zeros((m,p_False*d))
    data_X_2_False = np.zeros((m,p_False*d))
    
    for i in range(m):
        data_X[i,:] = new_X[i]
        data_X_2[i,:] = new_X_2[i]
        data_X_False[i,:] = new_X_False[i]
        data_X_2_False[i,:] = new_X_2_False[i]
        
        
    data_X=pd.DataFrame(data_X)
    data_X_2=pd.DataFrame(data_X_2)
    data_X_False=pd.DataFrame(data_X_False)
    data_X_2_False=pd.DataFrame(data_X_2_False)
    
    
    Y=pd.DataFrame(Y)
    Y_2=pd.DataFrame(Y_2)
    model = Ridge(fit_intercept=False,alpha=1) 
    wrong_model_alpha_1 = Ridge(fit_intercept=False,alpha=100) 
    wrong_model_alpha_2 = Ridge(fit_intercept=False,alpha=1000) 
    wrong_model_alpha_3 = Ridge(fit_intercept=False,alpha=10000) 
    wrong_model_linear = Ridge(fit_intercept=False,alpha=1) 
    
    model.fit(data_X, Y) 
    wrong_model_alpha_1.fit(data_X, Y) 
    wrong_model_alpha_2.fit(data_X, Y) 
    wrong_model_alpha_3.fit(data_X, Y) 
    wrong_model_linear.fit(data_X_False, Y) 
    
    predicts = model.predict(data_X) 
    predicts_2 = model.predict(data_X_2) 
    predicts_wrong_model_alpha_1=wrong_model_alpha_1.predict(data_X_2)  
    predicts_wrong_model_alpha_2=wrong_model_alpha_2.predict(data_X_2) 
    predicts_wrong_model_alpha_3=wrong_model_alpha_3.predict(data_X_2) 
    wrong_linear_model_predicts_2= wrong_model_linear.predict(data_X_2_False) 
    
    
    parameters = model.coef_ 
    wrong_model_alpha_1_parameters= wrong_model_alpha_1.coef_
    wrong_model_alpha_2_parameters= wrong_model_alpha_2.coef_
    wrong_model_alpha_3_parameters= wrong_model_alpha_3.coef_
    wrong_linear_model_parameters= wrong_model_linear.coef_
    
    
    #model predict result
    W_matrix=np.zeros((p,d))
    W_wrong_alpha_1_matrix=np.zeros((p,d))
    W_wrong_alpha_2_matrix=np.zeros((p,d))
    W_wrong_alpha_3_matrix=np.zeros((p,d))
    W_wrong_linear_matrix=np.zeros((p_False,d))
    for i in range(p):
        W_matrix[i,:]=parameters[0,i*d:i*d+d]
        W_wrong_alpha_1_matrix[i,:]=wrong_model_alpha_1_parameters[0,i*d:i*d+d]
        W_wrong_alpha_2_matrix[i,:]=wrong_model_alpha_2_parameters[0,i*d:i*d+d]
        W_wrong_alpha_3_matrix[i,:]=wrong_model_alpha_3_parameters[0,i*d:i*d+d]
    for i in range(p_False):
        W_wrong_linear_matrix[i,:]=wrong_linear_model_parameters[0,i*d:i*d+d]
    
    model_estimate=np.dot(np.transpose(X),W_matrix)
    model_estimate=np.transpose(model_estimate)
    model_estimate_2=np.dot(np.transpose(X_2),W_matrix)
    model_estimate_2=np.transpose(model_estimate_2)      
    Linear_Wrong_model_estimate_2=np.dot(np.transpose(X_2_False),W_wrong_linear_matrix)  
    Linear_Wrong_model_estimate_2=np.transpose(Linear_Wrong_model_estimate_2)     
    W_wrong_alpha_1_2=np.dot(np.transpose(X_2),W_wrong_alpha_1_matrix)
    W_wrong_alpha_1_2=np.transpose(W_wrong_alpha_1_2)    
    W_wrong_alpha_2_2=np.dot(np.transpose(X_2),W_wrong_alpha_2_matrix)
    W_wrong_alpha_2_2=np.transpose(W_wrong_alpha_2_2)    
    W_wrong_alpha_3_2=np.dot(np.transpose(X_2),W_wrong_alpha_3_matrix)
    W_wrong_alpha_3_2=np.transpose(W_wrong_alpha_3_2) 
    
    #Sigma matrix
    Sigma_matrix_sample=np.zeros((d,d))
    for i in range(m):
         temp_matrix=np.outer(Z[:,i],Z[:,i])
         Sigma_matrix_sample=Sigma_matrix_sample+temp_matrix
       #rank=np.linalg.matrix_rank(Sigma_matrix)
    Sigma_matrix_sample=Sigma_matrix_sample/m
    Sigma_matrix_sample_inverse = np.linalg.pinv(Sigma_matrix_sample)    
    Sigma_matrix_inverse = np.linalg.pinv(Sigma_matrix)  

    sample_U, sample_Sigma,sample_V = np.linalg.svd(Sigma_matrix_sample)
    for i in range(len(sample_Sigma)):
      if sample_Sigma[i]<0.00000000001:
        sample_Sigma[i]=0
      if sample_Sigma[i]>0.00000000001 and sample_Sigma[i]<1:
        sample_Sigma[i]=1
    Modeified_sample_Sigma=np.dot(np.dot(sample_U,np.diag(sample_Sigma)),sample_V)
    Modeified_sample_Sigma_inverse=np.linalg.pinv(Modeified_sample_Sigma)

    Ideal_U, Ideal_Sigma,Ideal_V = np.linalg.svd(Sigma_matrix)
    for i in range(len(Ideal_Sigma)):
      if Ideal_Sigma[i]<0.00000000001:
        Ideal_Sigma[i]=0
      if Ideal_Sigma[i]>0.00000000001 and Ideal_Sigma[i]<1:
        Ideal_Sigma[i]=1
    Modeified_Ideal_Sigma=np.dot(np.dot(Ideal_U,np.diag(Ideal_Sigma)),Ideal_V)
    Modeified_Ideal_Sigma_inverse=np.linalg.pinv(Modeified_Ideal_Sigma)
    
    Sigma_matrix_adjustment_sample_lambda_1=Sigma_matrix_sample+np.eye(d,d)
    Sigma_matrix_adjustment_sample_lambda_1_inverse=np.linalg.inv(Sigma_matrix_adjustment_sample_lambda_1)

    Vanilla_Double_robust_predict_2_sample=np.zeros((d,m))
    Clip_Double_robust_predict_2_sample=np.zeros((d,m))
    Lambda_Double_robust_predict_2_sample=np.zeros((d,m))
    # Vanilla_Perfect_Double_robust_predict_ideal_Sigma=np.zeros((d,m))
    # Clip_Perfect_Double_robust_predict_ideal_Sigma=np.zeros((d,m))
    Linear_Wrong_Double_Robust_2_sample=np.zeros((d,m))
    Clip_Linear_Wrong_Double_Robust_2_sample=np.zeros((d,m))
    Lambda_Linear_Wrong_Double_Robust_2_sample=np.zeros((d,m))
    
    Double_robust_wrong_alpha_1_predict_2=np.zeros((d,m))
    Clip_Double_robust_wrong_alpha_1_predict_2=np.zeros((d,m))
    Lambda_Double_robust_wrong_alpha_1_predict_2=np.zeros((d,m))
    
    Double_robust_wrong_alpha_2_predict_2=np.zeros((d,m))
    Clip_Double_robust_wrong_alpha_2_predict_2=np.zeros((d,m))
    Lambda_Double_robust_wrong_alpha_2_predict_2=np.zeros((d,m))
    
    Double_robust_wrong_alpha_3_predict_2=np.zeros((d,m))
    Clip_Double_robust_wrong_alpha_3_predict_2=np.zeros((d,m))
    Lambda_Double_robust_wrong_alpha_3_predict_2=np.zeros((d,m))
    for i in range(m):     
        a=np.array(Y_2.iloc[i,:])
        b=predicts_2[i,:]
        c=a-b
        #d_2.append(np.dot(Sigma_matrix_inverse,Z_2[:,i])) 
        #Vanilla_Double_robust_predict_2_ideal[:,i]= model_estimate_2[:,i]+np.dot(Sigma_matrix_inverse,Z_2[:,i])*c
        Vanilla_Double_robust_predict_2_sample[:,i]= model_estimate_2[:,i]+np.dot(Sigma_matrix_sample_inverse,Z_2[:,i])*c
        Clip_Double_robust_predict_2_sample[:,i]= model_estimate_2[:,i]+np.dot(Modeified_sample_Sigma_inverse,Z_2[:,i])*c
        Lambda_Double_robust_predict_2_sample[:,i]= model_estimate_2[:,i]+np.dot(Sigma_matrix_adjustment_sample_lambda_1_inverse,Z_2[:,i])*c
        
        Linear_Wrong_Double_Robust_2_sample[:,i]=Linear_Wrong_model_estimate_2[:,i]+np.dot(Sigma_matrix_sample_inverse,Z_2[:,i])*(np.array(Y_2.iloc[i,:])-wrong_linear_model_predicts_2[i,:])
        Clip_Linear_Wrong_Double_Robust_2_sample[:,i]=Linear_Wrong_model_estimate_2[:,i]+np.dot(Modeified_sample_Sigma_inverse,Z_2[:,i])*(np.array(Y_2.iloc[i,:])-wrong_linear_model_predicts_2[i,:])
        Lambda_Linear_Wrong_Double_Robust_2_sample[:,i]=Linear_Wrong_model_estimate_2[:,i]+np.dot(Sigma_matrix_adjustment_sample_lambda_1_inverse,Z_2[:,i])*(np.array(Y_2.iloc[i,:])-wrong_linear_model_predicts_2[i,:])
        
        Double_robust_wrong_alpha_1_predict_2[:,i] = W_wrong_alpha_1_2[:,i]+np.dot(Sigma_matrix_sample_inverse,Z_2[:,i])*(np.array(Y_2.iloc[i,:])-predicts_wrong_model_alpha_1[i,:])
        Clip_Double_robust_wrong_alpha_1_predict_2[:,i] = W_wrong_alpha_1_2[:,i]+np.dot(Modeified_sample_Sigma_inverse,Z_2[:,i])*(np.array(Y_2.iloc[i,:])-predicts_wrong_model_alpha_1[i,:])
        Lambda_Double_robust_wrong_alpha_1_predict_2[:,i] = W_wrong_alpha_1_2[:,i]+np.dot(Sigma_matrix_adjustment_sample_lambda_1_inverse,Z_2[:,i])*(np.array(Y_2.iloc[i,:])-predicts_wrong_model_alpha_1[i,:])
        
        Double_robust_wrong_alpha_2_predict_2[:,i] = W_wrong_alpha_2_2[:,i]+np.dot(Sigma_matrix_sample_inverse,Z_2[:,i])*(np.array(Y_2.iloc[i,:])-predicts_wrong_model_alpha_2[i,:])
        Clip_Double_robust_wrong_alpha_2_predict_2[:,i] = W_wrong_alpha_2_2[:,i]+np.dot(Modeified_sample_Sigma_inverse,Z_2[:,i])*(np.array(Y_2.iloc[i,:])-predicts_wrong_model_alpha_2[i,:])
        Lambda_Double_robust_wrong_alpha_2_predict_2[:,i] = W_wrong_alpha_2_2[:,i]+np.dot(Sigma_matrix_adjustment_sample_lambda_1_inverse,Z_2[:,i])*(np.array(Y_2.iloc[i,:])-predicts_wrong_model_alpha_2[i,:])
        
        Double_robust_wrong_alpha_3_predict_2[:,i] = W_wrong_alpha_3_2[:,i]+np.dot(Sigma_matrix_sample_inverse,Z_2[:,i])*(np.array(Y_2.iloc[i,:])-predicts_wrong_model_alpha_3[i,:])
        Clip_Double_robust_wrong_alpha_3_predict_2[:,i] = W_wrong_alpha_3_2[:,i]+np.dot(Modeified_sample_Sigma_inverse,Z_2[:,i])*(np.array(Y_2.iloc[i,:])-predicts_wrong_model_alpha_3[i,:])
        Lambda_Double_robust_wrong_alpha_3_predict_2[:,i] = W_wrong_alpha_3_2[:,i]+np.dot(Sigma_matrix_adjustment_sample_lambda_1_inverse,Z_2[:,i])*(np.array(Y_2.iloc[i,:])-predicts_wrong_model_alpha_3[i,:])
            
    return(model_estimate_2,Vanilla_Double_robust_predict_2_sample,Clip_Double_robust_predict_2_sample,Lambda_Double_robust_predict_2_sample,
           Linear_Wrong_model_estimate_2,Linear_Wrong_Double_Robust_2_sample,Clip_Linear_Wrong_Double_Robust_2_sample,Lambda_Linear_Wrong_Double_Robust_2_sample,
           W_wrong_alpha_1_2,Double_robust_wrong_alpha_1_predict_2,Clip_Double_robust_wrong_alpha_1_predict_2,Lambda_Linear_Wrong_Double_Robust_2_sample,
           W_wrong_alpha_2_2,Double_robust_wrong_alpha_2_predict_2,Clip_Double_robust_wrong_alpha_2_predict_2,Lambda_Double_robust_wrong_alpha_1_predict_2,
           W_wrong_alpha_3_2,Double_robust_wrong_alpha_3_predict_2,Clip_Double_robust_wrong_alpha_3_predict_2,Lambda_Double_robust_wrong_alpha_3_predict_2)

n_train_seq = np.array([400,  600,  800,  1000, 1200, 1400, 1600])
#n_train_seq = np.array([400,  600, 800])
n_holdout_seq = n_train_seq
n_train_max = 3000
n_holdout_max = 3000
n_test = 1000
runs = 50
n_jobs = 50

#numiter = 1000; batchsize = 10; long_factor = 1;

(A_mat, b_vec) = read_A_and_b()
feasible_matrix_row=pd.read_excel('feasible_matrix_row.xlsx',header=None)
feasible_matrix_column=pd.read_excel('feasible_matrix_column.xlsx',header=None)

#Removes all lines that are Nan
feasible_matrix_row=feasible_matrix_row.dropna(axis=0,how='all')
feasible_matrix_column=feasible_matrix_column.dropna(axis=0,how='all')

feasible_row=[]
feasible_column=[]
for i in range(70):
    feasible_row.append(feasible_matrix_row.iloc[5*i:5*i+5,:])
    feasible_column.append(feasible_matrix_column.iloc[4*i:4*i+4,:])

feasible_vector=[]
for i in range(70):
    temp_vector=np.zeros(40)
    temp_vector[0]=feasible_row[i].iloc[0,0]
    temp_vector[2]=feasible_row[i].iloc[0,1]
    temp_vector[4]=feasible_row[i].iloc[0,2]
    temp_vector[6]=feasible_row[i].iloc[0,3]
    temp_vector[9]=feasible_row[i].iloc[1,0]
    temp_vector[11]=feasible_row[i].iloc[1,1]
    temp_vector[13]=feasible_row[i].iloc[1,2]
    temp_vector[15]=feasible_row[i].iloc[1,3]
    temp_vector[18]=feasible_row[i].iloc[2,0]
    temp_vector[20]=feasible_row[i].iloc[2,1]
    temp_vector[22]=feasible_row[i].iloc[2,2]
    temp_vector[24]=feasible_row[i].iloc[2,3]
    temp_vector[27]=feasible_row[i].iloc[3,0]
    temp_vector[29]=feasible_row[i].iloc[3,1]
    temp_vector[31]=feasible_row[i].iloc[3,2]
    temp_vector[33]=feasible_row[i].iloc[3,3]
    temp_vector[36]=feasible_row[i].iloc[4,0]
    temp_vector[37]=feasible_row[i].iloc[4,1]
    temp_vector[38]=feasible_row[i].iloc[4,2]
    temp_vector[39]=feasible_row[i].iloc[4,3]
    temp_vector[1]=feasible_column[i].iloc[0,0]
    temp_vector[3]=feasible_column[i].iloc[0,1]
    temp_vector[5]=feasible_column[i].iloc[0,2]
    temp_vector[7]=feasible_column[i].iloc[0,3]
    temp_vector[8]=feasible_column[i].iloc[0,4]
    temp_vector[10]=feasible_column[i].iloc[1,0]
    temp_vector[12]=feasible_column[i].iloc[1,1]
    temp_vector[14]=feasible_column[i].iloc[1,2]
    temp_vector[16]=feasible_column[i].iloc[1,3]
    temp_vector[17]=feasible_column[i].iloc[1,4]
    temp_vector[19]=feasible_column[i].iloc[2,0]
    temp_vector[21]=feasible_column[i].iloc[2,1]
    temp_vector[23]=feasible_column[i].iloc[2,2]
    temp_vector[25]=feasible_column[i].iloc[2,3]
    temp_vector[26]=feasible_column[i].iloc[2,4]
    temp_vector[28]=feasible_column[i].iloc[3,0]
    temp_vector[30]=feasible_column[i].iloc[3,1]
    temp_vector[32]=feasible_column[i].iloc[3,2]
    temp_vector[34]=feasible_column[i].iloc[3,3]
    temp_vector[35]=feasible_column[i].iloc[3,4]
    feasible_vector.append(temp_vector)
    
#Check feasible vector
count=0
for i in range(70):
    if np.dot(A_mat,feasible_vector[i]).all()==b_vec.all():
        count=count+1
        #print(count)
Sigma_matrix=np.zeros((40,40))
for i in range(70):
    temp_matrix=np.outer(feasible_vector[i],feasible_vector[i])
    Sigma_matrix=Sigma_matrix+temp_matrix
    
Sigma_matrix=Sigma_matrix/70
rank = np.linalg.matrix_rank(Sigma_matrix)
print(rank)

p = 3
polykernel_degree = 3
grid_dim = 5
noise_half_width = 0.25
constant  = 3
verbose = False 
(A_mat, b_vec) = read_A_and_b()
(n_nodes, n_edges) = A_mat.shape
noise_high=0.5
noise_low=-0.5


X = np.random.normal(size=(p, 50))
poly = PolynomialFeatures(degree = polykernel_degree, interaction_only = True, include_bias = False)
X_new = np.transpose(poly.fit_transform(np.transpose(X)))
(p2, _) = X_new.shape

np.random.seed(18)
B_true = np.random.rand(n_edges, p2) 
data_train_1 = [generate_data_interactive(noise_low,noise_high,B_true, n_train_max, p, feasible_vector,polykernel_degree = polykernel_degree, noise_half_width = noise_half_width, constant = constant) for run in range(runs)]
data_holdout_1 = [generate_data_interactive(noise_low,noise_high,B_true, n_holdout_max, p, feasible_vector,polykernel_degree = polykernel_degree, noise_half_width = noise_half_width, constant = constant) for run in range(runs)]
data_test_1 = generate_data_interactive(noise_low,noise_high,B_true, n_test, p,feasible_vector, polykernel_degree = polykernel_degree, noise_half_width = noise_half_width, constant = constant) 

data_train_2 = [generate_data_interactive(noise_low,noise_high,B_true, n_train_max, p, feasible_vector,polykernel_degree = polykernel_degree, noise_half_width = noise_half_width, constant = constant) for run in range(runs)]
data_holdout_2 = [generate_data_interactive(noise_low,noise_high,B_true, n_holdout_max, p, feasible_vector,polykernel_degree = polykernel_degree, noise_half_width = noise_half_width, constant = constant) for run in range(runs)]
data_test_2 = generate_data_interactive(noise_low,noise_high,B_true, n_test, p,feasible_vector, polykernel_degree = polykernel_degree, noise_half_width = noise_half_width, constant = constant) 

data_test=[]
for i in range(len(data_test_1)):
  data_test.append(np.hstack((data_test_1[i],data_test_2[i])))

lambda_max = 100
num_lambda = 10
lambda_min_ratio = 1e-3
lambda_min = lambda_max*lambda_min_ratio
lambdas = np.exp(np.linspace(np.log(lambda_min), np.log(lambda_max), num = num_lambda))
lambdas = np.round(lambdas, 2)
lambdas = np.concatenate((np.array([0, 0.001, 0.01]), lambdas))
gammas = np.array([0.01, 0.1, 0.5, 1, 2])


output = "experiment_No_kernel.txt"
with open(output, 'w') as f:
    print("start", file = f)
    
################
#  No kernel  ##
################
with open(output, 'a') as f:
    print("################" , file = f)
    print("#  no_kernel   #", file = f)
    print("################" , file = f)

regret_all_Direct_Model = []
validation_all_Direct_Model = []
time_all_Direct_Model = []
loss_all_Direct_Model = []

regret_all_Sample_Double_Robust = []
validation_all_Sample_Double_Robust = []
time_all_Sample_Double_Robust = []
loss_all_Sample_Double_Robust = []

Clip_regret_all_Sample_Double_Robust = []
Clip_validation_all_Sample_Double_Robust = []
Clip_time_all_Sample_Double_Robust = []
Clip_loss_all_Sample_Double_Robust = []

Lambda_regret_all_Sample_Double_Robust = []
Lambda_validation_all_Sample_Double_Robust = []
Lambda_time_all_Sample_Double_Robust = []
Lambda_loss_all_Sample_Double_Robust = []


regret_all_Linear_Wrong_Direct_Model = []
validation_all_Linear_Wrong_Direct_Model = []
time_all_Linear_Wrong_Direct_Model = []
loss_all_Linear_Wrong_Direct_Model = []

regret_all_Linear_Wrong_Double_Robust_Sample_Sigma = []
validation_all_Linear_Wrong_Double_Robust_Sample_Sigma = []
time_all_Linear_Wrong_Double_Robust_Sample_Sigma = []
loss_all_Linear_Wrong_Double_Robust_Sample_Sigma = []

Clip_regret_all_Linear_Wrong_Double_Robust_Sample_Sigma = []
Clip_validation_all_Linear_Wrong_Double_Robust_Sample_Sigma = []
Clip_time_all_Linear_Wrong_Double_Robust_Sample_Sigma = []
Clip_loss_all_Linear_Wrong_Double_Robust_Sample_Sigma = []

Lambda_regret_all_Linear_Wrong_Double_Robust_Sample_Sigma = []
Lambda_validation_all_Linear_Wrong_Double_Robust_Sample_Sigma = []
Lambda_time_all_Linear_Wrong_Double_Robust_Sample_Sigma = []
Lambda_loss_all_Linear_Wrong_Double_Robust_Sample_Sigma = []

for i in range(len(n_train_seq)):
    

    n_train_true = n_train_seq[i]
    n_train = int(n_train_true/2)
    n_holdout_true = n_holdout_seq[i] 
    n_holdout = int(n_holdout_true/2)
    
    #Double_robust
    res_temp_0 = Parallel(n_jobs = n_jobs, verbose = 3)(delayed(Double_robust)(data_train_1[run][0][:,:n_train],data_train_1[run][1][:,:n_train],data_train_1[run][4][:,:n_train],data_train_1[run][5][:,:n_train],data_train_1[run][6][:,:n_train],
                                                                         data_holdout_1[run][0][:,:n_holdout],data_holdout_1[run][1][:,:n_holdout],data_holdout_1[run][4][:,:n_holdout],data_holdout_1[run][5][:,:n_holdout],data_holdout_1[run][6][:,:n_holdout],
                                                                         data_train_2[run][0][:,:n_train],data_train_2[run][1][:,:n_train],data_train_2[run][4][:,:n_train],data_train_2[run][5][:,:n_train],data_train_2[run][6][:,:n_train],
                                                                         data_holdout_2[run][0][:,:n_holdout],data_holdout_2[run][1][:,:n_holdout],data_holdout_2[run][4][:,:n_holdout],data_holdout_2[run][5][:,:n_holdout],data_holdout_2[run][6][:,:n_holdout],
                                                                         data_train_2[run][3][:,:n_train],data_holdout_2[run][3][:,:n_holdout],Sigma_matrix) for run in range(runs))
    SPO_data_train_2=[[] for i in range(runs)]
    SPO_data_holdout_2=[[] for i in range(runs)]
    for run in range(runs):
        SPO_data_train_2[run].append(data_train_2[run][0][:,:n_train])
        SPO_data_train_2[run].append(data_train_2[run][1][:,:n_train])
        SPO_data_train_2[run].append(data_train_2[run][2][:,:n_train])
        SPO_data_train_2[run].append(data_train_2[run][3][:,:n_train])
        SPO_data_holdout_2[run].append(data_holdout_2[run][0][:,:n_holdout])
        SPO_data_holdout_2[run].append(data_holdout_2[run][1][:,:n_holdout])
        SPO_data_holdout_2[run].append(data_holdout_2[run][2][:,:n_holdout])
        SPO_data_holdout_2[run].append(data_holdout_2[run][3][:,:n_holdout])
        for i in range(len(res_temp_0[run])):
          SPO_data_train_2[run].append(res_temp_0[run][i][:,:n_train])
          SPO_data_holdout_2[run].append(res_temp_0[run][i][:,n_train:n_train+n_holdout])

    #cross-fitting
    res_temp_0 = Parallel(n_jobs = n_jobs, verbose = 3)(delayed(Double_robust)(data_train_2[run][0][:,:n_train],data_train_2[run][1][:,:n_train],data_train_2[run][4][:,:n_train],data_train_2[run][5][:,:n_train],data_train_2[run][6][:,:n_train],
                                                                         data_holdout_2[run][0][:,:n_holdout],data_holdout_2[run][1][:,:n_holdout],data_holdout_2[run][4][:,:n_holdout],data_holdout_2[run][5][:,:n_holdout],data_holdout_2[run][6][:,:n_holdout],
                                                                         data_train_1[run][0][:,:n_train],data_train_1[run][1][:,:n_train],data_train_1[run][4][:,:n_train],data_train_1[run][5][:,:n_train],data_train_1[run][6][:,:n_train],
                                                                         data_holdout_1[run][0][:,:n_holdout],data_holdout_1[run][1][:,:n_holdout],data_holdout_1[run][4][:,:n_holdout],data_holdout_1[run][5][:,:n_holdout],data_holdout_1[run][6][:,:n_holdout],
                                                                         data_train_1[run][3][:,:n_train],data_holdout_1[run][3][:,:n_holdout],Sigma_matrix) for run in range(runs))
  
    
    SPO_data_train_1=[[] for i in range(runs)]
    SPO_data_holdout_1=[[] for i in range(runs)]
    for run in range(runs):
        SPO_data_train_1[run].append(data_train_1[run][0][:,:n_train])
        SPO_data_train_1[run].append(data_train_1[run][1][:,:n_train])
        SPO_data_train_1[run].append(data_train_1[run][2][:,:n_train])
        SPO_data_train_1[run].append(data_train_1[run][3][:,:n_train])
        SPO_data_holdout_1[run].append(data_holdout_1[run][0][:,:n_holdout])
        SPO_data_holdout_1[run].append(data_holdout_1[run][1][:,:n_holdout])
        SPO_data_holdout_1[run].append(data_holdout_1[run][2][:,:n_holdout])
        SPO_data_holdout_1[run].append(data_holdout_1[run][3][:,:n_holdout])
        for i in range(len(res_temp_0[run])):
          SPO_data_train_1[run].append(res_temp_0[run][i][:,:n_train])
          SPO_data_holdout_1[run].append(res_temp_0[run][i][:,n_train:n_train+n_holdout])

    data_train=[[] for i in range(runs)]
    data_holdout=[[] for i in range(runs)]
    for run in range(runs):
       for i in range(len(SPO_data_train_1[run])):
         data_train[run].append(np.hstack((SPO_data_train_1[run][i],SPO_data_train_2[run][i])))
         data_holdout[run].append(np.hstack((SPO_data_holdout_1[run][i],SPO_data_holdout_2[run][i])))


    n_train=n_train*2
    n_holdout=n_holdout*2
    print(n_holdout)
    print(n_train)
    

    with open(output, 'a') as f:
        print("n_train", n_train, file = f)
        print("n_train", n_train)
    time1 = time.time()

    res_temp = Parallel(n_jobs = n_jobs, verbose = 3)(delayed(replication_no_kernel)(
                            A_mat, b_vec, B_true, 
                            data_train[run][0][:, :n_train], data_train[run][1][:, :n_train], data_train[run][4][:, :n_train], data_train[run][3][:, :n_train],
                            data_holdout[run][0][:, :n_holdout], data_holdout[run][1][:, :n_holdout], data_holdout[run][4][:, :n_holdout], data_holdout[run][3][:, :n_holdout],
                            data_test[0], data_test[1], data_test[2], data_test[3],
                            # numiter = numiter, batchsize = batchsize, long_factor = long_factor,
                            #   loss_stop = loss_stop, tol = tol, stop = stop,
                            gammas = gammas, lambdas = lambdas, lambda_max = None, lambda_min_ratio = None, num_lambda = None,
                            verbose = verbose) for run in range(runs))
    
    regret_temp = [res_0[0] for res_0 in res_temp]
    regret_temp = [pd.DataFrame(res_0, index = [0]) for res_0 in regret_temp]
    regret_temp = pd.concat(regret_temp)
    
    validation_temp = [res_0[1] for res_0 in res_temp]
    validation_temp = [pd.DataFrame(res_0, index = [0]) for res_0 in validation_temp]
    validation_temp = pd.concat(validation_temp)
    
    time_temp = [res_0[2] for res_0 in res_temp]
    time_temp= [pd.DataFrame(res_0, index = [0]) for res_0 in time_temp]
    time_temp = pd.concat(time_temp)
    
    loss_temp = [res_0[3] for res_0 in res_temp]
    
    regret_temp["n"] = n_train
    validation_temp["n"] = n_train
    time_temp["n"] = n_train
    
    with open(output, 'a') as f:
        print("Direct Model")
        print("Direct Model", file = f)
        print("total time: ", time.time() - time1, file = f)
        print("seperate time", file = f)
        print(time_temp.mean(), file = f)
        print("-------", file = f)
        print("mean regret: ", file = f)
        print(regret_temp.mean(), file = f)      
        print("-------", file = f)
        print("        ", file = f)
        print("        ", file = f)
        
    regret_all_Direct_Model.append(regret_temp)
    validation_all_Direct_Model.append(validation_temp)
    time_all_Direct_Model.append(time_temp)
    loss_all_Direct_Model.append(loss_temp)
    
    pd.concat(regret_all_Direct_Model).to_csv("regret_no_kernel_Direct_Model.csv", index = False)
    pd.concat(validation_all_Direct_Model).to_csv("validation_no_kernel_Direct_Model.csv", index = False)
    pd.concat(time_all_Direct_Model).to_csv("time_no_kernel_Direct_Model.csv", index = False)
    pickle.dump(loss_all_Direct_Model, open("loss_no_kernel_Direct_Model.pkl", "wb"))  
    
    
    with open(output, 'a') as f:
        print("n_train", n_train, file = f)
        print("n_train", n_train)
    time1 = time.time()

    res_temp = Parallel(n_jobs = n_jobs, verbose = 3)(delayed(replication_no_kernel)(
                            A_mat, b_vec, B_true, 
                            data_train[run][0][:, :n_train], data_train[run][1][:, :n_train], data_train[run][5][:, :n_train], data_train[run][3][:, :n_train],
                            data_holdout[run][0][:, :n_holdout], data_holdout[run][1][:, :n_holdout], data_holdout[run][5][:, :n_holdout], data_holdout[run][3][:, :n_holdout],
                            data_test[0], data_test[1], data_test[2], data_test[3],
                            # numiter = numiter, batchsize = batchsize, long_factor = long_factor,
                            #   loss_stop = loss_stop, tol = tol, stop = stop,
                            gammas = gammas, lambdas = lambdas, lambda_max = None, lambda_min_ratio = None, num_lambda = None,
                            verbose = verbose) for run in range(runs))
    
    regret_temp = [res_0[0] for res_0 in res_temp]
    regret_temp = [pd.DataFrame(res_0, index = [0]) for res_0 in regret_temp]
    regret_temp = pd.concat(regret_temp)
    
    validation_temp = [res_0[1] for res_0 in res_temp]
    validation_temp = [pd.DataFrame(res_0, index = [0]) for res_0 in validation_temp]
    validation_temp = pd.concat(validation_temp)
    
    time_temp = [res_0[2] for res_0 in res_temp]
    time_temp = [pd.DataFrame(res_0, index = [0]) for res_0 in time_temp]
    time_temp = pd.concat(time_temp)
    
    loss_temp = [res_0[3] for res_0 in res_temp]
    
    regret_temp["n"] = n_train
    validation_temp["n"] = n_train
    time_temp["n"] = n_train
    
    with open(output, 'a') as f:
        print("Double Robust With Sample Sigma")
        print("Double Robust With Sample Sigma", file = f)
        print("total time: ", time.time() - time1, file = f)
        print("seperate time", file = f)
        print(time_temp.mean(), file = f)
        print("-------", file = f)
        print("mean regret: ", file = f)
        print(regret_temp.mean(), file = f)      
        print("-------", file = f)
        print("        ", file = f)
        print("        ", file = f)
        
    regret_all_Sample_Double_Robust.append(regret_temp)
    validation_all_Sample_Double_Robust.append(validation_temp)
    time_all_Sample_Double_Robust.append(time_temp)
    loss_all_Sample_Double_Robust.append(loss_temp)
    
    pd.concat(regret_all_Sample_Double_Robust).to_csv("regret_no_kernel_Sample_Double_Robust.csv", index = False)
    pd.concat(validation_all_Sample_Double_Robust).to_csv("validation_no_kernel_Sample_Double_Robust.csv", index = False)
    pd.concat(time_all_Sample_Double_Robust).to_csv("time_no_kernel_Sample_Double_Robust.csv", index = False)
    pickle.dump(loss_all_Sample_Double_Robust, open("loss_no_kernel_Sample_Double_Robust.pkl", "wb"))      
    
    with open(output, 'a') as f:
        print("n_train", n_train, file = f)
        print("n_train", n_train)
    time1 = time.time()

    res_temp = Parallel(n_jobs = n_jobs, verbose = 3)(delayed(replication_no_kernel)(
                            A_mat, b_vec, B_true, 
                            data_train[run][0][:, :n_train], data_train[run][1][:, :n_train], data_train[run][6][:, :n_train], data_train[run][3][:, :n_train],
                            data_holdout[run][0][:, :n_holdout], data_holdout[run][1][:, :n_holdout], data_holdout[run][6][:, :n_holdout], data_holdout[run][3][:, :n_holdout],
                            data_test[0], data_test[1], data_test[2], data_test[3],
                            # numiter = numiter, batchsize = batchsize, long_factor = long_factor,
                            #   loss_stop = loss_stop, tol = tol, stop = stop,
                            gammas = gammas, lambdas = lambdas, lambda_max = None, lambda_min_ratio = None, num_lambda = None,
                            verbose = verbose) for run in range(runs))
    
    regret_temp = [res_0[0] for res_0 in res_temp]
    regret_temp = [pd.DataFrame(res_0, index = [0]) for res_0 in regret_temp]
    regret_temp = pd.concat(regret_temp)
    
    validation_temp = [res_0[1] for res_0 in res_temp]
    validation_temp = [pd.DataFrame(res_0, index = [0]) for res_0 in validation_temp]
    validation_temp = pd.concat(validation_temp)
    
    time_temp = [res_0[2] for res_0 in res_temp]
    time_temp = [pd.DataFrame(res_0, index = [0]) for res_0 in time_temp]
    time_temp = pd.concat(time_temp)
    
    loss_temp = [res_0[3] for res_0 in res_temp]
    
    regret_temp["n"] = n_train
    validation_temp["n"] = n_train
    time_temp["n"] = n_train
    
    with open(output, 'a') as f:
        print("Double Robust With Sample Sigma Clip")
        print("Double Robust With Sample Sigma Clip", file = f)
        print("total time: ", time.time() - time1, file = f)
        print("seperate time", file = f)
        print(time_temp.mean(), file = f)
        print("-------", file = f)
        print("mean regret: ", file = f)
        print(regret_temp.mean(), file = f)      
        print("-------", file = f)
        print("        ", file = f)
        print("        ", file = f)
        
    Clip_regret_all_Sample_Double_Robust.append(regret_temp)
    Clip_validation_all_Sample_Double_Robust.append(validation_temp)
    Clip_time_all_Sample_Double_Robust.append(time_temp)
    Clip_loss_all_Sample_Double_Robust.append(loss_temp)
    
    pd.concat(Clip_regret_all_Sample_Double_Robust).to_csv("regret_no_kernel_Clip_Sample_Double_Robust.csv", index = False)
    pd.concat(Clip_validation_all_Sample_Double_Robust).to_csv("validation_no_kernel_Clip_Sample_Double_Robust.csv", index = False)
    pd.concat(Clip_time_all_Sample_Double_Robust).to_csv("time_no_kernel_Clip_Sample_Double_Robust.csv", index = False)
    pickle.dump(Clip_loss_all_Sample_Double_Robust, open("loss_no_kernel_Clip_Sample_Double_Robust.pkl", "wb"))

    with open(output, 'a') as f:
        print("n_train", n_train, file = f)
        print("n_train", n_train)
    time1 = time.time()

    res_temp = Parallel(n_jobs = n_jobs, verbose = 3)(delayed(replication_no_kernel)(
                            A_mat, b_vec, B_true, 
                            data_train[run][0][:, :n_train], data_train[run][1][:, :n_train], data_train[run][7][:, :n_train], data_train[run][3][:, :n_train],
                            data_holdout[run][0][:, :n_holdout], data_holdout[run][1][:, :n_holdout], data_holdout[run][7][:, :n_holdout], data_holdout[run][3][:, :n_holdout],
                            data_test[0], data_test[1], data_test[2], data_test[3],
                            # numiter = numiter, batchsize = batchsize, long_factor = long_factor,
                            #   loss_stop = loss_stop, tol = tol, stop = stop,
                            gammas = gammas, lambdas = lambdas, lambda_max = None, lambda_min_ratio = None, num_lambda = None,
                            verbose = verbose) for run in range(runs))
    
    regret_temp = [res_0[0] for res_0 in res_temp]
    regret_temp = [pd.DataFrame(res_0, index = [0]) for res_0 in regret_temp]
    regret_temp = pd.concat(regret_temp)
    
    validation_temp = [res_0[1] for res_0 in res_temp]
    validation_temp = [pd.DataFrame(res_0, index = [0]) for res_0 in validation_temp]
    validation_temp = pd.concat(validation_temp)
    
    time_temp = [res_0[2] for res_0 in res_temp]
    time_temp = [pd.DataFrame(res_0, index = [0]) for res_0 in time_temp]
    time_temp = pd.concat(time_temp)
    
    loss_temp = [res_0[3] for res_0 in res_temp]
    
    regret_temp["n"] = n_train
    validation_temp["n"] = n_train
    time_temp["n"] = n_train
    
    with open(output, 'a') as f:
        print("Double Robust With Sample Sigma Lambda")
        print("Double Robust With Sample Sigma Lambda", file = f)
        print("total time: ", time.time() - time1, file = f)
        print("seperate time", file = f)
        print(time_temp.mean(), file = f)
        print("-------", file = f)
        print("mean regret: ", file = f)
        print(regret_temp.mean(), file = f)      
        print("-------", file = f)
        print("        ", file = f)
        print("        ", file = f)
        
    Lambda_regret_all_Sample_Double_Robust.append(regret_temp)
    Lambda_validation_all_Sample_Double_Robust.append(validation_temp)
    Lambda_time_all_Sample_Double_Robust.append(time_temp)
    Lambda_loss_all_Sample_Double_Robust.append(loss_temp)
    
    pd.concat(Lambda_regret_all_Sample_Double_Robust).to_csv("regret_no_kernel_Lambda_Sample_Double_Robust.csv", index = False)
    pd.concat(Lambda_validation_all_Sample_Double_Robust).to_csv("validation_no_kernel_Lambda_Sample_Double_Robust.csv", index = False)
    pd.concat(Lambda_time_all_Sample_Double_Robust).to_csv("time_no_kernel_Lambda_Sample_Double_Robust.csv", index = False)
    pickle.dump(Lambda_loss_all_Sample_Double_Robust, open("loss_no_kernel_Lambda_Sample_Double_Robust.pkl", "wb"))


    with open(output, 'a') as f:
        print("n_train", n_train, file = f)
        print("n_train", n_train)
    time1 = time.time()

    res_temp = Parallel(n_jobs = n_jobs, verbose = 3)(delayed(replication_no_kernel)(
                            A_mat, b_vec, B_true, 
                            data_train[run][0][:, :n_train], data_train[run][1][:, :n_train], data_train[run][8][:, :n_train], data_train[run][3][:, :n_train],
                            data_holdout[run][0][:, :n_holdout], data_holdout[run][1][:, :n_holdout], data_holdout[run][8][:, :n_holdout], data_holdout[run][3][:, :n_holdout],
                            data_test[0], data_test[1], data_test[2], data_test[3],
                            # numiter = numiter, batchsize = batchsize, long_factor = long_factor,
                            #   loss_stop = loss_stop, tol = tol, stop = stop,
                            gammas = gammas, lambdas = lambdas, lambda_max = None, lambda_min_ratio = None, num_lambda = None,
                            verbose = verbose) for run in range(runs))
    
    regret_temp = [res_0[0] for res_0 in res_temp]
    regret_temp = [pd.DataFrame(res_0, index = [0]) for res_0 in regret_temp]
    regret_temp = pd.concat(regret_temp)
    
    validation_temp = [res_0[1] for res_0 in res_temp]
    validation_temp = [pd.DataFrame(res_0, index = [0]) for res_0 in validation_temp]
    validation_temp = pd.concat(validation_temp)
    
    time_temp = [res_0[2] for res_0 in res_temp]
    time_temp = [pd.DataFrame(res_0, index = [0]) for res_0 in time_temp]
    time_temp = pd.concat(time_temp)
    
    loss_temp = [res_0[3] for res_0 in res_temp]
    
    regret_temp["n"] = n_train
    validation_temp["n"] = n_train
    time_temp["n"] = n_train
    
    with open(output, 'a') as f:
        print("Linear Wrong Direct Model")
        print("Linear Wrong Direct Model",file = f)
        print("total time: ", time.time() - time1, file = f)
        print("seperate time", file = f)
        print(time_temp.mean(), file = f)
        print("-------", file = f)
        print("mean regret: ", file = f)
        print(regret_temp.mean(), file = f)      
        print("-------", file = f)
        print("        ", file = f)
        print("        ", file = f)
        
    regret_all_Linear_Wrong_Direct_Model.append(regret_temp)
    validation_all_Linear_Wrong_Direct_Model.append(validation_temp)
    time_all_Linear_Wrong_Direct_Model.append(time_temp)
    loss_all_Linear_Wrong_Direct_Model.append(loss_temp)
    
    pd.concat(regret_all_Linear_Wrong_Direct_Model).to_csv("regret_no_kernel_Linear_Wrong_Direct_Model.csv", index = False)
    pd.concat(validation_all_Linear_Wrong_Direct_Model).to_csv("validation_no_kernel_Linear_Wrong_Direct_Model.csv", index = False)
    pd.concat(time_all_Linear_Wrong_Direct_Model).to_csv("time_no_kerne_Linear_Wrong_Direct_Model.csv", index = False)
    pickle.dump(loss_all_Linear_Wrong_Direct_Model, open("loss_no_kernel_Linear_Wrong_Direct_Model.pkl", "wb"))  

    with open(output, 'a') as f:
        print("n_train", n_train, file = f)
        print("n_train", n_train)
    time1 = time.time()

    res_temp = Parallel(n_jobs = n_jobs, verbose = 3)(delayed(replication_no_kernel)(
                            A_mat, b_vec, B_true, 
                            data_train[run][0][:, :n_train], data_train[run][1][:, :n_train], data_train[run][9][:, :n_train], data_train[run][3][:, :n_train],
                            data_holdout[run][0][:, :n_holdout], data_holdout[run][1][:, :n_holdout], data_holdout[run][9][:, :n_holdout], data_holdout[run][3][:, :n_holdout],
                            data_test[0], data_test[1], data_test[2], data_test[3],
                            # numiter = numiter, batchsize = batchsize, long_factor = long_factor,
                            #   loss_stop = loss_stop, tol = tol, stop = stop,
                            gammas = gammas, lambdas = lambdas, lambda_max = None, lambda_min_ratio = None, num_lambda = None,
                            verbose = verbose) for run in range(runs))
    
    regret_temp = [res_0[0] for res_0 in res_temp]
    regret_temp = [pd.DataFrame(res_0, index = [0]) for res_0 in regret_temp]
    regret_temp = pd.concat(regret_temp)
    
    validation_temp = [res_0[1] for res_0 in res_temp]
    validation_temp = [pd.DataFrame(res_0, index = [0]) for res_0 in validation_temp]
    validation_temp = pd.concat(validation_temp)
    
    time_temp = [res_0[2] for res_0 in res_temp]
    time_temp = [pd.DataFrame(res_0, index = [0]) for res_0 in time_temp]
    time_temp = pd.concat(time_temp)
    
    loss_temp = [res_0[3] for res_0 in res_temp]
    
    regret_temp["n"] = n_train
    validation_temp["n"] = n_train
    time_temp["n"] = n_train
    
    with open(output, 'a') as f:
        print("Linear Wrong Double Robust Sample Sigma")
        print("Linear Wrong Double Robust Sample Sigma",file = f)
        print("total time: ", time.time() - time1, file = f)
        print("seperate time", file = f)
        print(time_temp.mean(), file = f)
        print("-------", file = f)
        print("mean regret: ", file = f)
        print(regret_temp.mean(), file = f)      
        print("-------", file = f)
        print("        ", file = f)
        print("        ", file = f)
        
    regret_all_Linear_Wrong_Double_Robust_Sample_Sigma.append(regret_temp)
    validation_all_Linear_Wrong_Double_Robust_Sample_Sigma.append(validation_temp)
    time_all_Linear_Wrong_Double_Robust_Sample_Sigma.append(time_temp)
    loss_all_Linear_Wrong_Double_Robust_Sample_Sigma.append(loss_temp)
    
    pd.concat(regret_all_Linear_Wrong_Double_Robust_Sample_Sigma).to_csv("regret_no_kernel_Linear_Wrong_Double_Robust_Sample_Sigma.csv", index = False)
    pd.concat(validation_all_Linear_Wrong_Double_Robust_Sample_Sigma).to_csv("validation_no_kernel_Linear_Wrong_Double_Robust_Sample_Sigma.csv", index = False)
    pd.concat(time_all_Linear_Wrong_Double_Robust_Sample_Sigma).to_csv("time_no_kerne_Linear_Wrong_Double_Robust_Sample_Sigma.csv", index = False)
    pickle.dump(loss_all_Linear_Wrong_Double_Robust_Sample_Sigma, open("loss_no_kernel_Linear_Wrong_Double_Robust_Sample_Sigma.pkl", "wb")) 
    
   
    with open(output, 'a') as f:
        print("n_train", n_train, file = f)
        print("n_train", n_train)
    time1 = time.time()

    res_temp = Parallel(n_jobs = n_jobs, verbose = 3)(delayed(replication_no_kernel)(
                            A_mat, b_vec, B_true, 
                            data_train[run][0][:, :n_train], data_train[run][1][:, :n_train], data_train[run][10][:, :n_train], data_train[run][3][:, :n_train],
                            data_holdout[run][0][:, :n_holdout], data_holdout[run][1][:, :n_holdout], data_holdout[run][10][:, :n_holdout], data_holdout[run][3][:, :n_holdout],
                            data_test[0], data_test[1], data_test[2], data_test[3],
                            # numiter = numiter, batchsize = batchsize, long_factor = long_factor,
                            #   loss_stop = loss_stop, tol = tol, stop = stop,
                            gammas = gammas, lambdas = lambdas, lambda_max = None, lambda_min_ratio = None, num_lambda = None,
                            verbose = verbose) for run in range(runs))
    
    regret_temp = [res_0[0] for res_0 in res_temp]
    regret_temp = [pd.DataFrame(res_0, index = [0]) for res_0 in regret_temp]
    regret_temp = pd.concat(regret_temp)
    
    validation_temp = [res_0[1] for res_0 in res_temp]
    validation_temp = [pd.DataFrame(res_0, index = [0]) for res_0 in validation_temp]
    validation_temp = pd.concat(validation_temp)
    
    time_temp = [res_0[2] for res_0 in res_temp]
    time_temp = [pd.DataFrame(res_0, index = [0]) for res_0 in time_temp]
    time_temp = pd.concat(time_temp)
    
    loss_temp = [res_0[3] for res_0 in res_temp]
    
    regret_temp["n"] = n_train
    validation_temp["n"] = n_train
    time_temp["n"] = n_train
    
    with open(output, 'a') as f:
        print("Linear Wrong Double Robust Sample Sigma Clip")
        print("Linear Wrong Double Robust Sample Sigma Clip",file = f)
        print("total time: ", time.time() - time1, file = f)
        print("seperate time", file = f)
        print(time_temp.mean(), file = f)
        print("-------", file = f)
        print("mean regret: ", file = f)
        print(regret_temp.mean(), file = f)      
        print("-------", file = f)
        print("        ", file = f)
        print("        ", file = f)
        
    Clip_regret_all_Linear_Wrong_Double_Robust_Sample_Sigma.append(regret_temp)
    Clip_validation_all_Linear_Wrong_Double_Robust_Sample_Sigma.append(validation_temp)
    Clip_time_all_Linear_Wrong_Double_Robust_Sample_Sigma.append(time_temp)
    Clip_loss_all_Linear_Wrong_Double_Robust_Sample_Sigma.append(loss_temp)
    
    pd.concat(Clip_regret_all_Linear_Wrong_Double_Robust_Sample_Sigma).to_csv("regret_no_kernel_Clip_Linear_Wrong_Double_Robust_Sample_Sigma.csv", index = False)
    pd.concat(Clip_validation_all_Linear_Wrong_Double_Robust_Sample_Sigma).to_csv("validation_no_kernel_Clip_Linear_Wrong_Double_Robust_Sample_Sigma.csv", index = False)
    pd.concat(Clip_time_all_Linear_Wrong_Double_Robust_Sample_Sigma).to_csv("time_no_kerne_Clip_Linear_Wrong_Double_Robust_Sample_Sigma.csv", index = False)
    pickle.dump(Clip_loss_all_Linear_Wrong_Double_Robust_Sample_Sigma, open("loss_no_kernel_Clip_Linear_Wrong_Double_Robust_Sample_Sigma.pkl", "wb")) 

    with open(output, 'a') as f:
        print("n_train", n_train, file = f)
        print("n_train", n_train)
    time1 = time.time()

    res_temp = Parallel(n_jobs = n_jobs, verbose = 3)(delayed(replication_no_kernel)(
                            A_mat, b_vec, B_true, 
                            data_train[run][0][:, :n_train], data_train[run][1][:, :n_train], data_train[run][11][:, :n_train], data_train[run][3][:, :n_train],
                            data_holdout[run][0][:, :n_holdout], data_holdout[run][1][:, :n_holdout], data_holdout[run][11][:, :n_holdout], data_holdout[run][3][:, :n_holdout],
                            data_test[0], data_test[1], data_test[2], data_test[3],
                            # numiter = numiter, batchsize = batchsize, long_factor = long_factor,
                            #   loss_stop = loss_stop, tol = tol, stop = stop,
                            gammas = gammas, lambdas = lambdas, lambda_max = None, lambda_min_ratio = None, num_lambda = None,
                            verbose = verbose) for run in range(runs))
    
    regret_temp = [res_0[0] for res_0 in res_temp]
    regret_temp = [pd.DataFrame(res_0, index = [0]) for res_0 in regret_temp]
    regret_temp = pd.concat(regret_temp)
    
    validation_temp = [res_0[1] for res_0 in res_temp]
    validation_temp = [pd.DataFrame(res_0, index = [0]) for res_0 in validation_temp]
    validation_temp = pd.concat(validation_temp)
    
    time_temp = [res_0[2] for res_0 in res_temp]
    time_temp = [pd.DataFrame(res_0, index = [0]) for res_0 in time_temp]
    time_temp = pd.concat(time_temp)
    
    loss_temp = [res_0[3] for res_0 in res_temp]
    
    regret_temp["n"] = n_train
    validation_temp["n"] = n_train
    time_temp["n"] = n_train
    
    with open(output, 'a') as f:
        print("Linear Wrong Double Robust Sample Sigma Lambda")
        print("Linear Wrong Double Robust Sample Sigma Lambda",file = f)
        print("total time: ", time.time() - time1, file = f)
        print("seperate time", file = f)
        print(time_temp.mean(), file = f)
        print("-------", file = f)
        print("mean regret: ", file = f)
        print(regret_temp.mean(), file = f)      
        print("-------", file = f)
        print("        ", file = f)
        print("        ", file = f)
        
    Lambda_regret_all_Linear_Wrong_Double_Robust_Sample_Sigma.append(regret_temp)
    Lambda_validation_all_Linear_Wrong_Double_Robust_Sample_Sigma.append(validation_temp)
    Lambda_time_all_Linear_Wrong_Double_Robust_Sample_Sigma.append(time_temp)
    Lambda_loss_all_Linear_Wrong_Double_Robust_Sample_Sigma.append(loss_temp)
    
    pd.concat(Lambda_regret_all_Linear_Wrong_Double_Robust_Sample_Sigma).to_csv("regret_no_kernel_Lambda_Linear_Wrong_Double_Robust_Sample_Sigma.csv", index = False)
    pd.concat(Lambda_validation_all_Linear_Wrong_Double_Robust_Sample_Sigma).to_csv("validation_no_kernel_Lambda_Linear_Wrong_Double_Robust_Sample_Sigma.csv", index = False)
    pd.concat(Lambda_time_all_Linear_Wrong_Double_Robust_Sample_Sigma).to_csv("time_no_kerne_Lambda_Linear_Wrong_Double_Robust_Sample_Sigma.csv", index = False)
    pickle.dump(Lambda_loss_all_Linear_Wrong_Double_Robust_Sample_Sigma, open("loss_no_kernel_Lambda_Linear_Wrong_Double_Robust_Sample_Sigma.pkl", "wb")) 
