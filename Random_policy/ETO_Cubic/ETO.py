from utility_functions import * 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge,Lasso
import mkl
import pandas as pd
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

def generate_sp_oracle(A_mat, b_vec, verbose = False):

    (m, d) = A_mat.shape

    model = gp.Model()
    model.setParam('OutputFlag', verbose)
    model.setParam("Threads", 1)

    w = model.addVars(d, lb = 0, ub = 1, name = 'w')
    model.update()
    for i in range(m):
        model.addConstr(gp.quicksum(A_mat[i, j] * w[j] for j in range(len(w))) == b_vec[i])
    model.update()

    def local_oracle(c):
        if len(c) != len(w):
            raise Exception("Sorry, c and w dimension mismatched")

        obj = gp.quicksum(c[i] * w[i] for i in range(len(w)))
        model.setObjective(obj, GRB.MINIMIZE)
        model.update()
        model.optimize()

        w_ast = [w[i].X for i in range(len(w))]
        z_ast = model.objVal

        return (z_ast, w_ast)
    
    return local_oracle

def oracle_dataset(c, oracle):
    (d, n) = c.shape
    z_star_data = np.zeros(n)
    w_star_data = np.zeros((d, n))
    for i in range(n):
        (z_i, w_i) = oracle(c[:,i])
        z_star_data[i] = z_i
        w_star_data[:,i] = w_i
    return (z_star_data, w_star_data)

def ETO(X_train, Y_transpose_Z_observed_train, Y_transpose_Z_expected_train,Z_train,
        X_train_2, Y_transpose_Z_observed_train_2, Y_transpose_Z_expected_train_2,Z_train_2,
        X_val, Y_transpose_Z_observed_val, Y_transpose_Z_expected_val,Z_val,
        X_val_2, Y_transpose_Z_observed_val_2, Y_transpose_Z_expected_val_2,Z_val_2,
        X_test,c_test_exp,lambda_max = 100, lambda_min_ratio = 1e-6, num_lambda = 10):
    
    sp_oracle = generate_sp_oracle(A_mat, b_vec, verbose)
    (z_star_test, w_star_test) = oracle_dataset(c_test_exp, sp_oracle)
    zstar_avg_test = np.mean(z_star_test)

    X=np.hstack((X_train,X_train_2))
    Z=np.hstack((Z_train,Z_train_2))
    Y=np.transpose(np.hstack((Y_transpose_Z_observed_train,Y_transpose_Z_observed_train_2)))
    
    X_val=np.hstack((X_val,X_val_2))
    Z_val=np.hstack((Z_val,Z_val_2))
    Y_val=np.transpose(np.hstack((Y_transpose_Z_observed_val,Y_transpose_Z_observed_val_2)))

    lambda_min = lambda_max*lambda_min_ratio
    lambdas = np.exp(np.linspace(np.log(lambda_min), np.log(lambda_max), num = num_lambda))
    num_lambda = len(lambdas)
    
    validation_loss_list = np.zeros(num_lambda)
    
    (p,m)=X.shape
    (d,m)=Z.shape
    new_X_train=[]
    new_X_valid=[]
    for i in range(m):
        temp_X_train=[]
        temp_X_valid=[]
        for j in range(p):
           for k in range(d):
               temp_X_train.append(X[j,i]*Z[k,i])
               temp_X_valid.append(X_val[j,i]*Z_val[k,i])
        new_X_train.append(temp_X_train)
        new_X_valid.append(temp_X_valid)
    data_X_train = np.zeros((m,p*d))
    data_X_vaild = np.zeros((m,p*d))
    
    for i in range(m):
        data_X_train[i,:] = new_X_train[i]
        data_X_vaild[i,:] = new_X_valid[i]
    
    Y_train=pd.DataFrame(Y)
    data_X_train=pd.DataFrame(data_X_train)
    for i in range(num_lambda):
        cur_lambda = lambdas[i]
        model = Ridge(alpha = cur_lambda/2, fit_intercept = False)
        model.fit(data_X_train,Y_train)
        c_val_hat=model.predict(data_X_vaild)
        err = c_val_hat - Y_val
        validation_loss_list[i] = np.sqrt(np.mean(np.sum(np.power(err, 2), axis = 1)))
    best_ind = np.argmin(validation_loss_list)
    best_lambda = lambdas[best_ind]
    ridge_linear_best = Ridge(alpha = best_lambda/2, fit_intercept = False)
    ridge_linear_best.fit(data_X_train,Y_train)
    
    parameters = ridge_linear_best.coef_ 
    W_matrix=np.zeros((p,d))
    for i in range(p):
        W_matrix[i,:]=parameters[0,i*d:i*d+d]
    model_estimate_test=np.dot(np.transpose(X_test),W_matrix)
    model_estimate_test=np.transpose(model_estimate_test)
    (edge, n_test) = c_test_exp.shape
    spo_sum = 0
    for i in range(n_test):
        (z_oracle, w_oracle) = sp_oracle(model_estimate_test[:, i])
        spo_loss_cur = np.dot(c_test_exp[:,i], w_oracle) - z_star_test[i]
        spo_sum = spo_sum + spo_loss_cur
    
    spo_loss_avg = spo_sum/n_test
    regret_all = {"zstar_avg_test": zstar_avg_test,
                 "ETO_regret": spo_loss_avg}
    best_lambda = {"ETO_lambda":best_lambda}
    validation_all = {"ETO_validation_loss":validation_loss_list}

    return (regret_all,best_lambda,validation_all)


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

regret_all_Correct_ETO=[]
lambda_all_Correct_ETO=[]

regret_all_Wrong_ETO=[]
lambda_all_Wrong_ETO=[]

output = "experiment_No_kernel.txt"
with open(output, 'w') as f:
    print("start", file = f)
    
################
#  No kernel  ##
################
with open(output, 'a') as f:
    print("################" , file = f)
    print("#  no_kernel ETO   #", file = f)
    print("################" , file = f)


for i in range(len(n_train_seq)):
    

    n_train_true = n_train_seq[i]
    n_train = int(n_train_true/2)
    n_holdout_true = n_holdout_seq[i] 
    n_holdout = int(n_holdout_true/2)
    
    with open(output, 'a') as f:
        print("n_train", n_train*2, file = f)
        print("n_train", n_train*2)
    time1 = time.time()
    
    #Correct ETO
    res_temp_0 = Parallel(n_jobs = n_jobs, verbose = 3)(delayed(ETO)(data_train_1[run][1][:,:n_train],data_train_1[run][4][:,:n_train],data_train_1[run][5][:,:n_train],data_train_1[run][6][:,:n_train],
                                                                             data_train_2[run][1][:,:n_train],data_train_2[run][4][:,:n_train],data_train_2[run][5][:,:n_train],data_train_2[run][6][:,:n_train],
                                                                         data_holdout_1[run][1][:,:n_holdout],data_holdout_1[run][4][:,:n_holdout],data_holdout_1[run][5][:,:n_holdout],data_holdout_1[run][6][:,:n_holdout],
                                                                         data_holdout_2[run][1][:,:n_holdout],data_holdout_2[run][4][:,:n_holdout],data_holdout_2[run][5][:,:n_holdout],data_holdout_2[run][6][:,:n_holdout],
                                                                         data_test[1],data_test[3]) for run in range(runs))
    regret_temp = [res_0[0] for res_0 in res_temp_0]
    regret_temp = [pd.DataFrame(res_0, index = [0]) for res_0 in regret_temp]
    regret_temp = pd.concat(regret_temp)
    
    lambda_temp = [res_0[1] for res_0 in res_temp_0]
    lambda_temp = [pd.DataFrame(res_0, index = [0]) for res_0 in lambda_temp]
    lambda_temp = pd.concat(lambda_temp)
    
    regret_temp["n"] = n_train*2
    lambda_temp["n"] = n_train*2
    
    with open(output, 'a') as f:
        print("Correct ETO")
        print("Correct ETO", file = f)
        print("total time: ", time.time() - time1, file = f)
        print("-------", file = f)
        print("mean regret: ", file = f)
        print(regret_temp.mean(), file = f)      
        print("-------", file = f)
        print("        ", file = f)
        print("        ", file = f)
        
    regret_all_Correct_ETO.append(regret_temp)
    lambda_all_Correct_ETO.append(lambda_temp)
    
    pd.concat(regret_all_Correct_ETO).to_csv("regret_all_Correct_ETO.csv", index = False)
    pd.concat(lambda_all_Correct_ETO).to_csv("lambda_all_Correct_ETO.csv", index = False)
    
    with open(output, 'a') as f:
        print("n_train", n_train*2, file = f)
        print("n_train", n_train*2)
    time1 = time.time()
    
    #Wrong ETO
    res_temp_0 = Parallel(n_jobs = n_jobs, verbose = 3)(delayed(ETO)(data_train_1[run][0][:,:n_train],data_train_1[run][4][:,:n_train],data_train_1[run][5][:,:n_train],data_train_1[run][6][:,:n_train],
                                                                             data_train_2[run][0][:,:n_train],data_train_2[run][4][:,:n_train],data_train_2[run][5][:,:n_train],data_train_2[run][6][:,:n_train],
                                                                         data_holdout_1[run][0][:,:n_holdout],data_holdout_1[run][4][:,:n_holdout],data_holdout_1[run][5][:,:n_holdout],data_holdout_1[run][6][:,:n_holdout],
                                                                         data_holdout_2[run][0][:,:n_holdout],data_holdout_2[run][4][:,:n_holdout],data_holdout_2[run][5][:,:n_holdout],data_holdout_2[run][6][:,:n_holdout],
                                                                         data_test[0],data_test[3]) for run in range(runs))
    regret_temp = [res_0[0] for res_0 in res_temp_0]
    regret_temp = [pd.DataFrame(res_0, index = [0]) for res_0 in regret_temp]
    regret_temp = pd.concat(regret_temp)
    
    lambda_temp = [res_0[1] for res_0 in res_temp_0]
    lambda_temp = [pd.DataFrame(res_0, index = [0]) for res_0 in lambda_temp]
    lambda_temp = pd.concat(lambda_temp)
    
    regret_temp["n"] = n_train*2
    lambda_temp["n"] = n_train*2
    
    with open(output, 'a') as f:
        print("Wrong ETO")
        print("Wrong ETO", file = f)
        print("total time: ", time.time() - time1, file = f)
        print("-------", file = f)
        print("mean regret: ", file = f)
        print(regret_temp.mean(), file = f)      
        print("-------", file = f)
        print("        ", file = f)
        print("        ", file = f)
        
    regret_all_Wrong_ETO.append(regret_temp)
    lambda_all_Wrong_ETO.append(lambda_temp)
    
    pd.concat(regret_all_Wrong_ETO).to_csv("regret_all_Wrong_ETO.csv", index = False)
    pd.concat(lambda_all_Wrong_ETO).to_csv("lambda_all_Wrong_ETO.csv", index = False)