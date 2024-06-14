from utility_functions_sgd import * 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge,Lasso
import mkl
from sklearn import tree
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
    idxs_array=np.zeros((1,n))
    
    for i in range(n):
        idxs = np.random.randint(0, 70, size=1)
        temp_Y_transpose_Z_observed=np.dot(c_observed[:,i],feasible_vector[idxs[0]])
        temp_Y_transpose_Z_expected=np.dot(c_expected[:,i],feasible_vector[idxs[0]])
        Y_transpose_Z_observed[0,i]=temp_Y_transpose_Z_observed
        Y_transpose_Z_expected[0,i]=temp_Y_transpose_Z_expected
        Z[:,i]=feasible_vector[idxs[0]]
        idxs_array[0,i]=idxs
    return [X_false, X_true, c_observed, c_expected,Y_transpose_Z_observed,Y_transpose_Z_expected,Z,idxs_array]

def Fake_generate_data_interactive(noise_low,noise_high,B_true, n, p,feasible_vector, Fake_feasible_vector,new_feasible_vector_1, 
                                   new_feasible_vector_2,new_feasible_vector_3,new_feasible_vector_4,polykernel_degree = 3,noise_half_width = 0.01, constant = 3):
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
    idxs_array=np.zeros((1,n))
    True_idxs_array=np.zeros((1,n))

    for i in range(n):
        if X_false[1,i]>0 and X_false[2,i]>0:
          idxs = np.random.randint(0, 75, size=1)
          temp_Y_transpose_Z_observed=np.dot(c_observed[:,i],new_feasible_vector_1[idxs[0]])
          temp_Y_transpose_Z_expected=np.dot(c_expected[:,i],new_feasible_vector_1[idxs[0]])
          Y_transpose_Z_observed[0,i]=temp_Y_transpose_Z_observed
          Y_transpose_Z_expected[0,i]=temp_Y_transpose_Z_expected
          Z[:,i]=new_feasible_vector_1[idxs[0]]
          idxs_array[0,i]=idxs
        if X_false[1,i]>0 and X_false[2,i]<=0:
          idxs = np.random.randint(0, 75, size=1)
          temp_Y_transpose_Z_observed=np.dot(c_observed[:,i],new_feasible_vector_2[idxs[0]])
          temp_Y_transpose_Z_expected=np.dot(c_expected[:,i],new_feasible_vector_2[idxs[0]])
          Y_transpose_Z_observed[0,i]=temp_Y_transpose_Z_observed
          Y_transpose_Z_expected[0,i]=temp_Y_transpose_Z_expected
          Z[:,i]=new_feasible_vector_2[idxs[0]]
          idxs_array[0,i]=idxs
        if X_false[1,i]<=0 and  X_false[2,i]>0:
          idxs = np.random.randint(0, 100, size=1)
          temp_Y_transpose_Z_observed=np.dot(c_observed[:,i],new_feasible_vector_3[idxs[0]])
          temp_Y_transpose_Z_expected=np.dot(c_expected[:,i],new_feasible_vector_3[idxs[0]])
          Y_transpose_Z_observed[0,i]=temp_Y_transpose_Z_observed
          Y_transpose_Z_expected[0,i]=temp_Y_transpose_Z_expected
          Z[:,i]=new_feasible_vector_3[idxs[0]]
        if X_false[1,i]<=0 and  X_false[2,i]<=0:
          idxs = np.random.randint(0, 100, size=1)
          temp_Y_transpose_Z_observed=np.dot(c_observed[:,i],new_feasible_vector_4[idxs[0]])
          temp_Y_transpose_Z_expected=np.dot(c_expected[:,i],new_feasible_vector_4[idxs[0]])
          Y_transpose_Z_observed[0,i]=temp_Y_transpose_Z_observed
          Y_transpose_Z_expected[0,i]=temp_Y_transpose_Z_expected
          Z[:,i]=new_feasible_vector_4[idxs[0]]
    for i in range(n):
        for j in range(70):
            if (Z[:,i]==feasible_vector[j]).all():
                True_idxs_array[0,i]=j
    return [X_false, X_true, c_observed, c_expected,Y_transpose_Z_observed,Y_transpose_Z_expected,Z,idxs_array,True_idxs_array]
n_train_seq = np.array([400,  600,  800,  1000, 1200, 1400, 1600])
#n_train_seq = np.array([400,  600])
n_holdout_seq = n_train_seq
n_train_max = 3000
n_holdout_max = 3000
n_test = 1000
runs = 50
n_jobs = 50

#numiter = 1000; batchsize = 10; long_factor = 1;

(A_mat, b_vec) = read_A_and_b()
(A_mat_new,b_vec_new)=read_A_and_b('A_and_b_new.csv')
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
data_test_1 = generate_data_interactive(noise_low,noise_high,B_true, n_test, p,feasible_vector, polykernel_degree = polykernel_degree, noise_half_width = noise_half_width, constant = constant) 
data_test_2 = generate_data_interactive(noise_low,noise_high,B_true, n_test, p,feasible_vector, polykernel_degree = polykernel_degree, noise_half_width = noise_half_width, constant = constant) 

data_test=[]
for i in range(len(data_test_1)):
  data_test.append(np.hstack((data_test_1[i],data_test_2[i])))
  
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

c_test_exp=data_test[3]
sp_oracle = generate_sp_oracle(A_mat, b_vec, verbose)
(z_star_test, w_star_test) = oracle_dataset(c_test_exp, sp_oracle)
num_list=np.zeros(70)
for m in range(70):
  num=0
  for i in range(2*1000):
     if (w_star_test[:,i]==feasible_vector[m]).all():
         num=num+1
  num_list[m]=num

data_test_cost_exp=np.zeros((70,2*n_test))
for i in range(2*n_test):
    for j in range(70):
      data_test_cost_exp[j,i]=np.dot(data_test[3][:,i],feasible_vector[j])
data_test.append(data_test_cost_exp)

best_list=[63,26,17,1,53,14,38,37,5,50,44,42,68,64,62,33,43,25,12,18]
Fake_feasible_vector=[]
for j in range(70):
    if j not in best_list:
       Fake_feasible_vector.append(feasible_vector[j])
       

new_feasible_vector_1=[]
new_feasible_vector_2=[]
new_feasible_vector_3=[]
new_feasible_vector_4=[]

for i in range(50):
        new_feasible_vector_1.append(Fake_feasible_vector[i])
        new_feasible_vector_2.append(Fake_feasible_vector[i])
        new_feasible_vector_3.append(Fake_feasible_vector[i])
        new_feasible_vector_4.append(Fake_feasible_vector[i])
for i in range(25):
        new_feasible_vector_1.append(Fake_feasible_vector[i])
        new_feasible_vector_2.append(Fake_feasible_vector[25+i])
        new_feasible_vector_3.append(Fake_feasible_vector[i])
        new_feasible_vector_4.append(Fake_feasible_vector[25+i])
for i in range(25):        
        new_feasible_vector_3.append(Fake_feasible_vector[i])
        new_feasible_vector_4.append(Fake_feasible_vector[25+i])
data_train_1 = [Fake_generate_data_interactive(noise_low,noise_high,B_true, n_train_max, p,feasible_vector, Fake_feasible_vector,new_feasible_vector_1,new_feasible_vector_2,new_feasible_vector_3,new_feasible_vector_4,polykernel_degree = polykernel_degree, noise_half_width = noise_half_width, constant = constant) for run in range(runs)]
data_holdout_1 = [Fake_generate_data_interactive(noise_low,noise_high,B_true, n_holdout_max, p,feasible_vector, Fake_feasible_vector,new_feasible_vector_1,new_feasible_vector_2,new_feasible_vector_3,new_feasible_vector_4,polykernel_degree = polykernel_degree, noise_half_width = noise_half_width, constant = constant) for run in range(runs)]
data_train_2 = [Fake_generate_data_interactive(noise_low,noise_high,B_true, n_train_max, p,feasible_vector, Fake_feasible_vector,new_feasible_vector_1,new_feasible_vector_2,new_feasible_vector_3,new_feasible_vector_4,polykernel_degree = polykernel_degree, noise_half_width = noise_half_width, constant = constant) for run in range(runs)]
data_holdout_2 = [Fake_generate_data_interactive(noise_low,noise_high,B_true, n_holdout_max, p,feasible_vector, Fake_feasible_vector,new_feasible_vector_1,new_feasible_vector_2,new_feasible_vector_3,new_feasible_vector_4,polykernel_degree = polykernel_degree, noise_half_width = noise_half_width, constant = constant) for run in range(runs)]

lambda_max = 100
num_lambda = 10
lambda_min_ratio = 1e-3
lambda_min = lambda_max*lambda_min_ratio
lambdas = np.exp(np.linspace(np.log(lambda_min), np.log(lambda_max), num = num_lambda))
lambdas = np.round(lambdas, 2)
lambdas = np.concatenate((np.array([0, 0.001, 0.01]), lambdas))
gammas = np.array([0.01, 0.1, 0.5, 1, 2])


Sigma_matrix=np.zeros((40,40))
for i in range(50):
    temp_matrix=np.outer(Fake_feasible_vector[i],Fake_feasible_vector[i])
    Sigma_matrix=Sigma_matrix+temp_matrix
    
Sigma_matrix=Sigma_matrix/50
rank = np.linalg.matrix_rank(Sigma_matrix)
print(rank)

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

regret_all_IPW = []
validation_all_IPW = []
time_all_IPW = []
loss_all_IPW = []

regret_all_ideal_IPW = []
validation_all_ideal_IPW = []
time_all_ideal_IPW = []
loss_all_ideal_IPW = []

regret_all_Tree_IPW = []
validation_all_Tree_IPW = []
time_all_Tree_IPW = []
loss_all_Tree_IPW = []

for i in range(len(n_train_seq)):

    n_train_true = n_train_seq[i]
    n_train = int(n_train_true/2)
    n_holdout_true = n_holdout_seq[i] 
    n_holdout = int(n_holdout_true/2)
    
    data_train_1_used=[[] for i in range(runs)]
    data_train_2_used=[[] for i in range(runs)]
    data_holdout_1_used=[[] for i in range(runs)]
    data_holdout_2_used=[[] for i in range(runs)]
    data_train=[[] for i in range(runs)]
    data_holdout=[[] for i in range(runs)]
    
    for k in range(runs):
        for i in range(9):
            data_train_1_used[k].append(data_train_1[k][i][:,:n_train])
            data_train_2_used[k].append(data_train_2[k][i][:,:n_train])
            data_holdout_1_used[k].append(data_holdout_1[k][i][:,:n_holdout])
            data_holdout_2_used[k].append(data_holdout_2[k][i][:,:n_holdout])

    for k in range(runs):
        for i in range(9):
            data_train[k].append(np.hstack((data_train_1_used[k][i],data_train_2_used[k][i])))
            data_holdout[k].append(np.hstack((data_holdout_1_used[k][i],data_holdout_2_used[k][i])))
    
    data_train_X_dependence_IPW=[[[] for l in range(4)] for i in range(runs)]
    data_holdout_X_dependence_IPW=[[[] for l in range(4)] for i in range(runs)]
               
    for k in range(runs):
        numtrain_1=0
        numtrain_2=0
        numtrain_3=0
        numtrain_4=0
        for j in range(n_train_true):
            if data_train[k][0][1,j]>0 and data_train[k][0][2,j]>0:
               numtrain_1=numtrain_1+1
            if data_train[k][0][1,j]>0 and data_train[k][0][2,j]<=0:
               numtrain_2=numtrain_2+1
            if data_train[k][0][1,j]<=0 and data_train[k][0][2,j]>0:
               numtrain_3=numtrain_3+1
            if data_train[k][0][1,j]<=0 and data_train[k][0][2,j]<=0:
               numtrain_4=numtrain_4+1
        for l in range(len(data_train[k])):
            (a,b)=data_holdout[k][l].shape
            temp_array_1=np.zeros((a,numtrain_1))
            temp_array_2=np.zeros((a,numtrain_2))
            temp_array_3=np.zeros((a,numtrain_3))
            temp_array_4=np.zeros((a,numtrain_4))
            num_1=0
            num_2=0
            num_3=0
            num_4=0
            for j in range(n_train_true):    
                if data_train[k][0][1,j]>0 and data_train[k][0][2,j]>0:
                    temp_array_1[:,num_1]=data_train[k][l][:,j]
                    num_1=num_1+1
                if data_train[k][0][1,j]>0 and data_train[k][0][2,j]<=0:
                    temp_array_2[:,num_2]=data_train[k][l][:,j]
                    num_2=num_2+1
                if data_train[k][0][1,j]<=0 and data_train[k][0][2,j]>0:
                    temp_array_3[:,num_3]=data_train[k][l][:,j]
                    num_3=num_3+1
                if data_train[k][0][1,j]<=0 and data_train[k][0][2,j]<=0:
                    temp_array_4[:,num_4]=data_train[k][l][:,j]
                    num_4=num_4+1
            data_train_X_dependence_IPW[k][0].append(temp_array_1)
            data_train_X_dependence_IPW[k][1].append(temp_array_2)
            data_train_X_dependence_IPW[k][2].append(temp_array_3)
            data_train_X_dependence_IPW[k][3].append(temp_array_4)

    for k in range(runs):
        for l in range(4):
          if l==0:
             unique, counts = np.unique(data_train_X_dependence_IPW[k][l][8], return_counts=True)
             if len(unique)==70:
               IPW_score_1=counts/numtrain_1
             else:
               IPW_score_1=0.00001*np.ones(70)
               for m in range(len(unique)):
                  IPW_score_1[int(unique[m])]=counts[m]/numtrain_1
          if l==1:
             unique, counts = np.unique(data_train_X_dependence_IPW[k][l][8], return_counts=True)
             if len(unique)==70:
               IPW_score_2=counts/numtrain_2
             else:
               IPW_score_2=0.00001*np.ones(70)
               for m in range(len(unique)):
                  IPW_score_2[int(unique[m])]=counts[m]/numtrain_2
          if l==2:
             unique, counts = np.unique(data_train_X_dependence_IPW[k][l][8], return_counts=True)
             if len(unique)==70:
               IPW_score_3=counts/numtrain_3
             else:
               IPW_score_3=0.00001*np.ones(70)
               for m in range(len(unique)):
                  IPW_score_3[int(unique[m])]=counts[m]/numtrain_3
          if l==3:
             unique, counts = np.unique(data_train_X_dependence_IPW[k][l][8], return_counts=True)
             if len(unique)==70:
               IPW_score_4=counts/numtrain_4
             else:
               IPW_score_4=0.00001*np.ones(70)
               for m in range(len(unique)):
                  IPW_score_4[int(unique[m])]=counts[m]/numtrain_4
                
        data_train_cost_hat=np.zeros((70,n_train_true))
        for j in range(n_train_true):
             if data_train[k][0][1,j]>0 and data_train[k][0][2,j]>0:
               data_train_cost_hat[int(data_train[k][8][0,j]),j]=data_train[k][4][0,j]/IPW_score_1[int(data_train[k][8][0,j])]
             if data_train[k][0][1,j]>0 and data_train[k][0][2,j]<=0:
               data_train_cost_hat[int(data_train[k][8][0,j]),j]=data_train[k][4][0,j]/IPW_score_2[int(data_train[k][8][0,j])]
             if data_train[k][0][1,j]<=0 and data_train[k][0][2,j]>0:
               data_train_cost_hat[int(data_train[k][8][0,j]),j]=data_train[k][4][0,j]/IPW_score_3[int(data_train[k][8][0,j])]
             if data_train[k][0][1,j]<=0 and data_train[k][0][2,j]<=0:
               data_train_cost_hat[int(data_train[k][8][0,j]),j]=data_train[k][4][0,j]/IPW_score_4[int(data_train[k][8][0,j])]
        data_train[k].append(data_train_cost_hat)
        
    for k in range(runs):
        unique, counts = np.unique(data_train[k][8], return_counts=True)
        if len(unique)==70:
           IPW_score=counts/n_train_true
        else:
           IPW_score=0.00001*np.ones(70)
           for m in range(len(unique)):
               IPW_score[int(unique[m])]=counts[m]/n_train_true

        data_train_cost_ideal_hat=np.zeros((70,n_train_true))
        for j in range(n_train_true):
            if data_train[k][0][1,j]>0 and data_train[k][0][2,j]>0 and (data_train[k][7][0,j]<25 or data_train[k][7][0,j]>=50): 
              data_train_cost_ideal_hat[int(data_train[k][8][0,j]),j]=(75/2)*data_train[k][4][0,j]
            if data_train[k][0][1,j]>0 and data_train[k][0][2,j]>0 and (25<=data_train[k][7][0,j]<50): 
              data_train_cost_ideal_hat[int(data_train[k][8][0,j]),j]=75*data_train[k][4][0,j]
            if data_train[k][0][1,j]<0 and data_train[k][0][2,j]<=0 and data_train[k][7][0,j]>=25: 
              data_train_cost_ideal_hat[int(data_train[k][8][0,j]),j]=(75/2)*data_train[k][4][0,j]
            if data_train[k][0][1,j]<0 and data_train[k][0][2,j]<=0 and data_train[k][7][0,j]<25: 
              data_train_cost_ideal_hat[int(data_train[k][8][0,j]),j]=75*data_train[k][4][0,j]
            
            if data_train[k][0][1,j]<=0 and data_train[k][0][2,j]>0 and (data_train[k][7][0,j]<25 or data_train[k][7][0,j]>=50): 
              data_train_cost_ideal_hat[int(data_train[k][8][0,j]),j]=(100/3)*data_train[k][4][0,j]
            if data_train[k][0][1,j]<=0 and data_train[k][0][2,j]>0 and (25<=data_train[k][7][0,j]<50): 
              data_train_cost_ideal_hat[int(data_train[k][8][0,j]),j]=100*data_train[k][4][0,j]
            if data_train[k][0][1,j]<=0 and data_train[k][0][2,j]<=0 and data_train[k][7][0,j]>=25: 
              data_train_cost_ideal_hat[int(data_train[k][8][0,j]),j]=(100/3)*data_train[k][4][0,j]
            if data_train[k][0][1,j]<=0 and data_train[k][0][2,j]<=0 and data_train[k][7][0,j]<25: 
              data_train_cost_ideal_hat[int(data_train[k][8][0,j]),j]=100*data_train[k][4][0,j]
        data_train[k].append(data_train_cost_ideal_hat)
    
    
    
    
    for k in range(runs):
        numholdout_1=0
        numholdout_2=0
        numholdout_3=0
        numholdout_4=0
        for j in range(n_holdout_true):
            if data_holdout[k][0][1,j]>0 and data_holdout[k][0][2,j]>0:
               numholdout_1=numholdout_1+1
            if data_holdout[k][0][1,j]>0 and data_holdout[k][0][2,j]<=0:
               numholdout_2=numholdout_2+1
            if data_holdout[k][0][1,j]<=0 and data_holdout[k][0][2,j]>0:
               numholdout_3=numholdout_3+1
            if data_holdout[k][0][1,j]<=0 and data_holdout[k][0][2,j]<=0:
               numholdout_4=numholdout_4+1
        for l in range(len(data_holdout[k])):
            (a,b)=data_holdout[k][l].shape
            temp_array_1=np.zeros((a,numholdout_1))
            temp_array_2=np.zeros((a,numholdout_2))
            temp_array_3=np.zeros((a,numholdout_3))
            temp_array_4=np.zeros((a,numholdout_4))
            num_1=0
            num_2=0
            num_3=0
            num_4=0
            for j in range(n_holdout_true):    
                if data_holdout[k][0][1,j]>0 and data_holdout[k][0][2,j]>0:
                    temp_array_1[:,num_1]=data_holdout[k][l][:,j]
                    num_1=num_1+1
                if data_holdout[k][0][1,j]>0 and data_holdout[k][0][2,j]<=0:
                    temp_array_2[:,num_2]=data_holdout[k][l][:,j]
                    num_2=num_2+1
                if data_holdout[k][0][1,j]<=0 and data_holdout[k][0][2,j]>0:
                    temp_array_3[:,num_3]=data_holdout[k][l][:,j]
                    num_3=num_3+1
                if data_holdout[k][0][1,j]<=0 and data_holdout[k][0][2,j]<=0:
                    temp_array_4[:,num_4]=data_holdout[k][l][:,j]
                    num_4=num_4+1
            data_holdout_X_dependence_IPW[k][0].append(temp_array_1)
            data_holdout_X_dependence_IPW[k][1].append(temp_array_2)
            data_holdout_X_dependence_IPW[k][2].append(temp_array_3)
            data_holdout_X_dependence_IPW[k][3].append(temp_array_4)

    for k in range(runs):
        for l in range(4):
          if l==0:
             unique, counts = np.unique(data_holdout_X_dependence_IPW[k][l][8], return_counts=True)
             if len(unique)==70:
               IPW_score_1=counts/numholdout_1
             else:
               IPW_score_1=0.00001*np.ones(70)
               for m in range(len(unique)):
                  IPW_score_1[int(unique[m])]=counts[m]/numholdout_1
          if l==1:
             unique, counts = np.unique(data_holdout_X_dependence_IPW[k][l][8], return_counts=True)
             if len(unique)==70:
               IPW_score_2=counts/numholdout_2
             else:
               IPW_score_2=0.00001*np.ones(70)
               for m in range(len(unique)):
                  IPW_score_2[int(unique[m])]=counts[m]/numholdout_2
          if l==2:
             unique, counts = np.unique(data_holdout_X_dependence_IPW[k][l][8], return_counts=True)
             if len(unique)==70:
               IPW_score_3=counts/numholdout_3
             else:
               IPW_score_3=0.00001*np.ones(70)
               for m in range(len(unique)):
                  IPW_score_3[int(unique[m])]=counts[m]/numholdout_3
          if l==3:
             unique, counts = np.unique(data_holdout_X_dependence_IPW[k][l][8], return_counts=True)
             if len(unique)==70:
               IPW_score_4=counts/numholdout_4
             else:
               IPW_score_4=0.00001*np.ones(70)
               for m in range(len(unique)):
                  IPW_score_4[int(unique[m])]=counts[m]/numholdout_4
                
        data_holdout_cost_hat=np.zeros((70,n_holdout_true))
        for j in range(n_holdout_true):
             if data_holdout[k][0][1,j]>0 and data_holdout[k][0][2,j]>0:
               data_holdout_cost_hat[int(data_holdout[k][8][0,j]),j]=data_holdout[k][4][0,j]/IPW_score_1[int(data_holdout[k][8][0,j])]
             if data_holdout[k][0][1,j]>0 and data_holdout[k][0][2,j]<=0:
               data_holdout_cost_hat[int(data_holdout[k][8][0,j]),j]=data_holdout[k][4][0,j]/IPW_score_2[int(data_holdout[k][8][0,j])]
             if data_holdout[k][0][1,j]<=0 and data_holdout[k][0][2,j]>0:
               data_holdout_cost_hat[int(data_holdout[k][8][0,j]),j]=data_holdout[k][4][0,j]/IPW_score_3[int(data_holdout[k][8][0,j])]
             if data_holdout[k][0][1,j]<=0 and data_holdout[k][0][2,j]<=0:
               data_holdout_cost_hat[int(data_holdout[k][8][0,j]),j]=data_holdout[k][4][0,j]/IPW_score_4[int(data_holdout[k][8][0,j])]
        data_holdout[k].append(data_holdout_cost_hat)
        
    for k in range(runs):
        unique, counts = np.unique(data_holdout[k][8], return_counts=True)
        if len(unique)==70:
           IPW_score=counts/n_holdout_true
        else:
           IPW_score=0.00001*np.ones(70)
           for m in range(len(unique)):
               IPW_score[int(unique[m])]=counts[m]/n_holdout_true

        data_holdout_cost_ideal_hat=np.zeros((70,n_holdout_true))
        for j in range(n_holdout_true):
            if data_holdout[k][0][1,j]>0 and data_holdout[k][0][2,j]>0 and (data_holdout[k][7][0,j]<25 or data_holdout[k][7][0,j]>=50): 
              data_holdout_cost_ideal_hat[int(data_holdout[k][8][0,j]),j]=(75/2)*data_holdout[k][4][0,j]
            if data_holdout[k][0][1,j]>0 and data_holdout[k][0][2,j]>0 and (25<=data_holdout[k][7][0,j]<50): 
              data_holdout_cost_ideal_hat[int(data_holdout[k][8][0,j]),j]=75*data_holdout[k][4][0,j]
            if data_holdout[k][0][1,j]<0 and data_holdout[k][0][2,j]<=0 and data_holdout[k][7][0,j]>=25: 
              data_holdout_cost_ideal_hat[int(data_holdout[k][8][0,j]),j]=(75/2)*data_holdout[k][4][0,j]
            if data_holdout[k][0][1,j]<0 and data_holdout[k][0][2,j]<=0 and data_holdout[k][7][0,j]<25: 
              data_holdout_cost_ideal_hat[int(data_holdout[k][8][0,j]),j]=75*data_holdout[k][4][0,j]
            
            if data_holdout[k][0][1,j]<=0 and data_holdout[k][0][2,j]>0 and (data_holdout[k][7][0,j]<25 or data_holdout[k][7][0,j]>=50): 
              data_holdout_cost_ideal_hat[int(data_holdout[k][8][0,j]),j]=(100/3)*data_holdout[k][4][0,j]
            if data_holdout[k][0][1,j]<=0 and data_holdout[k][0][2,j]>0 and (25<=data_holdout[k][7][0,j]<50): 
              data_holdout_cost_ideal_hat[int(data_holdout[k][8][0,j]),j]=100*data_holdout[k][4][0,j]
            if data_holdout[k][0][1,j]<=0 and data_holdout[k][0][2,j]<=0 and data_holdout[k][7][0,j]>=25: 
              data_holdout_cost_ideal_hat[int(data_holdout[k][8][0,j]),j]=(100/3)*data_holdout[k][4][0,j]
            if data_holdout[k][0][1,j]<=0 and data_holdout[k][0][2,j]<=0 and data_holdout[k][7][0,j]<25: 
              data_holdout_cost_ideal_hat[int(data_holdout[k][8][0,j]),j]=100*data_holdout[k][4][0,j]
        data_holdout[k].append(data_holdout_cost_ideal_hat)
    
    for k in range(runs):
        data_train_cost_hat=np.zeros((70,n_train_true))
        clf = tree.DecisionTreeClassifier(max_depth=3,random_state=18)
        a=np.transpose(data_train[k][0][1:,:])
        b=np.transpose(data_train[k][8])
        IPW_total_estimator = clf.fit(a,b)
        a_2=np.transpose(IPW_total_estimator.predict_proba(a))
        
        unique, counts = np.unique(data_train[k][8], return_counts=True)
        for j in range(n_train_true):
            for i in range(len(unique)):
                if data_train[k][8][0,j]==unique[i]:
                    data_train_cost_hat[int(data_train[k][8][0,j]),j]=data_train[k][4][0,j]/a_2[i,j]
        data_train[k].append(data_train_cost_hat)
    
    
    for k in range(runs):
        data_holdout_cost_hat=np.zeros((70,n_holdout_true))
        clf = tree.DecisionTreeClassifier(max_depth=3,random_state=18)
        a=np.transpose(data_holdout[k][0][1:,:])
        b=np.transpose(data_holdout[k][8])
        IPW_total_estimator = clf.fit(a,b)
        a_2=np.transpose(IPW_total_estimator.predict_proba(a))
        
        unique, counts = np.unique(data_holdout[k][8], return_counts=True)
        for j in range(n_holdout_true):
            for i in range(len(unique)):
                if data_holdout[k][8][0,j]==unique[i]:
                    data_holdout_cost_hat[int(data_holdout[k][8][0,j]),j]=data_holdout[k][4][0,j]/a_2[i,j]
        data_holdout[k].append(data_holdout_cost_hat)
        
    with open(output, 'a') as f:
        print("n_train", n_train_true, file = f)
        print("n_train", n_train_true)
    time1 = time.time()

    res_temp = Parallel(n_jobs = n_jobs, verbose = 3)(delayed(replication_no_kernel)(
                            A_mat_new, b_vec_new, B_true, 
                            data_train[run][0][:, :n_train_true], data_train[run][1][:, :n_train_true], data_train[run][11][:, :n_train_true], data_train[run][10][:, :n_train_true],
                            data_holdout[run][0][:, :n_holdout_true], data_holdout[run][1][:, :n_holdout_true], data_holdout[run][11][:, :n_holdout_true], data_holdout[run][10][:, :n_holdout_true],
                            data_test[0], data_test[1], data_test[2], data_test[8],
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
    
    regret_temp["n"] = n_train_true
    validation_temp["n"] = n_train_true
    time_temp["n"] = n_train_true
    
    with open(output, 'a') as f:
        print("Tree IPW")
        print("Tree IPW", file = f)
        print("total time: ", time.time() - time1, file = f)
        print("seperate time", file = f)
        print(time_temp.mean(), file = f)
        print("-------", file = f)
        print("mean regret: ", file = f)
        print(regret_temp.mean(), file = f)      
        print("-------", file = f)
        print("        ", file = f)
        print("        ", file = f)
        
    regret_all_Tree_IPW.append(regret_temp)
    validation_all_Tree_IPW.append(validation_temp)
    time_all_Tree_IPW.append(time_temp)
    loss_all_Tree_IPW.append(loss_temp)
    
    pd.concat(regret_all_Tree_IPW).to_csv("regret_no_kernel_Tree_IPW.csv", index = False)
    pd.concat(validation_all_Tree_IPW).to_csv("validation_no_kernel_Tree_IPW.csv", index = False)
    pd.concat(time_all_Tree_IPW).to_csv("time_no_kernel_Tree_IPW.csv", index = False)
    pickle.dump(loss_all_Tree_IPW, open("loss_no_kernel_Tree_IPW.pkl", "wb"))
