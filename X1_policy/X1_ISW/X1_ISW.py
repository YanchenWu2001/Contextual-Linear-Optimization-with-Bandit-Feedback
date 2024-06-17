from utility_functions import * 
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
        #这里我就把70个可行解的方案在feasible list中的索引作为标签
        temp_Y_transpose_Z_observed=np.dot(c_observed[:,i],feasible_vector[idxs[0]])
        temp_Y_transpose_Z_expected=np.dot(c_expected[:,i],feasible_vector[idxs[0]])
        Y_transpose_Z_observed[0,i]=temp_Y_transpose_Z_observed
        Y_transpose_Z_expected[0,i]=temp_Y_transpose_Z_expected
        Z[:,i]=feasible_vector[idxs[0]]
        idxs_array[0,i]=idxs
    return [X_false, X_true, c_observed, c_expected,Y_transpose_Z_observed,Y_transpose_Z_expected,Z,idxs_array]

def Fake_generate_data_interactive(noise_low,noise_high,B_true, n, p,feasible_vector, Fake_feasible_vector,new_feasible_vector_1, 
                                   new_feasible_vector_2,polykernel_degree = 3,noise_half_width = 0.01, constant = 3):
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
        if X_false[1,i]>0:
          idxs = np.random.randint(0, 75, size=1)
          temp_Y_transpose_Z_observed=np.dot(c_observed[:,i],new_feasible_vector_1[idxs[0]])
          temp_Y_transpose_Z_expected=np.dot(c_expected[:,i],new_feasible_vector_1[idxs[0]])
          Y_transpose_Z_observed[0,i]=temp_Y_transpose_Z_observed
          Y_transpose_Z_expected[0,i]=temp_Y_transpose_Z_expected
          Z[:,i]=new_feasible_vector_1[idxs[0]]
          idxs_array[0,i]=idxs
        if X_false[1,i]<=0:
          idxs = np.random.randint(0, 75, size=1)
          temp_Y_transpose_Z_observed=np.dot(c_observed[:,i],new_feasible_vector_2[idxs[0]])
          temp_Y_transpose_Z_expected=np.dot(c_expected[:,i],new_feasible_vector_2[idxs[0]])
          Y_transpose_Z_observed[0,i]=temp_Y_transpose_Z_observed
          Y_transpose_Z_expected[0,i]=temp_Y_transpose_Z_expected
          Z[:,i]=new_feasible_vector_2[idxs[0]]
          idxs_array[0,i]=idxs
          
    for i in range(n):
        for j in range(70):
            if (Z[:,i]==feasible_vector[j]).all():
                True_idxs_array[0,i]=j
    return [X_false, X_true, c_observed, c_expected,Y_transpose_Z_observed,Y_transpose_Z_expected,Z,idxs_array,True_idxs_array]

def Double_robust(X_train_False,X_train, Y_transpose_Z_observed_train, Y_transpose_Z_expected_train,Z_train,Z_train_idxs_1,
                  X_val_False,X_val, Y_transpose_Z_observed_val, Y_transpose_Z_expected_val,Z_val,Z_val_idxs_1,
                  X_train_False_2,X_train_2, Y_transpose_Z_observed_train_2, Y_transpose_Z_expected_train_2,Z_train_2,Z_train_idxs_2,
                  X_val_False_2,X_val_2, Y_transpose_Z_observed_val_2, Y_transpose_Z_expected_val_2,Z_val_2,Z_val_idxs_2,
                  Train_expected_edge_cost_2,Val_expected_edge_cost_2,feasible_vector,Sigma_matrix=False):
    X=np.hstack((X_train,X_val))
    Z=np.hstack((Z_train,Z_val))
    Y=np.transpose(np.hstack((Y_transpose_Z_observed_train,Y_transpose_Z_observed_val)))
    X_False=np.hstack((X_train_False,X_val_False))
    Z_idxs=np.hstack((Z_train_idxs_1,Z_val_idxs_1))
    
    X_2=np.hstack((X_train_2,X_val_2))
    Z_2=np.hstack((Z_train_2,Z_val_2))
    Y_2=np.transpose(np.hstack((Y_transpose_Z_observed_train_2,Y_transpose_Z_observed_val_2)))
    Y_2_expected=np.transpose(np.hstack((Y_transpose_Z_expected_train_2,Y_transpose_Z_expected_val_2)))
    Y_2_edge_expected=np.hstack((Train_expected_edge_cost_2,Val_expected_edge_cost_2))
    X_2_False=np.hstack((X_train_False_2,X_val_False_2))
    Z_idxs_2=np.hstack((Z_train_idxs_2,Z_val_idxs_2))
    
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
    wrong_model_alpha = Ridge(fit_intercept=False,alpha=1000)
    wrong_model_linear = Ridge(fit_intercept=False,alpha=1) 
    
    model.fit(data_X, Y) 
    wrong_model_alpha.fit(data_X, Y) 
    wrong_model_linear.fit(data_X_False, Y) 
    
    predicts = model.predict(data_X) 
    predicts_2 = model.predict(data_X_2) 
    wrong_alpha_model_predicts_2= wrong_model_alpha.predict(data_X_2) 
    wrong_linear_model_predicts_2= wrong_model_linear.predict(data_X_2_False) 
    
    
    parameters = model.coef_ 
    wrong_alpha_model_parameters= wrong_model_alpha.coef_
    wrong_linear_model_parameters= wrong_model_linear.coef_
    
    
    #model predict result
    W_matrix=np.zeros((p,d))
    W_wrong_alpha_matrix=np.zeros((p,d))
    W_wrong_linear_matrix=np.zeros((p_False,d))
    for i in range(p):
        W_matrix[i,:]=parameters[0,i*d:i*d+d]
        W_wrong_alpha_matrix[i,:]=wrong_alpha_model_parameters[0,i*d:i*d+d]
    for i in range(p_False):
        W_wrong_linear_matrix[i,:]=wrong_linear_model_parameters[0,i*d:i*d+d]
    
    model_estimate=np.dot(np.transpose(X),W_matrix)
    model_estimate=np.transpose(model_estimate)
    model_estimate_2=np.dot(np.transpose(X_2),W_matrix)
    model_estimate_2=np.transpose(model_estimate_2)    
    Alpha_Wrong_model_estimate_2=np.dot(np.transpose(X_2),W_wrong_alpha_matrix)
    Alpha_Wrong_model_estimate_2=np.transpose(Alpha_Wrong_model_estimate_2)    
    Linear_Wrong_model_estimate_2=np.dot(np.transpose(X_2_False),W_wrong_linear_matrix)  
    Linear_Wrong_model_estimate_2=np.transpose(Linear_Wrong_model_estimate_2)     
    
    #Sigma matrix,这里先计算理想矩阵
    Sigma_ideal_matrix_1=np.zeros((d,d))
    for i in range(50):
        if i <=24:
         temp_matrix=2*np.outer(Z[:,i],Z[:,i])
         Sigma_ideal_matrix_1=Sigma_ideal_matrix_1+temp_matrix
        else:
         temp_matrix=np.outer(Z[:,i],Z[:,i])
         Sigma_ideal_matrix_1=Sigma_ideal_matrix_1+temp_matrix
    Sigma_ideal_matrix_1=Sigma_ideal_matrix_1/75
    
    Sigma_ideal_matrix_2=np.zeros((d,d))
    for i in range(50):
        if i <=24:
         temp_matrix=np.outer(Z[:,i],Z[:,i])
         Sigma_ideal_matrix_2=Sigma_ideal_matrix_2+temp_matrix
        else:
         temp_matrix=2*np.outer(Z[:,i],Z[:,i])
         Sigma_ideal_matrix_2=Sigma_ideal_matrix_2+temp_matrix
    Sigma_ideal_matrix_2=Sigma_ideal_matrix_2/75
    
    Sigma_matrix_1_ideal_inverse = np.linalg.pinv(Sigma_ideal_matrix_1)    
    Sigma_matrix_2_ideal_inverse = np.linalg.pinv(Sigma_ideal_matrix_2)  

    Ideal_U, Ideal_Sigma,Ideal_V = np.linalg.svd(Sigma_ideal_matrix_1)
    for i in range(len(Ideal_Sigma)):
      if Ideal_Sigma[i]<0.00000000001:
        Ideal_Sigma[i]=0
      if Ideal_Sigma[i]>0.00000000001 and Ideal_Sigma[i]<1:
        Ideal_Sigma[i]=1
    Modeified_ideal_Sigma_matrix_1=np.dot(np.dot(Ideal_U,np.diag(Ideal_Sigma)),Ideal_V)
    Modeified_ideal_Sigma_matrix_1_inverse=np.linalg.pinv(Modeified_ideal_Sigma_matrix_1)

    Ideal_U, Ideal_Sigma,Ideal_V = np.linalg.svd(Sigma_ideal_matrix_2)
    for i in range(len(Ideal_Sigma)):
      if Ideal_Sigma[i]<0.00000000001:
        Ideal_Sigma[i]=0
      if Ideal_Sigma[i]>0.00000000001 and Ideal_Sigma[i]<1:
        Ideal_Sigma[i]=1
    Modeified_ideal_Sigma_matrix_2=np.dot(np.dot(Ideal_U,np.diag(Ideal_Sigma)),Ideal_V)
    Modeified_ideal_Sigma_matrix_2_inverse=np.linalg.pinv(Modeified_ideal_Sigma_matrix_2)
    
    #再计算采样矩阵
    Sigma_sample_matrix_1=np.zeros((d,d))
    Sigma_sample_matrix_2=np.zeros((d,d))
    num_1=0
    num_2=0
    for i in range(m):
       if X[1,i]>0:
           temp_matrix=np.outer(Z[:,i],Z[:,i])
           Sigma_sample_matrix_1=Sigma_sample_matrix_1+temp_matrix
           num_1=num_1+1
       else:
           temp_matrix=np.outer(Z[:,i],Z[:,i])
           Sigma_sample_matrix_2=Sigma_sample_matrix_2+temp_matrix
           num_2=num_2+1
    
    Sigma_sample_matrix_1=Sigma_sample_matrix_1/num_1
    Sigma_matrix_1_sample_inverse = np.linalg.pinv(Sigma_sample_matrix_1)    
    Sigma_sample_matrix_2=Sigma_sample_matrix_2/num_2
    Sigma_matrix_2_sample_inverse = np.linalg.pinv(Sigma_sample_matrix_2)
    
    sample_U, sample_Sigma,sample_V = np.linalg.svd(Sigma_sample_matrix_1)
    for i in range(len(sample_Sigma)):
      if sample_Sigma[i]<0.00000000001:
        sample_Sigma[i]=0
      if sample_Sigma[i]>0.00000000001 and sample_Sigma[i]<1:
        sample_Sigma[i]=1
    Modeified_sample_Sigma_matrix_1=np.dot(np.dot(sample_U,np.diag(sample_Sigma)),sample_V)
    Modeified_sample_Sigma_matrix_1_inverse=np.linalg.pinv(Modeified_sample_Sigma_matrix_1)

    sample_U, sample_Sigma,sample_V = np.linalg.svd(Sigma_sample_matrix_2)
    for i in range(len(sample_Sigma)):
      if sample_Sigma[i]<0.00000000001:
        sample_Sigma[i]=0
      if sample_Sigma[i]>0.00000000001 and sample_Sigma[i]<1:
        sample_Sigma[i]=1
    Modeified_sample_Sigma_matrix_2=np.dot(np.dot(sample_U,np.diag(sample_Sigma)),sample_V)
    Modeified_sample_Sigma_matrix_2_inverse=np.linalg.pinv(Modeified_sample_Sigma_matrix_2)
    
    Vanilla_Double_robust_predict_2_sample=np.zeros((d,m))
    Clip_Double_robust_predict_2_sample=np.zeros((d,m))
    Vanilla_Perfect_Double_robust_predict_ideal_Sigma=np.zeros((d,m))
    Clip_Perfect_Double_robust_predict_ideal_Sigma=np.zeros((d,m))
    Lambda_Double_robust_predict_2_sample=np.zeros((d,m))
    Lambda_Perfect_Double_robust_predict_ideal_Sigma=np.zeros((d,m))
    
    
    Tree_Vanilla_Double_robust_predict_2=np.zeros((d,m))
    Tree_Clip_Double_robust_predict_2=np.zeros((d,m))
    Tree_Lambda_Double_robust_predict_2_tree=np.zeros((d,m))
    
    Linear_IPW_ideal=np.zeros((d,m))
    Linear_IPW_sample=np.zeros((d,m))
    Linear_IPW_tree=np.zeros((d,m))
    Linear_IPW_ideal_Clip=np.zeros((d,m))
    Linear_IPW_sample_Clip=np.zeros((d,m))
    Linear_IPW_tree_Clip=np.zeros((d,m))
    Linear_IPW_ideal_Lambda=np.zeros((d,m))
    Linear_IPW_sample_Lambda=np.zeros((d,m))
    Linear_IPW_tree_Lambda=np.zeros((d,m))
    
    Sigma_matrix_1_adjustment_ideal_lambda_1=Sigma_ideal_matrix_1+np.eye(d,d)
    Sigma_matrix_2_adjustment_ideal_lambda_1=Sigma_ideal_matrix_2+np.eye(d,d)
    Sigma_matrix_1_adjustment_ideal_lambda_inverse=np.linalg.inv(Sigma_matrix_1_adjustment_ideal_lambda_1)
    Sigma_matrix_2_adjustment_ideal_lambda_inverse=np.linalg.inv(Sigma_matrix_2_adjustment_ideal_lambda_1)

    Sigma_matrix_1_adjustment_sample_lambda_1=Sigma_sample_matrix_1+np.eye(d,d)
    Sigma_matrix_2_adjustment_sample_lambda_1=Sigma_sample_matrix_2+np.eye(d,d)
    Sigma_matrix_1_adjustment_sample_lambda_inverse=np.linalg.inv(Sigma_matrix_1_adjustment_sample_lambda_1)
    Sigma_matrix_2_adjustment_sample_lambda_inverse=np.linalg.inv(Sigma_matrix_2_adjustment_sample_lambda_1)
    
    total_estimate_Sigma_DR=np.zeros((d,m))
    clf = tree.DecisionTreeClassifier(max_depth=3,random_state=18)
    
    (p,m)=X_False.shape
    #减去常数项
    p=p-1
    X_clf=np.zeros((m,p))
    Y_clf=np.zeros((m,1))
    X_clf_2=np.zeros((m,p))
    for i in range(m):
        X_clf[i,:]=X_False[1:,i]
        Y_clf[i,:]=Z_idxs[0,i]
        X_clf_2[i,:]=X_2_False[1:,i]

    IPW_total_estimator = clf.fit(X_clf,Y_clf)
    a_1=IPW_total_estimator.predict_proba(X_clf_2)
    a_1=np.transpose(a_1)
    a_2=IPW_total_estimator.predict(X_clf_2)
    
    #在这里计算每一个数据点的双鲁棒项中的IPW那一项
    
    unique, counts = np.unique(Z_idxs, return_counts=True)
    d_list=[]
    for k in range(m):
        a=np.array(Y_2.iloc[k,:])
        b=predicts_2[k,:]
        c=a-b
        Tree_estimate_Sigma=np.zeros((d,d))
        for j in range(len(unique)):
            Tree_estimate_Sigma=Tree_estimate_Sigma+np.outer(feasible_vector[int(unique[j])],feasible_vector[int(unique[j])])*a_1[j,k]
        Tree_estimate_Sigma_inverse=np.linalg.pinv(Tree_estimate_Sigma)   
        
        sample_U, Tree_Sigma,sample_V = np.linalg.svd(Tree_estimate_Sigma)
        for i in range(len(sample_Sigma)):
         if Tree_Sigma[i]<0.00000000001:
           Tree_Sigma[i]=0
         if Tree_Sigma[i]>0.00000000001 and Tree_Sigma[i]<1:
           Tree_Sigma[i]=1
        Modeified_Tree_Sigma=np.dot(np.dot(sample_U,np.diag(Tree_Sigma)),sample_V)
        Modeified_Tree_Sigma_inverse=np.linalg.pinv(Modeified_Tree_Sigma)
        
        Tree_Sigma_matrix_adjustment=Tree_estimate_Sigma+np.eye(d,d)
        Tree_Sigma_matrix_adjustment_inverse=np.linalg.pinv(Tree_Sigma_matrix_adjustment)
        
        Tree_Vanilla_Double_robust_predict_2[:,k]=model_estimate_2[:,k]+np.dot(Tree_estimate_Sigma_inverse,Z_2[:,k])*c
        Tree_Clip_Double_robust_predict_2[:,k]=model_estimate_2[:,k]+np.dot(Modeified_Tree_Sigma_inverse,Z_2[:,k])*c
        Tree_Lambda_Double_robust_predict_2_tree[:,k]=model_estimate_2[:,k]+np.dot(Tree_Sigma_matrix_adjustment_inverse,Z_2[:,k])*c
        Linear_IPW_tree[:,k]=np.dot(Tree_estimate_Sigma_inverse,Z_2[:,k])*np.array(Y_2.iloc[k,:])
        Linear_IPW_tree_Clip[:,k]=np.dot(Modeified_Tree_Sigma_inverse,Z_2[:,k])*np.array(Y_2.iloc[k,:])
        Linear_IPW_tree_Lambda[:,k]=np.dot(Tree_Sigma_matrix_adjustment_inverse,Z_2[:,k])*np.array(Y_2.iloc[k,:])
        
        
    for i in range(m): 
      
      if X[1,i]>0:
        a=np.array(Y_2.iloc[i,:])
        b=predicts_2[i,:]
        c=a-b
        #d_2.append(np.dot(Sigma_matrix_inverse,Z_2[:,i])) 
        #Vanilla_Double_robust_predict_2_ideal[:,i]= model_estimate_2[:,i]+np.dot(Sigma_matrix_inverse,Z_2[:,i])*c
        Vanilla_Double_robust_predict_2_sample[:,i]= model_estimate_2[:,i]+np.dot(Sigma_matrix_1_sample_inverse,Z_2[:,i])*c
        Clip_Double_robust_predict_2_sample[:,i]= model_estimate_2[:,i]+np.dot(Modeified_sample_Sigma_matrix_1_inverse,Z_2[:,i])*c
        Lambda_Double_robust_predict_2_sample[:,i]= model_estimate_2[:,i]+np.dot(Sigma_matrix_1_adjustment_sample_lambda_inverse,Z_2[:,i])*c
        Linear_IPW_sample[:,i]=np.dot(Sigma_matrix_1_sample_inverse,Z_2[:,i])*np.array(Y_2.iloc[i,:])
        Linear_IPW_sample_Clip[:,i]=np.dot(Modeified_sample_Sigma_matrix_1_inverse,Z_2[:,i])*np.array(Y_2.iloc[i,:])
        Linear_IPW_sample_Lambda[:,i]=np.dot(Sigma_matrix_1_adjustment_sample_lambda_inverse,Z_2[:,i])*np.array(Y_2.iloc[i,:])
        
      else:
        a=np.array(Y_2.iloc[i,:])
        b=predicts_2[i,:]
        c=a-b
        #d_2.append(np.dot(Sigma_matrix_inverse,Z_2[:,i])) 
        #Vanilla_Double_robust_predict_2_ideal[:,i]= model_estimate_2[:,i]+np.dot(Sigma_matrix_inverse,Z_2[:,i])*c
        Vanilla_Double_robust_predict_2_sample[:,i]= model_estimate_2[:,i]+np.dot(Sigma_matrix_2_sample_inverse,Z_2[:,i])*c
        Clip_Double_robust_predict_2_sample[:,i]= model_estimate_2[:,i]+np.dot(Modeified_sample_Sigma_matrix_2_inverse,Z_2[:,i])*c
        Lambda_Double_robust_predict_2_sample[:,i]= model_estimate_2[:,i]+np.dot(Sigma_matrix_2_adjustment_sample_lambda_inverse,Z_2[:,i])*c
        Linear_IPW_sample[:,i]=np.dot(Sigma_matrix_2_sample_inverse,Z_2[:,i])*np.array(Y_2.iloc[i,:])
        Linear_IPW_sample_Clip[:,i]=np.dot(Modeified_sample_Sigma_matrix_2_inverse,Z_2[:,i])*np.array(Y_2.iloc[i,:])
        Linear_IPW_sample_Lambda[:,i]=np.dot(Sigma_matrix_2_adjustment_sample_lambda_inverse,Z_2[:,i])*np.array(Y_2.iloc[i,:])
    
    for i in range(m):     
        if X[1,i]>0:
         c=np.array(Y_2.iloc[i,:])-np.array(Y_2_expected[i,:])
         Vanilla_Perfect_Double_robust_predict_ideal_Sigma[:,i]= Y_2_edge_expected[:,i]+np.dot(Sigma_matrix_1_ideal_inverse,Z_2[:,i])*c
         Clip_Perfect_Double_robust_predict_ideal_Sigma[:,i]= Y_2_edge_expected[:,i]+np.dot(Modeified_ideal_Sigma_matrix_1_inverse,Z_2[:,i])*c
         Lambda_Perfect_Double_robust_predict_ideal_Sigma[:,i]= Y_2_edge_expected[:,i]+np.dot(Sigma_matrix_1_adjustment_ideal_lambda_inverse,Z_2[:,i])*c
         #Vanilla_Double_robust_predict_with_sample_Sigma[:,i]= Y_2_edge_expected[:,i]+np.dot(Sigma_matrix_sample_inverse,Z_2[:,i])*c
         Linear_IPW_ideal[:,i]=np.dot(Sigma_matrix_1_ideal_inverse,Z_2[:,i])*np.array(Y_2.iloc[i,:])
         Linear_IPW_ideal_Clip[:,i]=np.dot(Modeified_ideal_Sigma_matrix_1_inverse,Z_2[:,i])*np.array(Y_2.iloc[i,:])
         Linear_IPW_ideal_Lambda[:,i]=np.dot(Sigma_matrix_1_adjustment_ideal_lambda_inverse,Z_2[:,i])*np.array(Y_2.iloc[i,:])
         
        else:
         c=np.array(Y_2.iloc[i,:])-np.array(Y_2_expected[i,:])
         Vanilla_Perfect_Double_robust_predict_ideal_Sigma[:,i]= Y_2_edge_expected[:,i]+np.dot(Sigma_matrix_2_ideal_inverse,Z_2[:,i])*c
         Clip_Perfect_Double_robust_predict_ideal_Sigma[:,i]= Y_2_edge_expected[:,i]+np.dot(Modeified_ideal_Sigma_matrix_2_inverse,Z_2[:,i])*c
         Lambda_Perfect_Double_robust_predict_ideal_Sigma[:,i]= Y_2_edge_expected[:,i]+np.dot(Sigma_matrix_2_adjustment_ideal_lambda_inverse,Z_2[:,i])*c
         Linear_IPW_ideal[:,i]=np.dot(Sigma_matrix_2_ideal_inverse,Z_2[:,i])*np.array(Y_2.iloc[i,:])
         Linear_IPW_ideal_Clip[:,i]=np.dot(Modeified_ideal_Sigma_matrix_2_inverse,Z_2[:,i])*np.array(Y_2.iloc[i,:])
         Linear_IPW_ideal_Lambda[:,i]=np.dot(Sigma_matrix_2_adjustment_ideal_lambda_inverse,Z_2[:,i])*np.array(Y_2.iloc[i,:])
            
    return(model_estimate_2,Vanilla_Double_robust_predict_2_sample,Clip_Double_robust_predict_2_sample,Vanilla_Perfect_Double_robust_predict_ideal_Sigma,
           Clip_Perfect_Double_robust_predict_ideal_Sigma,Lambda_Double_robust_predict_2_sample,Lambda_Perfect_Double_robust_predict_ideal_Sigma,
           Linear_IPW_ideal,Linear_IPW_sample,Tree_Vanilla_Double_robust_predict_2,Tree_Clip_Double_robust_predict_2,Tree_Lambda_Double_robust_predict_2_tree,Linear_IPW_tree,
           Linear_IPW_ideal_Clip,Linear_IPW_sample_Clip,Linear_IPW_ideal_Lambda,Linear_IPW_sample_Lambda,Linear_IPW_tree_Clip,Linear_IPW_tree_Lambda)

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

#让我们把所有测试集中的最优解从feasible_set里面剔除出来
best_list=[63,26,17,1,53,14,38,37,5,50,44,42,68,64,62,33,43,25,12,18]
Fake_feasible_vector=[]
for j in range(70):
    if j not in best_list:
       Fake_feasible_vector.append(feasible_vector[j])
       

new_feasible_vector_1=[]
new_feasible_vector_2=[]
for i in range(50):
        new_feasible_vector_1.append(Fake_feasible_vector[i])
        new_feasible_vector_2.append(Fake_feasible_vector[i])
        
for i in range(25):
        new_feasible_vector_1.append(Fake_feasible_vector[i])
        new_feasible_vector_2.append(Fake_feasible_vector[25+i])

data_train_1 = [Fake_generate_data_interactive(noise_low,noise_high,B_true, n_train_max, p,feasible_vector, Fake_feasible_vector,new_feasible_vector_1,new_feasible_vector_2,polykernel_degree = polykernel_degree, noise_half_width = noise_half_width, constant = constant) for run in range(runs)]
data_holdout_1 = [Fake_generate_data_interactive(noise_low,noise_high,B_true, n_holdout_max, p,feasible_vector, Fake_feasible_vector,new_feasible_vector_1,new_feasible_vector_2,polykernel_degree = polykernel_degree, noise_half_width = noise_half_width, constant = constant) for run in range(runs)]
data_train_2 = [Fake_generate_data_interactive(noise_low,noise_high,B_true, n_train_max, p,feasible_vector, Fake_feasible_vector,new_feasible_vector_1,new_feasible_vector_2,polykernel_degree = polykernel_degree, noise_half_width = noise_half_width, constant = constant) for run in range(runs)]
data_holdout_2 = [Fake_generate_data_interactive(noise_low,noise_high,B_true, n_holdout_max, p,feasible_vector, Fake_feasible_vector,new_feasible_vector_1,new_feasible_vector_2,polykernel_degree = polykernel_degree, noise_half_width = noise_half_width, constant = constant) for run in range(runs)]

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
    
regret_all_Linear_IPW_Sample = []
validation_all_Linear_IPW_Sample = []
time_all_Linear_IPW_Sample= []
loss_all_Linear_IPW_Sample = []


for i in range(len(n_train_seq)):
    

    n_train_true = n_train_seq[i]
    n_train = int(n_train_true/2)
    n_holdout_true = n_holdout_seq[i] 
    n_holdout = int(n_holdout_true/2)
    
    #Double_robust
    res_temp_0 = Parallel(n_jobs = n_jobs, verbose = 3)(delayed(Double_robust)(data_train_1[run][0][:,:n_train],data_train_1[run][1][:,:n_train],data_train_1[run][4][:,:n_train],data_train_1[run][5][:,:n_train],data_train_1[run][6][:,:n_train],data_train_1[run][8][:,:n_train],
                                                                         data_holdout_1[run][0][:,:n_holdout],data_holdout_1[run][1][:,:n_holdout],data_holdout_1[run][4][:,:n_holdout],data_holdout_1[run][5][:,:n_holdout],data_holdout_1[run][6][:,:n_holdout],data_holdout_1[run][8][:,:n_holdout],
                                                                         data_train_2[run][0][:,:n_train],data_train_2[run][1][:,:n_train],data_train_2[run][4][:,:n_train],data_train_2[run][5][:,:n_train],data_train_2[run][6][:,:n_train],data_train_2[run][8][:,:n_train],
                                                                         data_holdout_2[run][0][:,:n_holdout],data_holdout_2[run][1][:,:n_holdout],data_holdout_2[run][4][:,:n_holdout],data_holdout_2[run][5][:,:n_holdout],data_holdout_2[run][6][:,:n_holdout],data_holdout_2[run][8][:,:n_holdout],
                                                                         data_train_2[run][3][:,:n_train],data_holdout_2[run][3][:,:n_holdout],feasible_vector,Sigma_matrix) for run in range(runs))
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
    res_temp_0 = Parallel(n_jobs = n_jobs, verbose = 3)(delayed(Double_robust)(data_train_2[run][0][:,:n_train],data_train_2[run][1][:,:n_train],data_train_2[run][4][:,:n_train],data_train_2[run][5][:,:n_train],data_train_2[run][6][:,:n_train],data_train_2[run][8][:,:n_train],
                                                                         data_holdout_2[run][0][:,:n_holdout],data_holdout_2[run][1][:,:n_holdout],data_holdout_2[run][4][:,:n_holdout],data_holdout_2[run][5][:,:n_holdout],data_holdout_2[run][6][:,:n_holdout],data_holdout_2[run][8][:,:n_holdout],
                                                                         data_train_1[run][0][:,:n_train],data_train_1[run][1][:,:n_train],data_train_1[run][4][:,:n_train],data_train_1[run][5][:,:n_train],data_train_1[run][6][:,:n_train],data_train_1[run][8][:,:n_train],
                                                                         data_holdout_1[run][0][:,:n_holdout],data_holdout_1[run][1][:,:n_holdout],data_holdout_1[run][4][:,:n_holdout],data_holdout_1[run][5][:,:n_holdout],data_holdout_1[run][6][:,:n_holdout],data_holdout_1[run][8][:,:n_holdout],
                                                                         data_train_1[run][3][:,:n_train],data_holdout_1[run][3][:,:n_holdout],feasible_vector,Sigma_matrix) for run in range(runs))
  
    
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
                            data_train[run][0][:, :n_train], data_train[run][1][:, :n_train], data_train[run][16][:, :n_train], data_train[run][3][:, :n_train],
                            data_holdout[run][0][:, :n_holdout], data_holdout[run][1][:, :n_holdout], data_holdout[run][16][:, :n_holdout], data_holdout[run][3][:, :n_holdout],
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
        print("Linear IPW Tree")
        print("Linear IPW Tree", file = f)
        print("total time: ", time.time() - time1, file = f)
        print("seperate time", file = f)
        print(time_temp.mean(), file = f)
        print("-------", file = f)
        print("mean regret: ", file = f)
        print(regret_temp.mean(), file = f)      
        print("-------", file = f)
        print("        ", file = f)
        print("        ", file = f)
        
    regret_all_Linear_IPW_Tree.append(regret_temp)
    validation_all_Linear_IPW_Tree.append(validation_temp)
    time_all_Linear_IPW_Tree.append(time_temp)
    loss_all_Linear_IPW_Tree.append(loss_temp)
    
    pd.concat(regret_all_Linear_IPW_Tree).to_csv("regret_no_kernel_ISW_Tree.csv", index = False)
    pd.concat(validation_all_Linear_IPW_Tree).to_csv("validation_no_kernel_ISW_Tree.csv", index = False)
    pd.concat(time_all_Linear_IPW_Tree).to_csv("time_no_kernel_ISW_Tree.csv", index = False)
    pickle.dump(loss_all_Linear_IPW_Tree, open("loss_no_kernel_ISW_Tree.pkl", "wb"))
