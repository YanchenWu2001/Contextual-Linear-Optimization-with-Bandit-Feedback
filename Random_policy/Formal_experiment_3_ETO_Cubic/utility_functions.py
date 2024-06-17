import numpy as np
import csv
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel
from sklearn.kernel_ridge import KernelRidge
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
import pickle 
import time 
import mkl
from sklearn.neural_network import MLPRegressor
from numpy.linalg import norm
mkl.set_num_threads(1)


def generate_data_interactive(B_true, n, p, polykernel_degree = 3, 
                              noise_half_width = 0.25, constant = 3):
    (d, _) = B_true.shape
    X = np.random.normal(size=(p, n))
    poly = PolynomialFeatures(degree = polykernel_degree, interaction_only = True, include_bias = False)
    X_new = np.transpose(poly.fit_transform(np.transpose(X)))
    
    c_expected = np.matmul(B_true, X_new) + constant 
    epsilon = 1 - noise_half_width + 2 * noise_half_width * np.random.rand(d, n)
    c_observed = c_expected * epsilon

    X_false = np.concatenate((np.reshape(np.ones(n), (1, -1)), X), axis = 0)
    X_true = np.concatenate((np.reshape(np.ones(n), (1, -1)), X_new), axis = 0)
    
    return (X_false, X_true, c_observed, c_expected)

def read_A_and_b(file_path = None):
    if file_path is None:
        file_path = 'A_and_b.csv'

    A_and_b = pd.read_csv(file_path, header = None)
    A_and_b = A_and_b.to_numpy()
    A_mat = A_and_b[:, :-1]
    b_vec = A_and_b[:, -1]
    
    return (A_mat, b_vec)

def Kernel_transformation(X_train, X_val, gamma = 1):
    'X_train转置以后的维度是1000*40，那么这个K_mat的维度是1000*1000'
    K_mat = rbf_kernel(np.transpose(X_train), gamma = gamma)
    '不太明白K矩阵本身应该是对称的为什么还要相加除2？'
    K_mat = (K_mat + np.transpose(K_mat))/2
    '特征值复数取实部'
    eig_val = np.real(np.linalg.eig(K_mat)[0])
    eig_vector = np.real(np.linalg.eig(K_mat)[1])

    eig_val_positive = np.copy(eig_val)
    eig_val_positive[eig_val_positive <= 1e-4] = 0
    eig_val_positive_inv = np.copy(eig_val_positive)
    '对大于零的特征值进行取倒数'
    eig_val_positive_inv[eig_val_positive_inv > 0] = 1/eig_val_positive_inv[eig_val_positive_inv > 0]
    '把矩阵还原回来'
    K_mat_sqrt = np.matmul(np.matmul(eig_vector, np.diag(np.sqrt(eig_val_positive))), np.transpose(eig_vector))
    K_mat_sqrt_inv = np.matmul(np.matmul(eig_vector, np.diag(np.sqrt(eig_val_positive_inv))), np.transpose(eig_vector))
    '这里不太明白训练集乘验证集的含义，维度是1000*1000'
    K_mat_val = rbf_kernel(np.transpose(X_train), np.transpose(X_val), gamma = gamma)
    K_design_val = np.matmul(K_mat_sqrt_inv, K_mat_val)
    '但是根据后面的spo_kernel_predict函数，我对于这个k_design_val的维度有问题（矩阵还是向量？）'
    return (K_mat_sqrt, K_mat_sqrt_inv, K_design_val)


def generate_sp_oracle(A_mat, b_vec, verbose = False):

    (m, d) = A_mat.shape
    '''m为25表示25个节点，d为40表示40条边'''
    model = gp.Model()
    model.setParam('OutputFlag', verbose)
    model.setParam("Threads", 1)

    w = model.addVars(d, lb = 0, ub = 1, name = 'w')
    '添加决策变量w,歧视就是我们要选择的路径，还挺重要的'
    model.update()
    for i in range(m):
        model.addConstr(gp.quicksum(A_mat[i, j] * w[j] for j in range(len(w))) == b_vec[i])
    model.update()
    '这里表示的流量守恒一共有40个约束方程'
    def local_oracle(c):
        if len(c) != len(w):
            raise Exception("Sorry, c and w dimension mismatched")
        '这里是计算路径总代价作为模型的目标，决策变量维度应该也是40个'
        obj = gp.quicksum(c[i] * w[i] for i in range(len(w)))
        '设定最小化目标'
        model.setObjective(obj, GRB.MINIMIZE)
        model.update()
        model.optimize()
        'Gurobi的变量.X表示当前变量取值,这里是以列表的形式输出'
        w_ast = [w[i].X for i in range(len(w))]
        'objVal表示当前目标函数取值'
        z_ast = model.objVal

        return (z_ast, w_ast)
    
    return local_oracle

def oracle_dataset(c, oracle):
    '''d为40，n为1000'''
    (d, n) = c.shape
    '生成一个1000维的0向量'
    z_star_data = np.zeros(n)
    '生成一个40*1000的0矩阵'
    w_star_data = np.zeros((d, n))
    for i in range(n):
        '''这里把c的第i列输入进去，也就是一个1*40维度的向量，它包含了40条路径的全部信息'''
        (z_i, w_i) = oracle(c[:,i])
        '''这里应该是直接开了上帝视角，使用线性规划计算出了它的最优决策'''
        z_star_data[i] = z_i
        '把这1000条数据的代价计算出来，以及选择的最优路径向量'
        w_star_data[:,i] = w_i
    return (z_star_data, w_star_data)

def spo_linear_predict(B_est, X_val):
    c_hat = np.matmul(B_est, X_val)
    
    return c_hat 

def spo_kernel_predict(v_est, K_design_val):
    c_hat = np.matmul(v_est, K_design_val)
    return c_hat 

def ridge_kernel(X, c, gamma = 1, cur_lambda = 1):
    kr = KernelRidge(alpha = cur_lambda/2, kernel = "rbf", gamma = gamma)
    kr.fit(np.transpose(X), np.transpose(c))
    return kr 

def ridge_linear(X, c, cur_lambda = 1):

    (p, n) = X.shape
    (d, n2) = c.shape
    if n != n2:
        raise Exception("Sorry, c and X dimension mismatched")

    model = Ridge(alpha = cur_lambda/2, fit_intercept = False)
    model.fit(np.transpose(X), np.transpose(c))
        
    return model

def ls_linear(X, c):
    (p, n) = X.shape
    (d, n2) = c.shape
    if n != n2:
        raise Exception("Sorry, c and X dimension mismatched")

    model = LinearRegression(fit_intercept = False)
    model.fit(np.transpose(X), np.transpose(c))

    return model


def SPO_reformulation_kernel(K_mat_sqrt, K_mat_sqrt_inv, X, c, 
                             z_star_train, w_star_train, 
                             A_mat, b_vec, 
                             cur_lambda = 1, verbose = False):

    try:
        v_est = SPO_reformulation_linear(K_mat_sqrt, c, z_star_train, w_star_train, A_mat, b_vec, 
                                   cur_lambda = cur_lambda, verbose = verbose)
    except Exception as e:
        print("SPO_kernel error:", e)
        print("Optimization is unsuccessful with status code", model.status)
    return v_est

def SPO_reformulation_linear(X, c, z_star_train, w_star_train, A_mat, b_vec, 
                                   cur_lambda = 1, verbose = False):
    '把A的维度从(25，40)转置到(40,25)'
    A_mat_trans = np.transpose(A_mat)
    (n_nodes, n_edges) = A_mat.shape
    '在错误模型中p=6，正确模型中p=32'
    (p, n) = X.shape
    '正确错误d都是40'
    (d, n2) = c.shape

    if n != n2:
        raise Exception("Sorry, c and X dimension mismatched")

    try:
        model = gp.Model()
        '关闭显示模型的模型的过程信息'
        model.setParam('OutputFlag', verbose)
        model.setParam("Threads", 1)
        
        'p_var维度为40,1000，它是拉格朗日算子'
        p_var = model.addVars(n_nodes, n2, name = 'p')
        'B_var维度是40,6(对于错误模型来说)'
        B_var = model.addVars(d, p, name = "B")
        model.update()

        for i in range(n):
            for j in range(d):
                constr_lhs = -gp.quicksum(A_mat_trans[j, k] * p_var[k, i] for k in range(n_nodes))
                constr_rhs = c[j, i] - 2 * gp.quicksum(B_var[j, k] * X[k, i] for k in range(p))
                model.addConstr(constr_lhs >= constr_rhs)
        model.update()

        obj_noreg = 0 
        for i in range(n):
            term1 = - gp.quicksum(b_vec[k] * p_var[k, i] for k in range(n_nodes))
            term2 = 2 * gp.quicksum(w_star_train[j, i] * B_var[j, k] * X[k, i] for j in range(d) for k in range(p))
            obj_noreg = obj_noreg + term1 + term2 - z_star_train[i]

        obj = obj_noreg + n*(cur_lambda/2) * gp.quicksum(B_var[i, j]*B_var[i, j] for i in range(d) for j in range(p))
        model.setObjective(obj, GRB.MINIMIZE)

        model.update()

        model.optimize()
        '把B_est优化的结果以np.array的形式输出出来'
        B_est = np.array([[B_var[k, j].X for j in range(p)] for k in range(d)])
    except Exception as e:
        print("SPO_linear error:", e)
        print("Optimization is unsuccessful with status code", model.status)
        # status code table: https://www.gurobi.com/documentation/9.1/refman/optimization_status_codes.html
    return B_est

def spo_loss(c_hat, c_val, z_star_val, w_star_val, sp_oracle):
    '在该问题中有1000条数据'
    n_holdout = len(z_star_val)
    spo_sum = 0
    
    for i in range(n_holdout):
        '输入一个c_hat输出一些z和w的最优值2然后取和输入函数的最优值进行对比'
        '这里的c_hat是基于B计算出来的很有可能不准确'
        (z_oracle, w_oracle) = sp_oracle(c_hat[:, i])
        spo_loss_cur = np.dot(c_val[:,i], w_oracle) - z_star_val[i]
        spo_sum = spo_sum + spo_loss_cur
        
    spo_loss_avg = spo_sum/n_holdout
    '这里输出的是1000条数据的平均损失'
    return spo_loss_avg

def spo_sgd(features, c, sp_oracle, z_star_train, w_star_train, B_init, cur_lambda, 
                 numiter =  1000, batchsize = 10, long_factor = 0.1):
    
    (p, n) = features.shape
    '特征的维度32*1000'
    (d, n2) = c.shape
    '这个表示训练集可以看到的全部信息,也就是40条边的信息，40*1000'

    def subgrad_stochastic(B_new, cur_lambda):
        G_new = np.zeros(B_new.shape)
        '这里是把这个函数命名为subgrad,实际上在算法里面输入的是每次迭代的B_iter, cur_lambda'
        for j in range(batchsize):
            i = np.random.randint(n)
            '随机化一个i,但如果是完全梯度下降的话应该是遍历而不是batchsize加随机数的操作对吗？'
            spoplus_cost_vec = 2*np.matmul(B_new, features[:, i]) - c[:,i]
            '这里计算的是一个spoplus_cost_vec,但是实际上应该还是那个40维的列向量'
            (z_oracle, w_oracle) = sp_oracle(spoplus_cost_vec)
            '这里的sp_oracle是在这个spo_sgd函数传参数之前就已经用generate_sp_oracle函数定义好了输入的是A_mat和b_vec'
            w_star_diff = w_star_train[:, i] - np.array(w_oracle)
            'w_oracle表示刚刚接出来的，而w_star_train表示利用train数据集中的fullfeedback解出来的最优解'
            '在我们打算实现的方法里面好像没有去利用已知的数据信息去求解一个训练集最优解'
            '这里我可以理解当最优决策不再变化时，梯度就是0，非常make sense，那篇新论文里面的思路其实也是这个'
            '但是我的问题是在那篇新论文里面好像就是从给定一个初始值开始迭代，并没用fullfeedback信息去专门解出一个最优解出来，唯一利用到fullfeedback信息的还是要乘上一个小h'
            '这给我的感觉就是可控性是不是太差了那个算法，很可能受到h影响比较大，初始值影响比较大，相比之下这个算法我就感觉可行性要强很多'
            G_new = G_new + 2 * w_star_diff[:, np.newaxis] * features[:,i][np.newaxis, :]
            '这里原论文的算法是不是漏掉了2，这里注意一下维度，G是作为随机梯度信息的，但是它是一个40*32的矩阵，在我们的计划中是40*(4*40)'
        G_new = (1/batchsize)*G_new + cur_lambda*B_new
        '严格来说这个应该属于Batch SGD，那我们应该怎么设计算法呢'
        return G_new
    subgrad = subgrad_stochastic
    def step_size_long_dynamic(itn, G_new):
        return long_factor/np.sqrt(itn + 1)
    step_size = step_size_long_dynamic
        
    B_iter = B_init
    B_avg_iter = B_init
    step_size_sum = 0
 
    for itn in range(numiter):
        G_iter = subgrad(B_iter, cur_lambda)
        step_iter = step_size(itn, G_iter)
        '这个step_iter就是γ_s'
        step_size_sum = step_size_sum + step_iter
        step_avg = step_iter/step_size_sum
        B_avg_iter_temp = (1 - step_avg)*B_avg_iter + step_avg*B_iter
        '这里先算平均,再更新B_{t+1}'
        B_iter = B_iter - step_iter*G_iter
      
        B_avg_iter = B_avg_iter_temp
        
    return B_avg_iter

def validation_set_alg_kernel_SPO(A_mat, b_vec,
                       X_train, c_train, X_val, c_val, 
                       z_star_train, w_star_train, 
                       z_star_val, w_star_val,
                       sp_oracle = None, 
                       alg_type = "SGD", 
                       numiter = 1000, batchsize = 10, long_factor = 0.1,
                       gammas = None, lambdas = None, lambda_max = 100, lambda_min_ratio = 1e-6, num_lambda = 10,
                       verbose = False):
    
    if sp_oracle is None:
        sp_oracle = generate_sp_oracle(A_mat, b_vec, verbose)
    if lambdas is None:
        lambda_min = lambda_max*lambda_min_ratio
        lambdas = np.exp(np.linspace(np.log(lambda_min), np.log(lambda_max), num = num_lambda))
    if gammas is None:
        gammas = np.array([1/2])

    lambdas = lambdas[lambdas > 1e-6]  # remove lambda close to 0 
    lambdas = lambdas[lambdas <= 1]  # remove lambda larger than 1
    num_gamma = len(gammas)
    num_lambda = len(lambdas)
    validation_loss_list = np.zeros((num_gamma, num_lambda))
    
    for i in range(num_gamma):
        gamma = gammas[i]
        (K_mat_sqrt, K_mat_sqrt_inv, K_design_val) = Kernel_transformation(X_train, X_val, gamma = gamma)
        
        for j in range(num_lambda):
            cur_lambda = lambdas[j]
            '这里相当于转变了一下X数据，用RKHS的基底作为线性模型的回归'
            if alg_type == "reformulation":
                v_est = SPO_reformulation_kernel(K_mat_sqrt, K_mat_sqrt_inv, X_train, c_train, 
                                 z_star_train, w_star_train, 
                                 A_mat, b_vec, cur_lambda = cur_lambda)
                c_hat = spo_kernel_predict(v_est, K_design_val)
                validation_loss_list[i, j] = spo_loss(c_hat, c_val, z_star_val, w_star_val, sp_oracle)
            if alg_type == "SGD":
                '这个v_init的含义为B矩阵，维度为d*p，也就是40*1000'
                v_init = np.zeros((c_train.shape[0], K_mat_sqrt.shape[0]))
                v_est = spo_sgd(K_mat_sqrt, c_train, 
                     sp_oracle, z_star_train, w_star_train, v_init, cur_lambda, 
                     numiter =  numiter, batchsize = batchsize, long_factor = long_factor)
                '根据这一行代码推测出K_design_val'
                c_hat = spo_kernel_predict(v_est, K_design_val)
                validation_loss_list[i, j] = spo_loss(c_hat, c_val, z_star_val, w_star_val, sp_oracle)
    
    ind_list = list(np.ndindex(validation_loss_list.shape))
    best_ind = ind_list[np.argmin(validation_loss_list)]
    best_gamma = gammas[best_ind[0]]
    best_lambda = lambdas[best_ind[1]]
    
    return (best_gamma, best_lambda, validation_loss_list)

def validation_set_alg_kernel_ridge(A_mat, b_vec,
                       X_train, c_train, X_val, c_val, 
                       z_star_train, w_star_train, 
                       z_star_val, w_star_val,
                       sp_oracle = None,
                       gammas = None, lambdas = None, lambda_max = 100, lambda_min_ratio = 1e-6, num_lambda = 10,
                       verbose = False):
    
    if sp_oracle is None:
        sp_oracle = generate_sp_oracle(A_mat, b_vec, verbose)
    if lambdas is None:
        lambda_min = lambda_max*lambda_min_ratio
        lambdas = np.exp(np.linspace(np.log(lambda_min), np.log(lambda_max), num = num_lambda))
    if gammas is None:
        gammas = np.array([1/2])

    lambdas = lambdas[lambdas > 1e-6]  # remove lambda close to 0 
    lambdas = lambdas[lambdas <= 1]  # remove lambda larger than 1
    num_gamma = len(gammas)
    num_lambda = len(lambdas)
    validation_loss_list = np.zeros((num_gamma, num_lambda))
    
    for i in range(num_gamma):
        for j in range(num_lambda):
            gamma = gammas[i]
            cur_lambda = lambdas[j]
            
            kr = ridge_kernel(X_train, c_train, gamma = gamma, cur_lambda = cur_lambda)
            c_hat = kr.predict(np.transpose(X_val))
            err = c_hat - np.transpose(c_val)
            validation_loss_list[i, j] = np.sqrt(np.mean(np.sum(np.power(err, 2), axis = 1)))
    
    ind_list = list(np.ndindex(validation_loss_list.shape))
    best_ind = ind_list[np.argmin(validation_loss_list)]
    best_gamma = gammas[best_ind[0]]
    best_lambda = lambdas[best_ind[1]]
    
    return (best_gamma, best_lambda, validation_loss_list)

def validation_set_alg_linear_ridge(A_mat, b_vec,
                       X_train, c_train, X_val, c_val, 
                       z_star_train, w_star_train, 
                       z_star_val, w_star_val,
                       sp_oracle = None, 
                       lambdas = None, lambda_max = 100, lambda_min_ratio = 1e-6, num_lambda = 10,
                       verbose = False):
    if sp_oracle is None:
        sp_oracle = generate_sp_oracle(A_mat, b_vec, verbose)
    if lambdas is None:
        lambda_min = lambda_max*lambda_min_ratio
        lambdas = np.exp(np.linspace(np.log(lambda_min), np.log(lambda_max), num = num_lambda))
    num_lambda = len(lambdas)
    validation_loss_list = np.zeros(num_lambda)
    '这个10的-6次方是任意给定的吗？'
    for i in range(num_lambda):
        cur_lambda = lambdas[i]
        if cur_lambda >= 1e-6:
            '这里的岭回归不设置截距项，并且λ除以2'
            ridge = ridge_linear(X_train, c_train, cur_lambda = cur_lambda)
            c_hat = ridge.predict(np.transpose(X_val))
            err = c_hat - np.transpose(c_val)
            '这里的validation loss就是普通的预测误差了，和决策结果无关'
            validation_loss_list[i] = np.sqrt(np.mean(np.sum(np.power(err, 2), axis = 1)))
        if cur_lambda < 1e-6: # if lambda is close to 0, then run linear regressions 
            ls = ls_linear(X_train, c_train)
            c_hat = ls.predict(np.transpose(X_val))
            err = c_hat - np.transpose(c_val)
            validation_loss_list[i] = np.sqrt(np.mean(np.sum(np.power(err, 2), axis = 1)))

    best_ind = np.argmin(validation_loss_list)
    best_lambda = lambdas[best_ind]
    
    return (best_lambda, validation_loss_list)

def validation_set_alg_linear_SPO(A_mat, b_vec,
                       X_train, c_train, X_val, c_val, 
                       z_star_train, w_star_train, 
                       z_star_val, w_star_val,
                       sp_oracle = None, 
                       lambdas = None, lambda_max = 100, lambda_min_ratio = 1e-6, num_lambda = 10,
                       verbose = False):
    if sp_oracle is None:
        sp_oracle = generate_sp_oracle(A_mat, b_vec, verbose)
    '因为这里需要调整参数所以lambda给定的是None'
    if lambdas is None:
        lambda_min = lambda_max*lambda_min_ratio
        lambdas = np.exp(np.linspace(np.log(lambda_min), np.log(lambda_max), num = num_lambda))
    num_lambda = len(lambdas)
    '记录不同的lambda取值下所包含的验证集损失'
    validation_loss_list = np.zeros(num_lambda)
    
    for i in range(num_lambda):
        cur_lambda = lambdas[i]
        'SPO优化出现，输入的参数包括各个路径上的模型参数、带有噪声的c,最优成本以及最优决策向量，可行解约束的变量和lambda'
        B_est = SPO_reformulation_linear(X_train, c_train, z_star_train, w_star_train, A_mat, b_vec, cur_lambda = cur_lambda, verbose = verbose)
        '注意这里使用的c_hat使用的是验证集计算的，B_est相当于优化出来的每条路径的系数，x是当时的自变量'
        c_hat = spo_linear_predict(B_est, X_val)
        validation_loss_list[i] = spo_loss(c_hat, c_val, z_star_val, w_star_val, sp_oracle)


    best_ind = np.argmin(validation_loss_list)
    best_lambda = lambdas[best_ind]
    '记录下最优的lambda以及最优的SPOloss'
    return (best_lambda, validation_loss_list)

def replication_linear_spo(A_mat, b_vec, B_true, 
                             X_train, c_train, c_train_exp,
                             X_val, c_val, c_val_exp,
                             X_test, c_test, c_test_exp,
                             z_star_train, w_star_train, 
                             
                             z_star_val, w_star_val,
                             z_star_test, w_star_test, 
                             sp_oracle = None, 
                             lambdas = None, lambda_max = 100, lambda_min_ratio = 1e-6, 
                             num_lambda = 10, verbose = False):
    if sp_oracle is None:
        sp_oracle = generate_sp_oracle(A_mat, b_vec, verbose)
    
    (best_lambda_spo_linear, validation_loss_list) = validation_set_alg_linear_SPO(A_mat, b_vec,
                           X_train, c_train, X_val, c_val, 
                           z_star_train, w_star_train, 
                           z_star_val, w_star_val,
                           sp_oracle = sp_oracle, 
                           lambdas = lambdas, lambda_max = lambda_max, lambda_min_ratio = lambda_min_ratio, num_lambda = num_lambda,
                           verbose = verbose)
    
    '在经历完了相当复杂的validation_set_alg_linear_SPO函数之后，还没有结束'
    spo_linear_best = SPO_reformulation_linear(X_train, c_train, z_star_train, w_star_train, A_mat, b_vec, cur_lambda = best_lambda_spo_linear)
    '这里是对SPO又再次进行了重构，数据还是用一样的，只是lambda直接拿了最好的来算'
    c_hat_spo_linear = spo_linear_predict(spo_linear_best, X_test)
    '为什么这里要用到的是c_test_exp而不是c_test呢？在寻找最优策略时使用的是不含噪音的数据，所以这里算遗憾也要用上没有噪音的数据吗？'
    '但至少在优化模型中用到的C都是包含噪音的'
    regret_spo_linear = spo_loss(c_hat_spo_linear, c_test_exp, z_star_test, w_star_test, sp_oracle)

    return (regret_spo_linear, best_lambda_spo_linear, validation_loss_list)

def replication_linear_ridge(A_mat, b_vec, B_true, 
                             X_train, c_train, c_train_exp,
                             X_val, c_val, c_val_exp,
                             X_test, c_test, c_test_exp,
                             z_star_train, w_star_train, 
                             z_star_val, w_star_val,
                             z_star_test, w_star_test, 
                             sp_oracle = None, 
                             lambdas = None, lambda_max = 100, lambda_min_ratio = 1e-6, 
                             num_lambda = 10, verbose = False):
    if sp_oracle is None:
        sp_oracle = generate_sp_oracle(A_mat, b_vec, verbose)
    (best_lambda_ridge_linear, validation_loss_list) = validation_set_alg_linear_ridge(A_mat, b_vec,
                           X_train, c_train, X_val, c_val, 
                           z_star_train, w_star_train, 
                           z_star_val, w_star_val,
                           sp_oracle = sp_oracle, 
                           lambdas = lambdas, lambda_max = lambda_max, lambda_min_ratio = lambda_min_ratio, num_lambda = num_lambda,
                           verbose = verbose)
    if best_lambda_ridge_linear >= 1e-6:
        ridge_linear_best = ridge_linear(X_train, c_train, cur_lambda = best_lambda_ridge_linear)
        c_hat_ridge_linear = np.transpose(ridge_linear_best.predict(np.transpose(X_test)))
        regret_ridge_linear = spo_loss(c_hat_ridge_linear, c_test_exp, z_star_test, w_star_test, sp_oracle)
            # note that we are using the true regression function output c_test_exp 
            # (without noise) to evaluate the regret, and z_star_test, w_star_test are also generated 
            # from c_test_exp
    if best_lambda_ridge_linear <= 1e-6:  # if lambda is close to 0, then run linear regressions 
        ls_linear_best = ls_linear(X_train, c_train)
        c_hat_ls_linear = np.transpose(ls_linear_best.predict(np.transpose(X_test)))
        # spo_loss(c_hat, c_val, z_star_val, w_star_val, sp_oracle)
        regret_ridge_linear = spo_loss(c_hat_ls_linear, c_test_exp, z_star_test, w_star_test, sp_oracle)

    return (regret_ridge_linear, best_lambda_ridge_linear, validation_loss_list)

def replication_kernel_ridge(A_mat, b_vec, B_true, 
                             X_train, c_train, c_train_exp,
                             X_val, c_val, c_val_exp,
                             X_test, c_test, c_test_exp,
                             z_star_train, w_star_train, 
                             z_star_val, w_star_val, 
                             z_star_test, w_star_test, 
                             sp_oracle = None, 
                             gammas = None, lambdas = None, lambda_max = 100, lambda_min_ratio = 1e-6, 
                             num_lambda = 10, verbose = False):
    if sp_oracle is None:
        sp_oracle = generate_sp_oracle(A_mat, b_vec, verbose)
    (best_gamma_ridge_kernel, best_lambda_ridge_kernel, validation_loss_list) = validation_set_alg_kernel_ridge(A_mat, b_vec,
                       X_train, c_train, X_val, c_val, 
                       z_star_train, w_star_train, 
                       z_star_val, w_star_val,
                       sp_oracle = sp_oracle, 
                       gammas = gammas, lambdas = lambdas, lambda_max = lambda_max, lambda_min_ratio = lambda_min_ratio, num_lambda = num_lambda,
                       verbose = verbose)
    ridge_kernel_best = ridge_kernel(X_train, c_train, gamma = best_gamma_ridge_kernel, cur_lambda = best_lambda_ridge_kernel)
    c_hat_ridge_kernel = np.transpose(ridge_kernel_best.predict(np.transpose(X_test)))
    regret_ridge_kernel = spo_loss(c_hat_ridge_kernel, c_test_exp, z_star_test, w_star_test, sp_oracle)
    
    return (regret_ridge_kernel, best_gamma_ridge_kernel, best_lambda_ridge_kernel, validation_loss_list)

def replication_kernel_spo(A_mat, b_vec, B_true, 
                             X_train, c_train, c_train_exp,
                             X_val, c_val, c_val_exp,
                             X_test, c_test, c_test_exp,
                             z_star_train, w_star_train, 
                             z_star_val, w_star_val, 
                             z_star_test, w_star_test, 
                             sp_oracle = None, 
                             alg_type = "SGD", 
                             numiter = 1000, batchsize = 10, long_factor = 0.1,
                             gammas = None, lambdas = None, lambda_max = 100, lambda_min_ratio = 1e-6, 
                             num_lambda = 10, verbose = False):
    if sp_oracle is None:
        sp_oracle = generate_sp_oracle(A_mat, b_vec, verbose)

    (best_gamma_spo_kernel, best_lambda_spo_kernel, validation_loss_list) = validation_set_alg_kernel_SPO(A_mat, b_vec,
                           X_train, c_train, X_val, c_val, 
                           z_star_train, w_star_train, 
                           z_star_val, w_star_val, 
                           sp_oracle = sp_oracle, 
                           alg_type = alg_type, 
                           numiter = numiter, batchsize = batchsize, long_factor = long_factor,
                           gammas = gammas, lambdas = lambdas, lambda_max = lambda_max, lambda_min_ratio = lambda_min_ratio, num_lambda = num_lambda,
                           verbose = verbose)

    (K_mat_sqrt, K_mat_sqrt_inv, K_design_test) = Kernel_transformation(X_train, X_test, gamma = best_gamma_spo_kernel)

    if alg_type == "reformulation":
        spo_kernel_best =  SPO_reformulation_kernel(K_mat_sqrt, K_mat_sqrt_inv, X_train, c_train, z_star_train, w_star_train,  A_mat, b_vec, cur_lambda = best_lambda_spo_kernel)
        c_hat_spo_kernel = spo_kernel_predict(spo_kernel_best, K_design_test)
        regret_spo_kernel = spo_loss(c_hat_spo_kernel, c_test_exp, z_star_test, w_star_test, sp_oracle)
    if alg_type == "SGD":
        v_init = np.zeros((c_train.shape[0], K_mat_sqrt.shape[0]))
        v_est = spo_sgd(K_mat_sqrt, c_train, 
                     sp_oracle, z_star_train, w_star_train, v_init, best_lambda_spo_kernel, 
                     numiter =  numiter, batchsize = batchsize, long_factor = long_factor)
        c_hat_spo_kernel = spo_kernel_predict(v_est, K_design_test)
        regret_spo_kernel = spo_loss(c_hat_spo_kernel, c_test_exp, z_star_test, w_star_test, sp_oracle)

    return (regret_spo_kernel, best_gamma_spo_kernel, best_lambda_spo_kernel, validation_loss_list)
  
def replication_no_kernel(A_mat, b_vec, B_true, 
                             X_false_train, X_true_train, c_train, c_train_exp,
                              X_false_val, X_true_val, c_val, c_val_exp,
                             X_false_test, X_true_test,  c_test, c_test_exp, 
                            gammas = None, lambdas = None, lambda_max = 100, lambda_min_ratio = 1e-6, 
                             num_lambda = 10, verbose = False):

    n_train = X_false_train.shape[1]
    n_holdout = X_false_val.shape[1]
    n_test = X_false_test.shape[1]
    # generate oracle solution 
    sp_oracle = generate_sp_oracle(A_mat, b_vec, verbose)
    '输入A矩阵、B向量，确定好可行域的约束条件，然后再设定好一个优化模型，但是目前只是有目标函数、决策向量（手臂）、可行约束'
    (z_star_test, w_star_test) = oracle_dataset(c_test_exp, sp_oracle)
    '这里的c_test_exp是不含噪音的，维度为40*1000'
        # note that for test data, we are using the true regression function output c_test_exp 
        # (without noise) to generate the oracle values.
    '把50*40*1000的训练集、验证集挨个算了一遍最优情况'
    (z_star_train, w_star_train) = oracle_dataset(c_train, sp_oracle)
    (z_star_val, w_star_val) = oracle_dataset(c_val, sp_oracle)

    # misspecified spo 
    '''开始计算时间'''
    time0 = time.time() 
    '先搞错误模型的端对端优化'
    (regret_spo_false, best_lambda_spo_linear, validation_loss_spo_linear) = replication_linear_spo(A_mat, b_vec, B_true, 
                     X_false_train, c_train, c_train_exp,
                     X_false_val, c_val, c_val_exp,
                     X_false_test, c_test, c_test_exp,
                     z_star_train, w_star_train, 
                     z_star_val, w_star_val,
                     z_star_test, w_star_test, 
                     sp_oracle = sp_oracle, 
                     lambdas = lambdas, lambda_max = lambda_max, lambda_min_ratio = lambda_min_ratio, num_lambda = num_lambda,
                   verbose = verbose)
    time1 = time.time()
    spo_linear_time = time1 - time0

    # misspecified eto  
    time0 = time.time() 
    '再搞错误模型的先预测后优化，这里没有采用一般的线性回归，而是岭回归'
    (regret_ridge_linear, best_lambda_ridge_linear, validation_loss_ridge_linear) = replication_linear_ridge(A_mat, b_vec, B_true, 
                     X_false_train, c_train, c_train_exp,
                     X_false_val, c_val, c_val_exp,
                     X_false_test, c_test, c_test_exp,
                     z_star_train, w_star_train, 
                     z_star_val, w_star_val,
                     z_star_test, w_star_test, 
                     sp_oracle = sp_oracle, 
                     lambdas = lambdas, lambda_max = lambda_max, lambda_min_ratio = lambda_min_ratio, num_lambda = num_lambda,
                   verbose = verbose)
    time1 = time.time()
    ridge_linear_time = time1 - time0
    
    # correct spo 
    time0 = time.time() 
    (regret_spo_correct, best_lambda_spo_correct, validation_loss_spo_correct) = replication_linear_spo(A_mat, b_vec, B_true, 
                     X_true_train, c_train, c_train_exp,
                     X_true_val, c_val, c_val_exp,
                     X_true_test, c_test, c_test_exp,
                     z_star_train, w_star_train, 
                     z_star_val, w_star_val,
                     z_star_test, w_star_test, 
                     sp_oracle = sp_oracle, 
                     lambdas = lambdas, lambda_max = lambda_max, lambda_min_ratio = lambda_min_ratio, num_lambda = num_lambda,
                   verbose = verbose)
    time1 = time.time()
    spo_correct_time = time1 - time0

    # correct eto  
    time0 = time.time() 
    (regret_ridge_correct, best_lambda_ridge_correct, validation_loss_ridge_correct) = replication_linear_ridge(A_mat, b_vec, B_true, 
                     X_true_train, c_train, c_train_exp,
                     X_true_val, c_val, c_val_exp,
                     X_true_test, c_test, c_test_exp,
                     z_star_train, w_star_train, 
                     z_star_val, w_star_val,
                     z_star_test, w_star_test, 
                     sp_oracle = sp_oracle, 
                     lambdas = lambdas, lambda_max = lambda_max, lambda_min_ratio = lambda_min_ratio, num_lambda = num_lambda,
                   verbose = verbose)
    time1 = time.time()
    ridge_correct_time = time1 - time0

    regret_all = {"zstar_avg_test": z_star_test.mean(),
                 "SPO_wrong": regret_spo_false,
                "ETO_wrong": regret_ridge_linear,
                 "SPO_correct": regret_spo_correct,
                 "ETO_correct": regret_ridge_correct}
    validation_all = {"best_lambda_spo_linear": best_lambda_spo_linear,
                 "best_lambda_ridge_linear": best_lambda_ridge_linear,
                  "best_lambda_SPO_correct": best_lambda_spo_correct,
                 "best_lambda_ETO_correct": best_lambda_ridge_correct}
    time_all = {"SPO_wrong": spo_linear_time,
                "ETO_wrong": ridge_linear_time,
                "SPO_correct": spo_correct_time,
                 "ETO_correct": ridge_correct_time}
    validation_loss_all = {"SPO_wrong": validation_loss_spo_linear,
                "ETO_wrong": validation_loss_ridge_linear,
                "SPO_correct": validation_loss_spo_correct,
                 "ETO_correct": validation_loss_ridge_correct}
    return (regret_all, validation_all, time_all, validation_loss_all)

def replication_kernel_SGD(A_mat, b_vec, B_true, 
                              X_false_train, X_true_train, c_train, c_train_exp,
                              X_false_val, X_true_val, c_val, c_val_exp,
                             X_false_test, X_true_test,  c_test, c_test_exp, 
                             numiter = 1000, batchsize = 10, long_factor = 1,
                            gammas = None, lambdas = None, lambda_max = 100, lambda_min_ratio = 1e-6, 
                             num_lambda = 10, verbose = False):

    n_train = X_false_train.shape[1]
    n_holdout = X_false_val.shape[1]
    n_test = X_false_test.shape[1]
    # generate oracle solution 
    sp_oracle = generate_sp_oracle(A_mat, b_vec, verbose)
    (z_star_test, w_star_test) = oracle_dataset(c_test_exp, sp_oracle)
        # note that for test data, we are using the true regression function output c_test_exp 
        # (without noise) to generate the oracle values.
    (z_star_train, w_star_train) = oracle_dataset(c_train, sp_oracle)
    (z_star_val, w_star_val) = oracle_dataset(c_val, sp_oracle)

    # kernel spo 
    time0 = time.time() 
    (regret_spo_kernel, best_gamma_spo_kernel, best_lambda_spo_kernel, validation_loss_spo_kernel) = replication_kernel_spo(A_mat, b_vec, B_true, 
                             X_false_train, c_train, c_train_exp,
                             X_false_val, c_val, c_val_exp,
                             X_false_test, c_test, c_test_exp,
                             z_star_train, w_star_train, 
                             z_star_val, w_star_val,
                             z_star_test, w_star_test, 
                             sp_oracle = sp_oracle, 
                             alg_type = "SGD", 
                             numiter = numiter, batchsize = batchsize, long_factor = long_factor,
                             gammas = gammas, lambdas = lambdas, lambda_max = lambda_max, lambda_min_ratio = lambda_min_ratio, num_lambda = num_lambda,
                             verbose = verbose)
    time1 = time.time()
    spo_kernel_time = time1 - time0

    # kernel eto  
    time0 = time.time() 
    (regret_ridge_kernel, best_gamma_ridge_kernel, best_lambda_ridge_kernel, validation_loss_ridge_kernel) = replication_kernel_ridge(A_mat, b_vec, B_true, 
                             X_false_train, c_train, c_train_exp,
                             X_false_val, c_val, c_val_exp,
                             X_false_test, c_test, c_test_exp,
                             z_star_train, w_star_train, 
                             z_star_val, w_star_val,
                             z_star_test, w_star_test, 
                             sp_oracle = sp_oracle, 
                             gammas = gammas, lambdas = lambdas, lambda_max = lambda_max, lambda_min_ratio = lambda_min_ratio, num_lambda = num_lambda,
                             verbose = verbose)
    time1 = time.time()
    ridge_kernel_time = time1 - time0
    
    

    regret_all = {"SPO_kernel": regret_spo_kernel, 
                "ETO_kernel": regret_ridge_kernel}
    validation_all = {"best_gamma_spo_kernel": best_gamma_spo_kernel, 
                "best_lambda_spo_kernel": best_lambda_spo_kernel,
                 "best_gamma_ridge_kernel": best_gamma_ridge_kernel,
                 "best_lambda_ridge_kernel": best_lambda_ridge_kernel}
    time_all = {"SPO_kernel": spo_kernel_time, 
                "ETO_kernel": ridge_kernel_time}
    validation_loss_all = {"SPO_kernel": validation_loss_spo_kernel, 
                "ETO_kernel": validation_loss_ridge_kernel}
    return (regret_all, validation_all, time_all, validation_loss_all)

def replication_kernel_gurobi(A_mat, b_vec, B_true, 
                             X_false_train, X_true_train, c_train, c_train_exp,
                              X_false_val, X_true_val, c_val, c_val_exp,
                             X_false_test, X_true_test,  c_test, c_test_exp, 
                             numiter = 1000, batchsize = 10, long_factor = 0.1,
                            gammas = None, lambdas = None, lambda_max = 100, lambda_min_ratio = 1e-6, 
                             num_lambda = 10, verbose = False):

    n_train = X_false_train.shape[1]
    n_holdout = X_false_val.shape[1]
    n_test = X_false_test.shape[1]
    # generate oracle solution 
    sp_oracle = generate_sp_oracle(A_mat, b_vec, verbose)
    (z_star_test, w_star_test) = oracle_dataset(c_test_exp, sp_oracle)
        # note that for test data, we are using the true regression function output c_test_exp 
        # (without noise) to generate the oracle values.
    (z_star_train, w_star_train) = oracle_dataset(c_train, sp_oracle)
    (z_star_val, w_star_val) = oracle_dataset(c_val, sp_oracle)

    # kernel spo reformulation
    time0 = time.time() 
    (regret_spo_kernel, best_gamma_spo_kernel, best_lambda_spo_kernel, validation_loss_spo_kernel) = replication_kernel_spo(A_mat, b_vec, B_true, 
                             X_false_train, c_train, c_train_exp,
                             X_false_val, c_val, c_val_exp,
                             X_false_test, c_test, c_test_exp,
                             z_star_train, w_star_train, 
                             z_star_val, w_star_val,
                             z_star_test, w_star_test, 
                             sp_oracle = sp_oracle, 
                             alg_type = "reformulation", 
                             numiter = numiter, batchsize = batchsize, long_factor = long_factor,
                             gammas = gammas, lambdas = lambdas, lambda_max = lambda_max, lambda_min_ratio = lambda_min_ratio, num_lambda = num_lambda,
                             verbose = verbose)
    time1 = time.time()
    spo_kernel_time = time1 - time0
    

    regret_all = {"SPO_kernel": regret_spo_kernel}
    validation_all = {"best_gamma_spo_kernel": best_gamma_spo_kernel, 
                "best_lambda_spo_kernel": best_lambda_spo_kernel}
    time_all = {"SPO_kernel": spo_kernel_time}
    validation_loss_all = {"SPO_kernel": validation_loss_spo_kernel}
    return (regret_all, validation_all, time_all, validation_loss_all)



