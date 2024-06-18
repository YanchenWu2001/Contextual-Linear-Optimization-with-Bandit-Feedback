# Contextual-Linear-Optimization-with-Bandit-Feedback
Replication code for the paper "Contextual Linear Optimization with Bandit Feedback".

The Random_policy file corresponds to Tables 1 and 2 in the body of the paper, and Tables 3 and 4 in the appendix.

The X1_policy file corresponds to Tables 5 and 6 in the appendix.

The X1X2_policy file corresponds to Tables 7 and 8 in the appendix.

## Details

The meaning of no kernel in the file name is that we do not use the kernel method, the meaning of tree is that we use the decision tree model to learn the policy in X1_policy and X1X2_policy settings, the meaning of Sample is to estimate the policy according to the probability of occurrence of historical data in Random policy setting, and the meaning of Linear Wrong is using the misspecified Nuisance model.

### Table 1 and 3:

● ETO: 

Random_policy/ETO/ETO.py --> Random_policy/ETO/regret_all_Wrong_ETO.csv & regret_all_Correct_ETO.csv --> Result_processing_example_code.R --> Random_policy/ETO/Regret_ETO.csv --> Random_policy/ETO/Regret_ETO.xlsx

● SPO+ DM & SPO+ DR PI & SPO+ DR Lambda & SPO+ DR Clip: 

Random_policy/SPO+DM_DR_PI_Lamdba_Clip/SPO+DM_DR_PI_Lamdba_Clip.py --> Random_policy/SPO+DM_DR_PI_Lamdba_Clip/regret_no_kernel_Direct_Model.csv & regret_no_kernel_Sample_Double_Robust.csv & 
regret_no_kernel_Lambda_Sample_Double_Robust.csv & regret_no_kernel_Clip_Sample_Double_Robust.csv --> Result_processing_example_code.R --> Random_policy/SPO+DM_DR_PI_Lamdba_Clip/Nuisance_model_well_specified_result.csv --> Random_policy/SPO+DM_DR_PI_Lamdba_Clip/Nuisance_model_well_specified_result.xlsx

● SPO+ ISW

Random_policy/ISW/ISW.py --> Random_policy/ISW/regret_no_kernel_ISW_Sample.csv --> Result_processing_example_code.R --> Random_policy/ETO/ISW_result.csv --> Random_policy/ETO/ISW_result.xlsx

● Naive ETO

Random_policy/Naive_ETO/Naive_ETO.py --> Random_policy/Naive_ETO/regret_all_Correct_ETO.csv & regret_all_Wrong_ETO.csv --> Result_processing_example_code.R --> Random_policy/Naive_ETO/Regret_ETO.csv --> Random_policy/Naive_ETO/Regret_ETO.xlsx

● Naive SPO+ DM

Random_policy/Naive_SPO+DM/Naive_SPO+DM.py --> Random_policy/regret_no_kernel_CDM & regret_no_kernel_Wrong_CDM --> Result_processing_example_code.R --> Random_policy/Naive_SPO+DM/Naive_SPO+DM_result.csv --> Random_policy/Naive_SPO+DM/Naive_SPO+DM_result.xlsx

● Naive SPO+ DR

Random_policy/Naive_SPO+DR/Naive_SPO+DR.py --> Random_policy/regret_no_kernel_CDR & regret_no_kernel_Wrong_CDR --> Result_processing_example_code.R --> Random_policy/Naive_SPO+DR/Naive_SPO+DR_result.csv --> Random_policy/Naive_SPO+DR/Naive_SPO+DR_result.xlsx

● Naive SPO+ IPW

Random_policy/Naive_SPO+IPW/Naive_SPO+IPW.py --> Random_policy/Naive_SPO+IPW/regret_no_kernel_IPW.csv --> Result_processing_example_code.R --> Random_policy/Naive_SPO+IPW/Naive_SPO+IPW_result.csv --> Random_policy/Naive_SPO+IPW/Naive_SPO+IPW_result.xlsx

### Table 2 and 4:

● ETO (misspecified degree 2): 

Random_policy/ETO_misspecified2/ETO_misspecified2.py --> Random_policy/ETO/regret_all_Wrong_ETO.csv & regret_all_Correct_ETO.csv --> Result_processing_example_code.R --> Random_policy/ETO_misspecified2/Regret_ETO.csv --> Random_policy/ETO_misspecified2/Regret_ETO.xlsx

● ETO (misspecified degree 4): 

Random_policy/ETO/ETO.py --> Random_policy/ETO/regret_all_Wrong_ETO.csv & regret_all_Correct_ETO.csv --> Result_processing_example_code.R --> Random_policy/ETO/Regret_ETO.csv --> Random_policy/ETO/Regret_ETO.xlsx

● SPO+ DM & SPO+ DR PI & SPO+ DR Lambda & SPO+ DR Clip (Nuisance Model $F^{\text{N}}$ is well-specified and Policy-inducing model $F$ is misspecified degree 4) : 

Random_policy/SPO+DM_DR_PI_Lamdba_Clip/SPO+DM_DR_PI_Lamdba_Clip.py --> Random_policy/SPO+DM_DR_PI_Lamdba_Clip/regret_no_kernel_Direct_Model.csv & regret_no_kernel_Sample_Double_Robust.csv & 
regret_no_kernel_Lambda_Sample_Double_Robust.csv & regret_no_kernel_Clip_Sample_Double_Robust.csv --> Result_processing_example_code.R --> Random_policy/SPO+DM_DR_PI_Lamdba_Clip/Nuisance_model_well_specified_result.csv --> Random_policy/SPO+DM_DR_PI_Lamdba_Clip/Nuisance_model_well_specified_result.xlsx

● SPO+ DM & SPO+ DR PI & SPO+ DR Lambda & SPO+ DR Clip (Nuisance Model $F^{\text{N}}$ is misspecified degree 4 and Policy-inducing model $F$ is well-specified or Nuisance Model $F^{\text{N}}$ and Nuisance Model $F^{\text{N}}$ are both misspecified degree 4) : 

Random_policy/SPO+DM_DR_PI_Lamdba_Clip/SPO+DM_DR_PI_Lamdba_Clip.py --> Random_policy/SPO+DM_DR_PI_Lamdba_Clip/regret_no_kernel_Linear_Wrong_Direct_Model.csv &  regret_no_kernel_Linear_Wrong_Sample_Double_Robust.csv & 
regret_no_kernel_Lambda_Linear_Wrong_Sample_Double_Robust.csv & regret_no_kernel_Clip_Linear_Wrong_Sample_Double_Robust.csv --> Result_processing_example_code.R --> Random_policy/SPO+DM_DR_PI_Lamdba_Clip/Nuisance_model_misspecified_result.csv --> Random_policy/SPO+DM_DR_PI_Lamdba_Clip/Nuisance_model_misspecified_result.xlsx

● SPO+ DM & SPO+ DR PI & SPO+ DR Lambda & SPO+ DR Clip (Nuisance Model $F^{\text{N}}$ is well-specified and Policy-inducing model $F$ is misspecified degree 2) : 

Random_policy/SPO+DM_DR_PI_Lamdba_Clip_misspecified2/SPO+DM_DR_PI_Lamdba_Clip_misspecified.py --> Random_policy/SPO+DM_DR_PI_Lamdba_Clip/regret_no_kernel_Direct_Model.csv & regret_no_kernel_Sample_Double_Robust.csv & 
regret_no_kernel_Lambda_Sample_Double_Robust.csv & regret_no_kernel_Clip_Sample_Double_Robust.csv --> Result_processing_example_code.R --> Random_policy/SPO+DM_DR_PI_Lamdba_Clip_misspecified2/Nuisance_model_well_specified_result.csv --> Random_policy/SPO+DM_DR_PI_Lamdba_Clip_misspecified2/Nuisance_model_well_specified_result.xlsx
