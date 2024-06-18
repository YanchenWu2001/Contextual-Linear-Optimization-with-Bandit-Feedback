# Contextual-Linear-Optimization-with-Bandit-Feedback
Replication code for the paper "Contextual Linear Optimization with Bandit Feedback".

The Random_policy file corresponds to Tables 1 and 2 in the body of the paper, and Tables 3 and 4 in the appendix.

The X1_policy file corresponds to Tables 5 and 6 in the appendix.

The X1X2_policy file corresponds to Tables 7 and 8 in the appendix.

## Details

File name interpretation

The meaning of no kernel in the file name is that we do not use the kernel method, the meaning of tree is that we use the decision tree model to learn the policy in X1_policy and X1X2_policy settings, the meaning of Sample is to estimate the policy according to the probability of occurrence of historical data in Random policy setting, and the meaning of Linear Wrong is using the misspecified Nuisance model.

### Table 1 and 3

● ETO

Random_policy/ETO/ETO.py --> Random_policy/ETO/regret_all_Wrong_ETO.csv & regret_all_Correct_ETO.csv --> Result_processing_example_code.R --> Random_policy/ETO/Regret_ETO.csv --> Random_policy/ETO/Regret_ETO.xlsx

● SPO+ DM & SPO+ DR PI & SPO+ DR Lambda & SPO+ DR Clip

Random_policy/SPO+DM_DR_PI_Lamdba_Clip/SPO+DM_DR_PI_Lamdba_Clip.py --> Random_policy/SPO+DM_DR_PI_Lamdba_Clip/regret_no_kernel_Direct_Model.csv & regret_no_kernel_Sample_Double_Robust.csv & regret_no_kernel_Lambda_Sample_Double_Robust.csv & regret_no_kernel_Clip_Sample_Double_Robust.csv --> Result_processing_example_code.R --> Random_policy/SPO+DM_DR_PI_Lamdba_Clip/Nuisance_model_well_specified_result.csv --> Random_policy/SPO+DM_DR_PI_Lamdba_Clip/Nuisance_model_well_specified_result.xlsx

● SPO+ ISW

Random_policy/ISW/ISW.py --> Random_policy/ISW/regret_no_kernel_ISW_Sample.csv --> Result_processing_example_code.R --> Random_policy/ETO/ISW_result.csv --> Random_policy/ETO/ISW_result.xlsx

● Naive ETO

Random_policy/Naive_ETO/Naive_ETO.py --> Random_policy/Naive_ETO/regret_all_Correct_ETO.csv & regret_all_Wrong_ETO.csv --> Result_processing_example_code.R --> Random_policy/Naive_ETO/Regret_ETO.csv --> Random_policy/Naive_ETO/Regret_ETO.xlsx

● Naive SPO+ DM

Random_policy/Naive_SPO+DM/Naive_SPO+DM.py --> Random_policy/Naive_SPO+DM/regret_no_kernel_CDM & regret_no_kernel_Wrong_CDM --> Result_processing_example_code.R --> Random_policy/Naive_SPO+DM/Naive_SPO+DM_result.csv --> Random_policy/Naive_SPO+DM/Naive_SPO+DM_result.xlsx

● Naive SPO+ DR

Random_policy/Naive_SPO+DR/Naive_SPO+DR.py --> Random_policy/Naive_SPO+DR/regret_no_kernel_CDR & regret_no_kernel_Wrong_CDR --> Result_processing_example_code.R --> Random_policy/Naive_SPO+DR/Naive_SPO+DR_result.csv --> Random_policy/Naive_SPO+DR/Naive_SPO+DR_result.xlsx

● Naive SPO+ IPW

Random_policy/Naive_SPO+IPW/Naive_SPO+IPW.py --> Random_policy/Naive_SPO+IPW/regret_no_kernel_IPW.csv --> Result_processing_example_code.R --> Random_policy/Naive_SPO+IPW/Naive_SPO+IPW_result.csv --> Random_policy/Naive_SPO+IPW/Naive_SPO+IPW_result.xlsx

### Table 2 and 4

● ETO (misspecified degree 2)

Random_policy/ETO_misspecified2/ETO_misspecified2.py --> Random_policy/ETO_misspecified2/regret_all_Wrong_ETO.csv & regret_all_Correct_ETO.csv --> Result_processing_example_code.R --> Random_policy/ETO_misspecified2/Regret_ETO.csv --> Random_policy/ETO_misspecified2/Regret_ETO.xlsx

● ETO (misspecified degree 4)

Random_policy/ETO/ETO.py --> Random_policy/ETO/regret_all_Wrong_ETO.csv & regret_all_Correct_ETO.csv --> Result_processing_example_code.R --> Random_policy/ETO/Regret_ETO.csv --> Random_policy/ETO/Regret_ETO.xlsx

● SPO+ DM & SPO+ DR PI & SPO+ DR Lambda & SPO+ DR Clip 

1. Nuisance Model $F^{\text{N}}$ is well-specified and Policy-inducing model $F$ is misspecified degree 4

Random_policy/SPO+DM_DR_PI_Lamdba_Clip/SPO+DM_DR_PI_Lamdba_Clip.py --> Random_policy/SPO+DM_DR_PI_Lamdba_Clip/regret_no_kernel_Direct_Model.csv & regret_no_kernel_Sample_Double_Robust.csv & regret_no_kernel_Lambda_Sample_Double_Robust.csv & regret_no_kernel_Clip_Sample_Double_Robust.csv --> Result_processing_example_code.R --> Random_policy/SPO+DM_DR_PI_Lamdba_Clip/Nuisance_model_well_specified_result.csv --> Random_policy/SPO+DM_DR_PI_Lamdba_Clip/Nuisance_model_well_specified_result.xlsx

● SPO+ DM & SPO+ DR PI & SPO+ DR Lambda & SPO+ DR Clip 

1. Nuisance Model $F^{\text{N}}$ is misspecified degree 4, Policy-inducing model $F$ is well-specified 
2. Nuisance Model $F^{\text{N}}$, Nuisance Model $F^{\text{N}}$ are both misspecified degree 4

Random_policy/SPO+DM_DR_PI_Lamdba_Clip/SPO+DM_DR_PI_Lamdba_Clip.py --> Random_policy/SPO+DM_DR_PI_Lamdba_Clip/regret_no_kernel_Linear_Wrong_Direct_Model.csv &  regret_no_kernel_Linear_Wrong_Sample_Double_Robust.csv & regret_no_kernel_Lambda_Linear_Wrong_Sample_Double_Robust.csv & regret_no_kernel_Clip_Linear_Wrong_Sample_Double_Robust.csv --> Result_processing_example_code.R --> Random_policy/SPO+DM_DR_PI_Lamdba_Clip/Nuisance_model_misspecified_result.csv --> Random_policy/SPO+DM_DR_PI_Lamdba_Clip/Nuisance_model_misspecified_result.xlsx

● SPO+ DM & SPO+ DR PI & SPO+ DR Lambda & SPO+ DR Clip

1. Nuisance Model $F^{\text{N}}$ is well-specified and Policy-inducing model $F$ is misspecified degree 2

Random_policy/SPO+DM_DR_PI_Lamdba_Clip_misspecified2/SPO+DM_DR_PI_Lamdba_Clip_misspecified2.py --> Random_policy/SPO+DM_DR_PI_Lamdba_Clip_misspecified2/regret_no_kernel_Direct_Model.csv & regret_no_kernel_Sample_Double_Robust.csv & regret_no_kernel_Lambda_Sample_Double_Robust.csv & regret_no_kernel_Clip_Sample_Double_Robust.csv --> Result_processing_example_code.R --> Random_policy/SPO+DM_DR_PI_Lamdba_Clip_misspecified2/Nuisance_model_well_specified_result.csv --> Random_policy/SPO+DM_DR_PI_Lamdba_Clip_misspecified2/Nuisance_model_well_specified_result.xlsx

● SPO+ DM & SPO+ DR PI & SPO+ DR Lambda & SPO+ DR Clip 

1. Nuisance Model $F^{\text{N}}$ is misspecified degree 2, Policy-inducing model $F$ is well-specified 
2. Nuisance Model $F^{\text{N}}$, Nuisance Model $F^{\text{N}}$ are both misspecified degree 2

Random_policy/SPO+DM_DR_PI_Lamdba_Clip_misspecified2/SPO+DM_DR_PI_Lamdba_Clip_misspecified2.py --> Random_policy/SPO+DM_DR_PI_Lamdba_Clip_misspecified2/regret_no_kernel_Linear_Wrong_Direct_Model.csv &  regret_no_kernel_Linear_Wrong_Sample_Double_Robust.csv & regret_no_kernel_Lambda_Linear_Wrong_Sample_Double_Robust.csv & regret_no_kernel_Clip_Linear_Wrong_Sample_Double_Robust.csv --> Result_processing_example_code.R --> Random_policy/SPO+DM_DR_PI_Lamdba_Clip_misspecified2/Nuisance_model_misspecified2_result.csv --> Random_policy/SPO+DM_DR_PI_Lamdba_Clip_misspecified2/Nuisance_model_misspecified2_result.xlsx

● Naive SPO+ DM 

1. Nuisance Model $F^{\text{N}}$ is well-specified, Policy-inducing model $F$ is misspecified degree 4
2. $F^{\text{N}}$ is misspecified degree 4, Policy-inducing model $F$ is well-specified
3. Nuisance Model $F^{\text{N}}$, Nuisance Model $F^{\text{N}}$ are both misspecified degree 4

Random_policy/Naive_SPO+DM/Naive_SPO+DM.py --> Random_policy/Naive_SPO+DM/regret_no_kernel_CDM & regret_no_kernel_Wrong_CDM --> Result_processing_example_code.R --> Random_policy/Naive_SPO+DM/Naive_SPO+DM_result.csv --> Random_policy/Naive_SPO+DM/Naive_SPO+DM_result.xlsx

● Naive SPO+ DM 

1. Nuisance Model $F^{\text{N}}$ is well-specified, Policy-inducing model $F$ is misspecified degree 2
2. $F^{\text{N}}$ is misspecified degree 2, Policy-inducing model $F$ is well-specified
3. Nuisance Model $F^{\text{N}}$, Nuisance Model $F^{\text{N}}$ are both misspecified degree 2

Random_policy/Naive_SPO+DM_misspecified2/Naive_SPO+DM_misspecified2.py --> Random_policy/Naive_SPO+DM_misspecified2/regret_no_kernel_CDM & regret_no_kernel_Wrong_CDM --> Result_processing_example_code.R --> Random_policy/Naive_SPO+DM_misspecified2/Naive_SPO+DM_misspecified2_result.csv --> Random_policy/Naive_SPO+DM_misspecified2/Naive_SPO+DM_misspecified2_result.xlsx

● Naive SPO+ DR 

1. Nuisance Model $F^{\text{N}}$ is well-specified, Policy-inducing model $F$ is misspecified degree 4
2. $F^{\text{N}}$ is misspecified degree 4, Policy-inducing model $F$ is well-specified
3. Nuisance Model $F^{\text{N}}$, Nuisance Model $F^{\text{N}}$ are both misspecified degree 4

Random_policy/Naive_SPO+DR/Naive_SPO+DR.py --> Random_policy/Naive_SPO+DR/regret_no_kernel_CDR & regret_no_kernel_Wrong_CDR --> Result_processing_example_code.R --> Random_policy/Naive_SPO+DR/Naive_SPO+DR_result.csv --> Random_policy/Naive_SPO+DR/Naive_SPO+DR_result.xlsx

● Naive SPO+ DR 

1. Nuisance Model $F^{\text{N}}$ is well-specified, Policy-inducing model $F$ is misspecified degree 2
2. $F^{\text{N}}$ is misspecified degree 2, Policy-inducing model $F$ is well-specified
3. Nuisance Model $F^{\text{N}}$, Nuisance Model $F^{\text{N}}$ are both misspecified degree 2

Random_policy/Naive_SPO+DR_misspecified2/Naive_SPO+DR_misspecified2.py --> Random_policy/Naive_SPO+DR_misspecified2/regret_no_kernel_CDR & regret_no_kernel_Wrong_CDR --> Result_processing_example_code.R --> Random_policy/Naive_SPO+DR_misspecified2/Naive_SPO+DR_misspecified2_result.csv --> Random_policy/Naive_SPO+DR_misspecified2/Naive_SPO+DR_misspecified2_result.xlsx

### Table 5

● ETO

X1_policy/X1_ETO/X1_ETO.py --> X1_policy/X1_ETO/regret_all_Wrong_ETO.csv & regret_all_Correct_ETO.csv --> Result_processing_example_code.R --> X1_policy/X1_ETO/Regret_ETO.csv --> X1_policy/X1_ETO/Regret_ETO.xlsx

● SPO+ DM & SPO+ DR PI & SPO+ DR Lambda & SPO+ DR Clip

X1_policy/X1_SPO+DM_DR_PI_Lamdba_Clip/X1_SPO+DM_DR_PI_Lamdba_Clip.py --> X1_policy/X1_SPO+DM_DR_PI_Lamdba_Clip/regret_no_kernel_Direct_Model.csv & regret_no_kernel_Sample_Double_Robust_tree.csv & regret_no_kernel_Lambda_Sample_Double_Robust_tree.csv & regret_no_kernel_Clip_Sample_Double_Robust_tree.csv --> Result_processing_example_code.R --> X1_policy/X1_SPO+DM_DR_PI_Lamdba_Clip/X1_Nuisance_model_well_specified_result.csv --> X1_policy/X1_SPO+DM_DR_PI_Lamdba_Clip/X1_Nuisance_model_well_specified_result.xlsx

● SPO+ ISW

X1_policy/X1_ISW/X1_ISW.py --> X1_policy/X1_ISW/regret_no_kernel_ISW_Sample.csv --> Result_processing_example_code.R --> X1_policy/X1_ETO/X1_ISW_result.csv --> X1_policy/X1_ETO/X1_ISW_result.xlsx

● Naive ETO

X1_policy/X1_Naive_ETO/X1_Naive_ETO.py --> X1_policy/X1_Naive_ETO/regret_all_Correct_ETO.csv & regret_all_Wrong_ETO.csv --> Result_processing_example_code.R --> X1_policy/X1_Naive_ETO/Regret_ETO.csv --> X1_policy/X1_Naive_ETO/Regret_ETO.xlsx

● Naive SPO+ DM

X1_policy/X1_Naive_SPO+DM/X1_Naive_SPO+DM.py --> X1_policy/X1_Naive_SPO+DM/regret_no_kernel_CDM & regret_no_kernel_Wrong_CDM --> Result_processing_example_code.R --> X1_policy/X1_Naive_SPO+DM/X1_Naive_SPO+DM_result.csv --> X1_policy/X1_Naive_SPO+DM/X1_Naive_SPO+DM_result.xlsx

● Naive SPO+ DR

X1_policy/X1_Naive_SPO+DR/X1_Naive_SPO+DR.py --> X1_policy/X1_Naive_SPO+DR/regret_no_kernel_CDR & regret_no_kernel_Wrong_CDR --> Result_processing_example_code.R --> X1_policy/X1_Naive_SPO+DR/X1_Naive_SPO+DR_result.csv --> X1_policy/X1_Naive_SPO+DR/X1_Naive_SPO+DR_result.xlsx

● Naive SPO+ IPW

X1_policy/X1_Naive_SPO+IPW/X1_Naive_SPO+IPW.py --> X1_policy/X1_Naive_SPO+IPW/regret_no_kernel_IPW.csv --> Result_processing_example_code.R --> X1_policy/X1_Naive_SPO+IPW/X1_Naive_SPO+IPW_result.csv --> X1_policy/X1_Naive_SPO+IPW/X1_Naive_SPO+IPW_result.xlsx

### Table 6

● ETO (misspecified degree 2)

X1_policy/X1_ETO_misspecified2/X1_ETO_misspecified2.py --> X1_policy/X1_ETO_misspecified2/regret_all_Wrong_ETO.csv & regret_all_Correct_ETO.csv --> Result_processing_example_code.R --> X1_policy/X1_ETO_misspecified2/Regret_ETO.csv --> X1_policy/X1_ETO_misspecified2/Regret_ETO.xlsx

● ETO (misspecified degree 4)

X1_policy/X1_ETO/X1_ETO.py --> X1_policy/X1_ETO/regret_all_Wrong_ETO.csv & regret_all_Correct_ETO.csv --> Result_processing_example_code.R --> X1_policy/X1_ETO/Regret_ETO.csv --> X1_policy/X1_ETO/Regret_ETO.xlsx

● SPO+ DM & SPO+ DR PI & SPO+ DR Lambda & SPO+ DR Clip 

1. Nuisance Model $F^{\text{N}}$ is well-specified and Policy-inducing model $F$ is misspecified degree 4

X1_policy/X1_SPO+DM_DR_PI_Lamdba_Clip/X1_SPO+DM_DR_PI_Lamdba_Clip.py --> X1_policy/X1_SPO+DM_DR_PI_Lamdba_Clip/regret_no_kernel_Direct_Model_tree.csv & regret_no_kernel_Sample_Double_Robust_tree.csv & regret_no_kernel_Lambda_Sample_Double_Robust_tree.csv & regret_no_kernel_Clip_Sample_Double_Robust_tree.csv --> Result_processing_example_code.R --> X1_policy/X1_SPO+DM_DR_PI_Lamdba_Clip/X1_Nuisance_model_well_specified_result.csv --> X1_policy/X1_SPO+DM_DR_PI_Lamdba_Clip/X1_Nuisance_model_well_specified_result.xlsx

● SPO+ DM & SPO+ DR PI & SPO+ DR Lambda & SPO+ DR Clip 

1. Nuisance Model $F^{\text{N}}$ is misspecified degree 4, Policy-inducing model $F$ is well-specified 
2. Nuisance Model $F^{\text{N}}$, Nuisance Model $F^{\text{N}}$ are both misspecified degree 4

X1_policy/X1_SPO+DM_DR_PI_Lamdba_Clip/X1_SPO+DM_DR_PI_Lamdba_Clip.py --> X1_policy/X1_SPO+DM_DR_PI_Lamdba_Clip/regret_no_kernel_Linear_Wrong_Direct_Model.csv &  regret_no_kernel_Linear_Wrong_Double_Robust_Tree_Sigma.csv & regret_no_kernel_Lambda_Linear_Wrong_Double_Robust_Tree_Sigma.csv & regret_no_kernel_Clip_Linear_Wrong_Double_Robust_Tree_Sigma.csv --> Result_processing_example_code.R --> X1_policy/X1_SPO+DM_DR_PI_Lamdba_Clip/X1_Nuisance_model_misspecified_result.csv --> X1_policy/X1_SPO+DM_DR_PI_Lamdba_Clip/X1_Nuisance_model_misspecified_result.xlsx

● SPO+ DM & SPO+ DR PI & SPO+ DR Lambda & SPO+ DR Clip 

1. Nuisance Model $F^{\text{N}}$ is well-specified and Policy-inducing model $F$ is misspecified degree 2

X1_policy/X1_SPO+DM_DR_PI_Lamdba_Clip_misspecified2/X1_SPO+DM_DR_PI_Lamdba_Clip_misspecified2.py --> X1_policy/X1_SPO+DM_DR_PI_Lamdba_Clip_misspecified2/regret_no_kernel_Direct_Model_tree.csv & regret_no_kernel_Sample_Double_Robust_tree.csv & regret_no_kernel_Lambda_Sample_Double_Robust_tree.csv & regret_no_kernel_Clip_Sample_Double_Robust_tree.csv --> Result_processing_example_code.R --> X1_policy/X1_SPO+DM_DR_PI_Lamdba_Clip_misspecified2/X1_Nuisance_model_well_specified_result.csv --> X1_policy/X1_SPO+DM_DR_PI_Lamdba_Clip_misspecified2/X1_Nuisance_model_well_specified_result.xlsx

● SPO+ DM & SPO+ DR PI & SPO+ DR Lambda & SPO+ DR Clip 

1. Nuisance Model $F^{\text{N}}$ is misspecified degree 2, Policy-inducing model $F$ is well-specified 
2. Nuisance Model $F^{\text{N}}$, Nuisance Model $F^{\text{N}}$ are both misspecified degree 2

X1_policy/X1_SPO+DM_DR_PI_Lamdba_Clip_misspecified2/X1_SPO+DM_DR_PI_Lamdba_Clip_misspecified2.py --> X1_policy/X1_SPO+DM_DR_PI_Lamdba_Clip_misspecified2/regret_no_kernel_Linear_Wrong_Direct_Model.csv &  regret_no_kernel_Linear_Wrong_Double_Robust_Tree_Sigma.csv & regret_no_kernel_Lambda_Linear_Wrong_Double_Robust_Tree_Sigma.csv & regret_no_kernel_Clip_Linear_Wrong_Double_Robust_Tree_Sigma.csv --> Result_processing_example_code.R --> X1_policy/X1_SPO+DM_DR_PI_Lamdba_Clip_misspecified2/X1_Nuisance_model_misspecified2_result.csv --> X1_policy/X1_SPO+DM_DR_PI_Lamdba_Clip_misspecified2/X1_Nuisance_model_misspecified2_result.xlsx

● Naive SPO+ DM 

1. Nuisance Model $F^{\text{N}}$ is well-specified, Policy-inducing model $F$ is misspecified degree 4
2. $F^{\text{N}}$ is misspecified degree 4, Policy-inducing model $F$ is well-specified
3. Nuisance Model $F^{\text{N}}$, Nuisance Model $F^{\text{N}}$ are both misspecified degree 4

X1_policy/X1_Naive_SPO+DM/X1_Naive_SPO+DM.py --> X1_policy/X1_Naive_SPO+DM/regret_no_kernel_CDM & regret_no_kernel_Wrong_CDM --> Result_processing_example_code.R --> X1_policy/X1_Naive_SPO+DM/X1_Naive_SPO+DM_result.csv --> X1_policy/X1_Naive_SPO+DM/X1_Naive_SPO+DM_result.xlsx

● Naive SPO+ DM 

1. Nuisance Model $F^{\text{N}}$ is well-specified, Policy-inducing model $F$ is misspecified degree 2
2. $F^{\text{N}}$ is misspecified degree 2, Policy-inducing model $F$ is well-specified
3. Nuisance Model $F^{\text{N}}$, Nuisance Model $F^{\text{N}}$ are both misspecified degree 2

X1_policy/X1_Naive_SPO+DM_misspecified2/X1_Naive_SPO+DM_misspecified2.py --> X1_policy/X1_Naive_SPO+DM_misspecified2/regret_no_kernel_CDM & regret_no_kernel_Wrong_CDM --> Result_processing_example_code.R --> X1_policy/X1_Naive_SPO+DM_misspecified2/X1_Naive_SPO+DM_misspecified2_result.csv --> X1_policy/X1_Naive_SPO+DM_misspecified2/X1_Naive_SPO+DM_misspecified2_result.xlsx

● Naive SPO+ DR 

1. Nuisance Model $F^{\text{N}}$ is well-specified, Policy-inducing model $F$ is misspecified degree 4
2. $F^{\text{N}}$ is misspecified degree 4, Policy-inducing model $F$ is well-specified
3. Nuisance Model $F^{\text{N}}$, Nuisance Model $F^{\text{N}}$ are both misspecified degree 4

X1_policy/X1_Naive_SPO+DR/X1_Naive_SPO+DR.py --> X1_policy/X1_Naive_SPO+DR/regret_no_kernel_CDR & regret_no_kernel_Wrong_CDR --> Result_processing_example_code.R --> X1_policy/X1_Naive_SPO+DR/X1_Naive_SPO+DR_result.csv --> X1_policy/X1_Naive_SPO+DR/X1_Naive_SPO+DR_result.xlsx

● Naive SPO+ DR 

1. Nuisance Model $F^{\text{N}}$ is well-specified, Policy-inducing model $F$ is misspecified degree 2
2. $F^{\text{N}}$ is misspecified degree 2, Policy-inducing model $F$ is well-specified
3. Nuisance Model $F^{\text{N}}$, Nuisance Model $F^{\text{N}}$ are both misspecified degree 2

X1_policy/X1_Naive_SPO+DR_misspecified2/X1_Naive_SPO+DR_misspecified2.py --> X1_policy/X1_Naive_SPO+DR_misspecified2/regret_no_kernel_CDR & regret_no_kernel_Wrong_CDR --> Result_processing_example_code.R --> X1_policy/X1_Naive_SPO+DR_misspecified2/X1_Naive_SPO+DR_misspecified2_result.csv --> X1_policy/X1_Naive_SPO+DR_misspecified2/X1_Naive_SPO+DR_misspecified2_result.xlsx

### Table 7

● ETO

X1X2_policy/X1X2_ETO/X1X2_ETO.py --> X1X2_policy/X1X2_ETO/regret_all_Wrong_ETO.csv & regret_all_Correct_ETO.csv --> Result_processing_example_code.R --> X1X2_policy/X1X2_ETO/Regret_ETO.csv --> X1X2_policy/X1X2_ETO/Regret_ETO.xlsx

● SPO+ DM & SPO+ DR PI & SPO+ DR Lambda & SPO+ DR Clip

X1X2_policy/X1X2_SPO+DM_DR_PI_Lamdba_Clip/X1X2_SPO+DM_DR_PI_Lamdba_Clip.py --> X1X2_policy/X1X2_SPO+DM_DR_PI_Lamdba_Clip/regret_no_kernel_Direct_Model.csv & regret_no_kernel_Sample_Double_Robust_tree.csv & regret_no_kernel_Lambda_Sample_Double_Robust_tree.csv & regret_no_kernel_Clip_Sample_Double_Robust_tree.csv --> Result_processing_example_code.R --> X1X2_policy/X1X2_SPO+DM_DR_PI_Lamdba_Clip/X1X2_Nuisance_model_well_specified_result.csv --> X1X2_policy/X1X2_SPO+DM_DR_PI_Lamdba_Clip/X1X2_Nuisance_model_well_specified_result.xlsx

● SPO+ ISW

X1X2_policy/X1X2_ISW/X1X2_ISW.py --> X1X2_policy/X1X2_ISW/regret_no_kernel_ISW_Sample.csv --> Result_processing_example_code.R --> X1X2_policy/X1X2_ETO/X1X2_ISW_result.csv --> X1X2_policy/X1X2_ETO/X1X2_ISW_result.xlsx

● Naive ETO

X1X2_policy/X1X2_Naive_ETO/X1X2_Naive_ETO.py --> X1X2_policy/X1X2_Naive_ETO/regret_all_Correct_ETO.csv & regret_all_Wrong_ETO.csv --> Result_processing_example_code.R --> X1X2_policy/X1X2_Naive_ETO/Regret_ETO.csv --> X1X2_policy/X1X2_Naive_ETO/Regret_ETO.xlsx

● Naive SPO+ DM

X1X2_policy/X1X2_Naive_SPO+DM/X1X2_Naive_SPO+DM.py --> X1X2_policy/X1X2_Naive_SPO+DM/regret_no_kernel_CDM & regret_no_kernel_Wrong_CDM --> Result_processing_example_code.R --> X1X2_policy/X1X2_Naive_SPO+DM/X1X2_Naive_SPO+DM_result.csv --> X1X2_policy/X1X2_Naive_SPO+DM/X1X2_Naive_SPO+DM_result.xlsx

● Naive SPO+ DR

X1X2_policy/X1X2_Naive_SPO+DR/X1X2_Naive_SPO+DR.py --> X1X2_policy/X1X2_Naive_SPO+DR/regret_no_kernel_CDR & regret_no_kernel_Wrong_CDR --> Result_processing_example_code.R --> X1X2_policy/X1X2_Naive_SPO+DR/X1X2_Naive_SPO+DR_result.csv --> X1X2_policy/X1X2_Naive_SPO+DR/X1X2_Naive_SPO+DR_result.xlsx

● Naive SPO+ IPW

X1X2_policy/X1X2_Naive_SPO+IPW/X1X2_Naive_SPO+IPW.py --> X1X2_policy/X1X2_Naive_SPO+IPW/regret_no_kernel_IPW.csv --> Result_processing_example_code.R --> X1X2_policy/X1X2_Naive_SPO+IPW/X1X2_Naive_SPO+IPW_result.csv --> X1X2_policy/X1X2_Naive_SPO+IPW/X1X2_Naive_SPO+IPW_result.xlsx

### Table 8

● ETO (misspecified degree 2)

X1X2_policy/X1X2_ETO_misspecified2/X1X2_ETO_misspecified2.py --> X1X2_policy/X1X2_ETO_misspecified2/regret_all_Wrong_ETO.csv & regret_all_Correct_ETO.csv --> Result_processing_example_code.R --> X1X2_policy/X1X2_ETO_misspecified2/Regret_ETO.csv --> X1X2_policy/X1X2_ETO_misspecified2/Regret_ETO.xlsx

● ETO (misspecified degree 4)

X1X2_policy/X1X2_ETO/X1X2_ETO.py --> X1X2_policy/X1X2_ETO/regret_all_Wrong_ETO.csv & regret_all_Correct_ETO.csv --> Result_processing_example_code.R --> X1X2_policy/X1X2_ETO/Regret_ETO.csv --> X1X2_policy/X1X2_ETO/Regret_ETO.xlsx

● SPO+ DM & SPO+ DR PI & SPO+ DR Lambda & SPO+ DR Clip 

1. Nuisance Model $F^{\text{N}}$ is well-specified and Policy-inducing model $F$ is misspecified degree 4

X1X2_policy/X1X2_SPO+DM_DR_PI_Lamdba_Clip/X1X2_SPO+DM_DR_PI_Lamdba_Clip.py --> X1X2_policy/X1X2_SPO+DM_DR_PI_Lamdba_Clip/regret_no_kernel_Direct_Model_tree.csv & regret_no_kernel_Sample_Double_Robust_tree.csv & regret_no_kernel_Lambda_Sample_Double_Robust_tree.csv & regret_no_kernel_Clip_Sample_Double_Robust_tree.csv --> Result_processing_example_code.R --> X1X2_policy/X1X2_SPO+DM_DR_PI_Lamdba_Clip/X1X2_Nuisance_model_well_specified_result.csv --> X1X2_policy/X1X2_SPO+DM_DR_PI_Lamdba_Clip/X1X2_Nuisance_model_well_specified_result.xlsx

● SPO+ DM & SPO+ DR PI & SPO+ DR Lambda & SPO+ DR Clip 

1. Nuisance Model $F^{\text{N}}$ is misspecified degree 4, Policy-inducing model $F$ is well-specified 
2. Nuisance Model $F^{\text{N}}$, Nuisance Model $F^{\text{N}}$ are both misspecified degree 4

X1X2_policy/X1X2_SPO+DM_DR_PI_Lamdba_Clip/X1X2_SPO+DM_DR_PI_Lamdba_Clip.py --> X1X2_policy/X1X2_SPO+DM_DR_PI_Lamdba_Clip/regret_no_kernel_Linear_Wrong_Direct_Model.csv &  regret_no_kernel_Linear_Wrong_Double_Robust_Tree_Sigma.csv & regret_no_kernel_Lambda_Linear_Wrong_Double_Robust_Tree_Sigma.csv & regret_no_kernel_Clip_Linear_Wrong_Double_Robust_Tree_Sigma.csv --> Result_processing_example_code.R --> X1X2_policy/X1X2_SPO+DM_DR_PI_Lamdba_Clip/X1X2_Nuisance_model_misspecified_result.csv --> X1X2_policy/X1X2_SPO+DM_DR_PI_Lamdba_Clip/X1X2_Nuisance_model_misspecified_result.xlsx

● SPO+ DM & SPO+ DR PI & SPO+ DR Lambda & SPO+ DR Clip 

1. Nuisance Model $F^{\text{N}}$ is well-specified and Policy-inducing model $F$ is misspecified degree 2

X1X2_policy/X1X2_SPO+DM_DR_PI_Lamdba_Clip_misspecified2/X1X2_SPO+DM_DR_PI_Lamdba_Clip_misspecified2.py --> X1X2_policy/X1X2_SPO+DM_DR_PI_Lamdba_Clip_misspecified2/regret_no_kernel_Direct_Model_tree.csv & regret_no_kernel_Sample_Double_Robust_tree.csv & regret_no_kernel_Lambda_Sample_Double_Robust_tree.csv & regret_no_kernel_Clip_Sample_Double_Robust_tree.csv --> Result_processing_example_code.R --> X1X2_policy/X1X2_SPO+DM_DR_PI_Lamdba_Clip_misspecified2/X1X2_Nuisance_model_well_specified_result.csv --> X1X2_policy/X1X2_SPO+DM_DR_PI_Lamdba_Clip_misspecified2/X1X2_Nuisance_model_well_specified_result.xlsx

● SPO+ DM & SPO+ DR PI & SPO+ DR Lambda & SPO+ DR Clip 

1. Nuisance Model $F^{\text{N}}$ is misspecified degree 2, Policy-inducing model $F$ is well-specified 
2. Nuisance Model $F^{\text{N}}$, Nuisance Model $F^{\text{N}}$ are both misspecified degree 2

X1X2_policy/X1X2_SPO+DM_DR_PI_Lamdba_Clip_misspecified2/X1X2_SPO+DM_DR_PI_Lamdba_Clip_misspecified2.py --> X1X2_policy/X1X2_SPO+DM_DR_PI_Lamdba_Clip_misspecified2/regret_no_kernel_Linear_Wrong_Direct_Model.csv &  regret_no_kernel_Linear_Wrong_Double_Robust_Tree_Sigma.csv & regret_no_kernel_Lambda_Linear_Wrong_Double_Robust_Tree_Sigma.csv & regret_no_kernel_Clip_Linear_Wrong_Double_Robust_Tree_Sigma.csv --> Result_processing_example_code.R --> X1X2_policy/X1X2_SPO+DM_DR_PI_Lamdba_Clip_misspecified2/X1X2_Nuisance_model_misspecified2_result.csv --> X1X2_policy/X1X2_SPO+DM_DR_PI_Lamdba_Clip_misspecified2/X1X2_Nuisance_model_misspecified2_result.xlsx

● Naive SPO+ DM 

1. Nuisance Model $F^{\text{N}}$ is well-specified, Policy-inducing model $F$ is misspecified degree 4
2. $F^{\text{N}}$ is misspecified degree 4, Policy-inducing model $F$ is well-specified
3. Nuisance Model $F^{\text{N}}$, Nuisance Model $F^{\text{N}}$ are both misspecified degree 4

X1X2_policy/X1X2_Naive_SPO+DM/X1X2_Naive_SPO+DM.py --> X1X2_policy/X1X2_Naive_SPO+DM/regret_no_kernel_CDM & regret_no_kernel_Wrong_CDM --> Result_processing_example_code.R --> X1X2_policy/X1X2_Naive_SPO+DM/X1X2_Naive_SPO+DM_result.csv --> X1X2_policy/X1X2_Naive_SPO+DM/X1X2_Naive_SPO+DM_result.xlsx

● Naive SPO+ DM 

1. Nuisance Model $F^{\text{N}}$ is well-specified, Policy-inducing model $F$ is misspecified degree 2
2. $F^{\text{N}}$ is misspecified degree 2, Policy-inducing model $F$ is well-specified
3. Nuisance Model $F^{\text{N}}$, Nuisance Model $F^{\text{N}}$ are both misspecified degree 2

X1X2_policy/X1X2_Naive_SPO+DM_misspecified2/X1X2_Naive_SPO+DM_misspecified2.py --> X1X2_policy/X1X2_Naive_SPO+DM_misspecified2/regret_no_kernel_CDM & regret_no_kernel_Wrong_CDM --> Result_processing_example_code.R --> X1X2_policy/X1X2_Naive_SPO+DM_misspecified2/X1X2_Naive_SPO+DM_misspecified2_result.csv --> X1X2_policy/X1X2_Naive_SPO+DM_misspecified2/X1X2_Naive_SPO+DM_misspecified2_result.xlsx

● Naive SPO+ DR 

1. Nuisance Model $F^{\text{N}}$ is well-specified, Policy-inducing model $F$ is misspecified degree 4
2. $F^{\text{N}}$ is misspecified degree 4, Policy-inducing model $F$ is well-specified
3. Nuisance Model $F^{\text{N}}$, Nuisance Model $F^{\text{N}}$ are both misspecified degree 4

X1X2_policy/X1X2_Naive_SPO+DR/X1X2_Naive_SPO+DR.py --> X1X2_policy/X1X2_Naive_SPO+DR/regret_no_kernel_CDR & regret_no_kernel_Wrong_CDR --> Result_processing_example_code.R --> X1X2_policy/X1X2_Naive_SPO+DR/X1X2_Naive_SPO+DR_result.csv --> X1X2_policy/X1X2_Naive_SPO+DR/X1X2_Naive_SPO+DR_result.xlsx

● Naive SPO+ DR 

1. Nuisance Model $F^{\text{N}}$ is well-specified, Policy-inducing model $F$ is misspecified degree 2
2. $F^{\text{N}}$ is misspecified degree 2, Policy-inducing model $F$ is well-specified
3. Nuisance Model $F^{\text{N}}$, Nuisance Model $F^{\text{N}}$ are both misspecified degree 2

X1X2_policy/X1X2_Naive_SPO+DR_misspecified2/X1X2_Naive_SPO+DR_misspecified2.py --> X1X2_policy/X1X2_Naive_SPO+DR_misspecified2/regret_no_kernel_CDR & regret_no_kernel_Wrong_CDR --> Result_processing_example_code.R --> X1X2_policy/X1X2_Naive_SPO+DR_misspecified2/X1X2_Naive_SPO+DR_misspecified2_result.csv --> X1X2_policy/X1X2_Naive_SPO+DR_misspecified2/X1X2_Naive_SPO+DR_misspecified2_result.xlsx

## Dependencies

### Python 3.8.8

● gurobipy 11.0.0

● joblib 1.0.1

● numpy 1.20.1

● scikit-learn 0.24.1

### R 3.6.3

● tidyverse 1.3.1
