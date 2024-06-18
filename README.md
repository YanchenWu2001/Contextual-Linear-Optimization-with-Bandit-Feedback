# Contextual-Linear-Optimization-with-Bandit-Feedback
Replication code for the paper "Contextual Linear Optimization with Bandit Feedback".

The Random_policy file corresponds to Tables 1 and 2 in the body of the paper, and Tables 3 and 4 in the appendix.

The X1_policy file corresponds to Tables 5 and 6 in the appendix.

The X1X2_policy file corresponds to Tables 7 and 8 in the appendix.

## Details

The meaning of no kernel in the file name is that we do not use the kernel method, the meaning of tree is that we use the decision tree model to learn the policy in X1_policy and X1X2_policy settings, and the meaning of sample is to estimate the policy according to the probability of occurrence of historical data in Random policy setting.

### Table 1 and 3:

● ETO: 

Random_policy/ETO/ETO.py --> Random_policy/ETO/regret_all_Wrong_ETO.csv & regret_all_Correct_ETO.csv --> Result_processing_example_code.R --> Random_policy/ETO/Regret_ETO.csv --> Regret_ETO.xlsx

● SPO+ DM & SPO+ DR PI & SPO+ DR Lambda & SPO+ DR Clip: 

Random_policy/SPO+DM_DR_PI_Lamdba_Clip/SPO+DM_DR_PI_Lamdba_Clip.py --> Random_policy/SPO+DM_DR_PI_Lamdba_Clip/regret_no_kernel_Direct_Model.csv &  regret_no_kernel_Clip_Sample_Double_Robust.csv & 
regret_no_kernel_Lambda_Sample_Double_Robust.csv & regret_no_kernel_Clip_Sample_Double_Robust.csv --> Result_processing_example_code.R --> Random_policy/SPO+DM_DR_PI_Lamdba_Clip/Nuisance_model_well_specified_result.csv --> Random_policy/SPO+DM_DR_PI_Lamdba_Clip/Nuisance_model_well_specified_result.xlsx
