library(tidyverse)
rep_num = 50

res1 = read_csv("regret_all_Correct_ETO.csv")
colnames(res1)[2] = "ETO_correct_Direct_Model"
res1 = res1 %>% pivot_longer(c("ETO_correct_Direct_Model"), names_to = "methods", values_to = "regret")
res_summary1 = res1 %>% group_by(n, methods) %>% summarise(avg_regret = mean(regret), std = sd(regret))

res2 = read_csv("regret_all_Wrong_ETO.csv")
colnames(res2)[2] = "ETO_wrong_Direct_Mode"
res2 = res2 %>% pivot_longer(c("ETO_wrong_Direct_Mode"), names_to = "methods", values_to = "regret")
res_summary2 = res2 %>% group_by(n, methods) %>% summarise(avg_regret = mean(regret), std = sd(regret))

res_summary = rbind(res_summary1, res_summary2)

table(res_summary$methods)
res_summary$method_type = NA
res_summary$method_type[grepl("ETO_correct_Direct_Model", res_summary$methods, fixed=T)] = "ETO Direct Model"
res_summary$method_type[grepl("ETO_wrong_Direct_Mode", res_summary$methods, fixed=T)] = "ETO Direct Model"

res_summary$method_type = factor(res_summary$method_type, levels = c("ETO Direct Model"))

res_summary$setting = NA
res_summary$setting[grepl("correct", res_summary$methods)] = "Correct linear"
res_summary$setting[grepl("wrong", res_summary$methods)] = "Wrong linear"
res_summary$setting = factor(res_summary$setting, levels = c("Correct linear",
                                                             "Wrong linear"))
res_summary = res_summary %>% mutate(lb = avg_regret - 1.96*std/sqrt(rep_num), ub = avg_regret + 1.96*std/sqrt(rep_num))
write.csv(res_summary, "Regret_ETO.csv", row.names= FALSE )
