## DOCUMENT DETAILS ----------------------------------------------------------

# Project: CDT in NLP Individual Project
# Document Title: Stats for Simulations
# Author: Sydelle de Souza
# Institution: University of Edinburgh
# Supervisors: Dr Frank Mollica and Dr Alex Doumas
# Date: 2022/12/21
# Python version: 3.9.12

#-----------------------------------------------------------------------------#

## COMMENTS -------------------------------------------------------------------

# this file contains the code for analysing the results of the experiment. we
# start with experiment one which simulates L1 processing.

#-----------------------------------------------------------------------------#

## ACKNOWLEDGEMENTS  ----------------------------------------------------------

#
#

#-----------------------------------------------------------------------------#

## Set-Up ---------------------------------------------------------------------

# set current working directory to the directory of this file

setwd(dirname(sys.frame(1)$ofile))

getwd()
#install.packages(c("tidyverse",
#                    "skimr",
#                    "ggpubr",
#                    "pander",
#                    "xtable",
#                    "patchwork",
#                    "car",
#                    "afex",
#                    "sjPlot",
#                    "lme4",
#                    "boot",
#                    "emmeans"))


# load packages

library(tidyverse) # data wrangling
library(skimr) # summary statistics
library(ggpubr) # for publication-ready plots
library(pander) # for publication-ready tables
library(xtable) # for latex tables
library(patchwork) # for combining plots
library(car) # for anovas and other stats
library(afex) # for mixed effects models
library(sjPlot) # for plotting mixed effects models
library(lme4) # for mixed effects models
library(boot) # for bootstrapping
library(emmeans) # for post-hoc tests
library(BMS) # for Bayes factors and Bayesian model averaging


# set seed for reproducibility

set.seed(17890)

# set options

options(digits = 3)
emm_options(pbkrtest.limit = 24125) # for emmeans

# loading three datasets
stimuli <- read.csv("data\\stimuli.csv", header = TRUE) # stimuli dataset
nrow(stimuli) # 90

human <- read.csv("data\\ALSLA-results.csv", header = TRUE) # human experiment results (Souza, 2021)
nrow(human) # 28802

minerva2 <- read.csv("results\\combo_results-stimuli-99p--mix0.6-concat-m2k_0.93-m2mi_300.csv", 
                    header = TRUE) # minerva2 results
nrow(minerva2) # 142560

## Some Data Wrangling ------------------------------------------------------------#

# remove trailing whitespace

human$item <- str_trim(human$item, side = "right")
minerva2$item <- str_trim(minerva2$item, side = "right")

# rename vars for easier plotting
human$l1 <- ifelse(human$l1 == "EN", "L1", "L2")
unique(human$l1)

#rename factor levels
human$collType <- as.factor(human$collType)
unique(human$collType)
levels(human$collType) <- c("Baseline", "Congruent",
                            "Incongruent", "Productive")

# reorder factor levels
human$collType <- factor(human$collType,
                  levels = c("Productive", "Congruent",
                            "Incongruent", "Baseline"))
                             #relevel factors
human$collType <- relevel(human$collType, ref = "Productive")
head(human) # check: looks good!

# group by item and collType and get a count
stimuli_df <- human %>% group_by(item, collType) %>% summarise(n = n())
head(stimuli_df)

# add collType from stimuli_df to minerva2 by matching to item column
minerva2$collType <- stimuli_df$collType[match(minerva2$item,
                                                 stimuli_df$item)]


# rename some cols for simplicity
colnames(minerva2)
names(minerva2)[6] <- "space"
names(minerva2)[7] <- "freq"

unique(minerva2$space)
colnames(minerva2)
minerva2$space <- ifelse(minerva2$space == "en", "EN",
                        ifelse(minerva2$space == "pt", "PT",
                          ifelse(minerva2$space == "en_noise", "Noise (EN)", "Noise (PT)")))
unique(minerva2$space)



minerva2$freq <- ifelse(minerva2$freq == "en", "EN", 
                        ifelse(minerva2$freq == "pt", "PT",
                          ifelse(minerva2$freq == "equal", "Equal", "Mixed 0.6")))

minerva2$condition <- paste(minerva2$space, minerva2$freq, sep = " - ")
unique(minerva2$condition)
## Descriptive Statistics ------------------------------------------------------#

minerva2_desc_bar <- ggbarplot(minerva2,
                     x = "collType", y = "rt",
                     facet.by = c("freq", "space"),
                     #title = "MINERVA2 Simulations: Tau by Collocation Type",
                     add = "mean_se",
                     fill = "collType",
                     xlab = "Experimental Condition",
                     ylab = "Tau",
                     font.legend = c(22, "bold"),
                     legend.title = "Item Type",
                     ggtheme = theme_bw(),
                     #label = TRUE,
                     lab.vjust = -4,
                     font.title = c(22, "bold"),
                     font.x = c("30", "bold"),
                     font.y = c("30", "bold"),
                     palette = c("#F10C66",
                                 "#38AECC",
                                 "#054A91",
                                 "#FFCC00"),
                     panel.labs.background = list(color = "black",
                                                  fill = "white"),
                          panel.labs.font = list(size = 25,
                                                  face = "bold")) +
                    font("xy.text", size = 18) +
                    theme(legend.position = "bottom") + rremove("x.text")
minerva2_desc_bar

ggsave("minerva2_desc_bar_max.png", minerva2_desc_bar, width = 20, height = 20, dpi = 450)




minerva2_desc_bar2 <- ggbarplot(minerva2 %>% 
                    filter(freq != "Equal" & space != "Noise (EN)" & space != "Noise (PT)"),
                     x = "collType", y = "rt",
                     facet.by = c("freq", "space"),
                     title = "MINERVA2",
                     add = "mean_se",
                     fill = "collType",
                     xlab = "Experimental Condition",
                     ylab = "Tau",
                     font.legend = c(22, "bold"),
                     legend.title = "Item Type",
                     ggtheme = theme_bw(),
                     #label = TRUE,
                     lab.vjust = -4,
                     font.title = c(35, "bold"),
                     font.x = c("30", "bold"),
                     font.y = c("30", "bold"),
                     palette = c("#F10C66",
                                 "#38AECC",
                                 "#054A91",
                                 "#FFCC00"),
                                 ncol = 4,
                     panel.labs.background = list(color = "black",
                                                fill = "white"),
                          panel.labs.font = list(size = 35,
                                                face = "bold")) +
                    font("xy.text", size = 22) +
                    theme(legend.position = "bottom") + rremove("x.text")

minerva2_desc_bar2

ggsave("minerva2_exp_results.png", minerva2_desc_bar2, width = 20, height = 20, dpi = 450)

human_results_plot <- ggbarplot(human %>% filter(l1 == "L1"),
                     x = "collType", y = "RT",
                     facet.by = "l1",
                     title = "Human",
                     add = "mean_se",
                     fill = "collType",
                     xlab = "Experimental Condition",
                     ylab = "Reaction Time (ms)",
                     ggtheme = theme_bw(),
                     label = FALSE,
                     lab.vjust = -4,
                     #ylim(500, 1600),
                     font.title = c(35, "bold"),
                     font.x = c("30", "bold"),
                     font.y = c("30", "bold"),
                     font.legend = c(22, "bold"),
                     legend.title = "Item Type:",
                     palette = c("#F10C66",
                                 "#38AECC",
                                 "#054A91",
                                 "#FFCC00"),
                     panel.labs.background = list(color = "black",
                                                  fill = "white"),
                          panel.labs.font = list(size = 35,
                                                  face = "bold")) +
                    font("xy.text", size = 22) +
                    theme(legend.position = "bottom") + rremove("x.text")

human_results_plot

#ggsave("human_desc_bar.png", human_results_plot, width = 20, height = 20, dpi = 450)
ggsave("null_desc_bar.png", minerva2_desc_bar2, width = 20, height = 20, dpi = 450)

arrange <- ggarrange(human_results_plot, minerva2_desc_bar2,
                      ncol = 2, nrow = 1, label.x = "Experiment", 
                      common.legend = TRUE, legend = "bottom",
                      font.label = c(55, "bold"))
arrange
ggsave("l1control_desc_bar.png", arrange, width = 20, height = 20, dpi = 450)



m1_human_l1 <- lmer(RT ~ collType + l1 +
              (1 | ID) + (1 | item),
              data = human %>%
              filter(acc == 1))

summary(m1_human_l1)

emm.m1_l1 <- emmeans(m1_human_l1, ~collType, type = "response")

m1_human_l1 <- lmer(RT ~ collType + l1 +
              (1 | ID) + (1 | item),
              data = human %>% filter(collType != "Incongruent"))

summary(m1_human_l1)



m1_en_en <- lmer(rt ~ collType + 
              (1 | id) + (1 | item), 
              data = minerva2 %>% 
              filter(space == "EN" & freq == "EN" & collType != "Incongruent"))

summary(m1_en_en)

m1_en_pt <- lmer(rt ~ collType + 
              (1 | id) + (1 | item), 
              data = minerva2 %>% 
              filter(space == "EN" & freq == "PT"))

summary(m1_en_pt)

m1_en_mix06 <- lmer(rt ~ collType + 
              (1 | id) + (1 | item), 
              data = minerva2 %>% 
              filter(space == "EN" & freq == "Mixed 0.6"))

summary(m1_en_mix06)

m1_pt_pt <- lmer(rt ~ collType + 
              (1 | id) + (1 | item), 
              data = minerva2 %>% 
              filter(space == "Portuguese" & freq == "Portuguese"))

summary(m1_pt_pt)

m1_pt_en <- lmer(rt ~ collType + 
              (1 | id) + (1 | item), 
              data = minerva2 %>% 
              filter(space == "Portuguese" & freq == "English"))

summary(m1_pt_en)

m1_pt_mix06 <- lmer(rt ~ collType + 
              (1 | id) + (1 | item), 
              data = minerva2 %>% filter(space == "Portuguese" & freq == "Mixed 0.6"))

summary(m1_pt_mix06)

m1_en_al_en <- lmer(rt ~ collType + 
              (1 | id) + (1 | item), 
              data = minerva2 %>% 
              filter(space == "English Aligned" & freq == "English"))

summary(m1_en_al_en)

m1_en_al_pt <- lmer(rt ~ collType + 
              (1 | id) + (1 | item), 
              data = minerva2 %>% 
              filter(space == "English Aligned" & freq == "Portuguese"))

summary(m1_en_al_pt)

m1_en_al_mix06 <- lmer(rt ~ collType + 
              (1 | id) + (1 | item), 
              data = minerva2 %>% 
              filter(space == "English Aligned" & freq == "Mixed 0.6"))

summary(m1_en_al_mix06)


m1_humans <- mixed(RT ~ l1*collType + 
              (1 | ID) + (1 | item), 
              data = human, method = "LRT")
summary(m1_humans)

m1_humans_l1 <- mixed(RT ~ collType +
              (1 | ID) + (1 | item),
              data = human %>% 
              filter(l1 == "English"), method = "LRT")
m1_humans_l1
summary(m1_humans_l1)

m1_humans_l2 <- mixed(RT ~ collType +
              (1 | ID) + (1 | item),
              data = human %>% 
              filter(l1 == "Portuguese"), method = "LRT")
m1_humans_l2
summary(m1_humans_l2)

m1_humans_high <- mixed(RT ~ collType + 
              (1 | ID) + (1 | item), 
              data = human %>% 
              filter(proficiency == "high"))

summary(m1_humans_high)


emm.m1_humans_l1 <- emmeans(m1_humans_l1, ~collType, type = "response") 
pw.emm.m1_l1 <- pairs(emm.m1_humans_l1)
pw.emm.m1_l1

emm.m1_humans_l2 <- emmeans(m1_humans_l2, ~collType, type = "response", lmerTest.limit = 20515) 
pw.emm.m1_l2 <- pairs(emm.m1_humans_l2)
pw.emm.m1_l2

plot.m1_human_l1 <- emmip(m1_humans_l2, ~collType, type = "response", xlab = "Condition", 
                          ylab = "Linear Prediction", lty = 2, 
                          linearg = list(linetype = "solid", size = 1.1, color = "005876"), 
                          dotarg = list(size = 1.5, stroke = 2, color = "005876"))

ggpubr::ggpar(plot.m1_human_l1, legend.title = "Condition", palette = c("#005876", "#D50032",
                                  "#EBA70E", "#81B920")) + theme_bw()


# barplots with bootstrapped confidence intervals

#rename factor levels
minerva2$space <- as.factor(minerva2$space)
minerva2$freq <- as.factor(minerva2$freq)
unique(minerva2$space)
 #relevel factors
minerva2$space <- relevel(minerva2$space, ref = "Portuguese")
minerva2$freq <- relevel(minerva2$freq, ref = "Portuguese")


results_plot <- ggbarplot(minerva2,
                     x = "collType", y = "rt",
                     facet.by = c("freq", "space"),
                     #title = "Simulating mean RTs by Collocation Type",
                     add = "mean_ci",
                     fill = "collType",
                     xlab = "Experimental Condition",
                     ylab = "Tau",
                     font.legend = c(18, "bold"),
                     legend.title = "Item Type",
                     ggtheme = theme_bw(),
                     #label = TRUE,
                     lab.vjust = -4,
                     font.title = c(22, "bold"),
                     font.x = c("30", "bold"),
                     font.y = c("30", "bold"),
                     palette = c("#005876", "#D50032",
                                  "#EBA70E", "#81B920",
                                  "#00AFBB", "#4A7875",
                                  "#E7B800"),
                     panel.labs.background = list(color = "black",
                                                  fill = "white"),
                          panel.labs.font = list(size = 25,
                                                  face = "bold")) +
                    font("xy.text", size = 15) +
                    theme(legend.position = "bottom") + rremove("x.text")
results_plot

ggsave("results_plot.png", results_plot, width = 20, height = 20, dpi = 450)

#rename factor levels
human$collType <- as.factor(human$collType)
human$proficiency <- as.factor(human$proficiency)
levels(human$collType) <- c("Baseline", "Congruent",
                            "Incongruent", "Productive")
levels(human$proficiency) <- c("L1", "High", "Intermediate")
# reorder factor levels
human$collType <- factor(human$collType,
                  levels = c("Productive", "Congruent",
                            "Incongruent", "Baseline"))

human$proficiency <- factor(human$proficiency, levels = c("L1", "High", "Intermediate"))

#relevel factors
human$collType <- relevel(human$collType, ref = "Productive")




human_results_plot <- ggbarplot(human,
                     x = "collType", y = "RT",
                     facet.by = "l1",
                     #title = "Mean RTs by Collocation Type",
                     add = "mean_ci",
                     fill = "collType",
                     xlab = "Experimental Condition",
                     ylab = "Reaction Time (ms)",
                     ggtheme = theme_bw(),
                     label = FALSE,
                     lab.vjust = -4,
                     #ylim(500, 1600),
                     font.title = c(30, "bold"),
                     font.x = c("30", "bold"),
                     font.y = c("30", "bold"),
                     font.legend = c(18, "bold"),
                     legend.title = "Item Type:",
                     palette = c("#005876", "#D50032",
                                  "#EBA70E", "#81B920",
                                  "#00AFBB", "#4A7875",
                                  "#E7B800"),
                     panel.labs.background = list(color = "black",
                                                  fill = "white"),
                          panel.labs.font = list(size = 35,
                                                  face = "bold")) +
                    font("xy.text", size = 22) +
                    theme(legend.position = "bottom") + rremove("x.text") 

human_results_plot

hist(minerva2$rt, breaks = 1000, col = "grey", main = "Reaction Times", xlab = "Reaction Time (ms)", ylab = "Frequency") + theme_bw() + facet_grid(freq ~ space)


sim_plot <- ggviolin(minerva2 %>% filter(space != "English Aligned"), x = "collType", y ="rt",
                     facet.by = c("freq", "space"),
                     #title = "Mean RTs by Collocation Type",
                     add = "mean_ci",
                     fill = "collType",
                     xlab = "Experimental Condition",
                     ylab = "Reaction Time (ms)",
                     ggtheme = theme_bw(),
                     #label = FALSE,
                     lab.vjust = -4,
                     #ylim(500, 1600),
                     font.title = c(30, "bold"),
                     font.x = c("30", "bold"),
                     font.y = c("30", "bold"),
                     font.legend = c(18, "bold"),
                     legend.title = "Item Type:",
                     palette = c("#005876", "#D50032",
                                  "#EBA70E", "#81B920",
                                  "#00AFBB", "#4A7875",
                                  "#E7B800"),
                     panel.labs.background = list(color = "black",
                                                  fill = "white"),
                          panel.labs.font = list(size = 35,
                                                  face = "bold")) +
                    font("xy.text", size = 22) +
                    theme(legend.position = "bottom") + rremove("x.text") 

sim_plot

ggsave("sim_plot.png", sim_plot, width = 20, height = 20, dpi = 450)

ggsave("human_results_plot.png", human_results_plot, width = 10, height = 10)

# save plot as svg
ggsave("human_results_plot.svg", human_results_plot, width = 8, height = 10)


human_l1 <- filter(human, proficiency == "native")
nrow(human_l1)
human_l2 <- filter(human, proficiency != "native")


by_item_avg_human_l1 <- human_l1 %>% group_by(item) %>% summarise(mean = mean(RT))
head(by_item_avg_human_l1)

by_item_avg_human_l2 <- human_l2 %>% group_by(item) %>% summarise(mean = mean(RT))
head(by_item_avg_human_l2)

by_item_avg_sim_en_en <- en_en %>% group_by(item) %>% summarise(mean = mean(rt))
head(by_item_avg_sim_en_en)

by_item_avg_sim_pt_pt <- pt_pt %>% group_by(item) %>% summarise(mean = mean(rt))
head(by_item_avg_sim_pt_pt)

by_item_avg_sim_en_pt <- en_pt %>% group_by(item) %>% summarise(mean = mean(rt))
head(by_item_avg_sim_en_pt)

by_item_avg_sim_pt_en <- pt_en %>% group_by(item) %>% summarise(mean = mean(rt))
head(by_item_avg_sim_pt_en)

by_item_avg_sim_en_mix06 <- en_mix06 %>% group_by(item) %>% summarise(mean = mean(rt))
head(by_item_avg_sim_en_pt_mix06)

by_item_avg_sim_pt_mix06 <- pt_mix06 %>% group_by(item) %>% summarise(mean = mean(rt))
head(by_item_avg_sim_pt_en_mix06)

by_item_avg_sim_en_al_en <- en_al_en %>% group_by(item) %>% summarise(mean = mean(rt))
head(by_item_avg_sim_en_al_en)

by_item_avg_sim_en_al_pt <- en_al_pt %>% group_by(item) %>% summarise(mean = mean(rt))
head(by_item_avg_sim_en_al_pt)

by_item_avg_sim_en_al_mix06 <- en_al_mix06 %>% group_by(item) %>% summarise(mean = mean(rt))
head(by_item_avg_sim_en_al_mix06)


corr_test <- function(df1, df2) {
  cor.test(df1$mean, df2$mean, method = "spearman")
}

corr_test(by_item_avg_human_l1, by_item_avg_sim_en_en)

corr_test(by_item_avg_human_l2, by_item_avg_sim_pt_pt)

corr_test(by_item_avg_human_l2, by_item_avg_sim_en_pt)

corr_test(by_item_avg_human_l2, by_item_avg_sim_pt_en)

corr_test(by_item_avg_human_l2, by_item_avg_sim_en_mix06)

corr_test(by_item_avg_human_l2, by_item_avg_sim_pt_mix06)

corr_test(by_item_avg_human_l2, by_item_avg_sim_en_al_en)

corr_test(by_item_avg_human_l2, by_item_avg_sim_en_al_pt)

corr_test(by_item_avg_human_l2, by_item_avg_sim_en_al_mix06)


ggscatter()