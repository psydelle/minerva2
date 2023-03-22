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

# set current working directory

setwd(dirname(sys.frame(1)$ofile))

# load packages

library(tidyverse) # data wrangling
library(skimr) # summary statistics
library(ggpubr) # for publication-ready plots
library(pander) # for publication-ready tables
library(xtable) # for latex tables
library(patchwork) # for combining plots
library(car) # for anovas and other stats
library(afex) 
library(sjPlot) 
library(lme4)
library(boot) # for bootstrapping
library(emmeans) 

# set seed for reproducibility

set.seed(123)

# set options

options(digits = 3)

# load data
stimuli <- read.csv("stimuli.csv", header = TRUE)
alsla <- read.csv("ALSLA-results.csv", header = TRUE)

en_en <- read.csv("results\\l1-results-stimuli-lang_en-freq_en-concat.csv", 
                    header = TRUE)
en_en$space <- "English"
en_en$freq <- "English"
nrow(en_en)

en_pt <- read.csv("results\\l1-results-stimuli-lang_en-freq_pt-concat.csv", 
                    header = TRUE)
en_pt$space <- "English"
en_pt$freq <- "Portuguese"
nrow(en_pt)

en_mix06 <- read.csv("results\\l1-results-stimuli-lang_en-freq_pt-concat.csv", 
                    header = TRUE)
en_mix06$space <- "English"
en_mix06$freq <- "Mixed 0.6"
nrow(en_mix06)

en_al_en <- read.csv("results\\l1-results-stimuli-lang_en_aligned-freq_en-concat.csv", 
                    header = TRUE)
en_al_en$space <- "English Aligned"
en_al_en$freq <- "English"
nrow(en_al_en)

en_al_pt <- read.csv("results\\l1-results-stimuli-lang_en_aligned-freq_pt-concat.csv", 
                    header = TRUE)
en_al_pt$space <- "English Aligned"
en_al_pt$freq <- "Portuguese"
nrow(en_al_pt)

en_al_mix06 <- read.csv("results\\l1-results-stimuli-lang_en_aligned-freq_mix-mix0.6-concat.csv", 
                    header = TRUE)
en_al_mix06$space <- "English Aligned"
en_al_mix06$freq <- "Mixed 0.6"
nrow(en_al_mix06)


pt_pt <- read.csv("results\\l1-results-stimuli-lang_pt-freq_pt-concat.csv", 
                    header = TRUE)
pt_pt$space <- "Portuguese"
pt_pt$freq <- "Portuguese"
nrow(pt_pt)

pt_en <- read.csv("results\\l1-results-stimuli-lang_pt-freq_en-concat.csv", 
                    header = TRUE)
pt_en$space <- "Portuguese"
pt_en$freq <- "English"
nrow(pt_en)

pt_mix06 <- read.csv("results\\l1-results-stimuli-lang_pt-freq_mix-mix0.6-concat.csv", 
                    header = TRUE)
pt_mix06$space <- "Portuguese"
pt_mix06$freq <- "Mixed 0.6"
nrow(pt_mix06)


en_al_whole_en <- read.csv("results\\l1-results-stimuli-lang_en_aligned-freq_en-concat-align_refit_whole.csv", 
                            header = TRUE)
en_al_whole_en$space <- "English Aligned Whole"
en_al_whole_en$freq <- "English"
nrow(en_al_whole_en)

en_al_whole_pt <- read.csv("results\\l1-results-stimuli-lang_en_aligned-freq_pt-concat-align_refit_whole.csv", 
                            header = TRUE)
en_al_whole_pt$space <- "English Aligned Whole"
en_al_whole_pt$freq <- "Portuguese"
nrow(en_al_whole_pt)

en_al_whole_mix06 <- read.csv("results\\l1-results-stimuli-lang_en_aligned-freq_mix-mix0.6-concat-align_refit_whole.csv", 
                            header = TRUE)
en_al_whole_mix06$space <- "English Aligned Whole"
en_al_whole_mix06$freq <- "Mixed 0.6"
nrow(en_al_whole_mix06)



## Data Wrangling ------------------------------------------------------------#

results_df <- rbind(en_en, en_pt, en_mix06, en_al_en, en_al_pt, en_al_mix06, 
                    pt_pt, pt_en, pt_mix06)
head(results_df)
alsla$l1 <- ifelse(alsla$l1 == "EN", "English", "Portuguese")

#rename factor levels
alsla$collType <- as.factor(alsla$collType)
levels(alsla$collType) <- c("Baseline", "Congruent",
                            "Incongruent", "Productive")

# reorder factor levels
alsla$collType <- factor(alsla$collType, 
                  levels = c("Productive", "Congruent", 
                            "Incongruent", "Baseline"))
                             #relevel factors
alsla$collType <- relevel(alsla$collType, ref = "Productive")

# group by item and collType and get a count
stimuli_df <- alsla %>% group_by(item, collType) %>% summarise(n = n())
head(stimuli_df)

# add collType from stimuli_df to en_en by matching to item column
results_df$collType <- stimuli_df$collType[match(results_df$item,
                                                 stimuli_df$item)]


m1_en_en <- lmer(rt ~ collType + 
              (1 | id) + (1 | item), 
              data = results_df %>% 
              filter(space == "English" & freq == "English"))

summary(m1_en_en)

m1_en_pt <- lmer(rt ~ collType + 
              (1 | id) + (1 | item), 
              data = results_df %>% 
              filter(space == "English" & freq == "Portuguese"))

summary(m1_en_pt)

m1_en_mix06 <- lmer(rt ~ collType + 
              (1 | id) + (1 | item), 
              data = results_df %>% 
              filter(space == "English" & freq == "Mixed 0.6"))

summary(m1_en_mix06)

m1_pt_pt <- lmer(rt ~ collType + 
              (1 | id) + (1 | item), 
              data = results_df %>% 
              filter(space == "Portuguese" & freq == "Portuguese"))

summary(m1_pt_pt)

m1_pt_en <- lmer(rt ~ collType + 
              (1 | id) + (1 | item), 
              data = results_df %>% 
              filter(space == "Portuguese" & freq == "English"))

summary(m1_pt_en)

m1_pt_mix06 <- lmer(rt ~ collType + 
              (1 | id) + (1 | item), 
              data = results_df %>% filter(space == "Portuguese" & freq == "Mixed 0.6"))

summary(m1_pt_mix06)

m1_en_al_en <- lmer(rt ~ collType + 
              (1 | id) + (1 | item), 
              data = results_df %>% 
              filter(space == "English Aligned" & freq == "English"))

summary(m1_en_al_en)

m1_en_al_pt <- lmer(rt ~ collType + 
              (1 | id) + (1 | item), 
              data = results_df %>% 
              filter(space == "English Aligned" & freq == "Portuguese"))

summary(m1_en_al_pt)

m1_en_al_mix06 <- lmer(rt ~ collType + 
              (1 | id) + (1 | item), 
              data = results_df %>% 
              filter(space == "English Aligned" & freq == "Mixed 0.6"))

summary(m1_en_al_mix06)


m1_humans <- mixed(RT ~ l1*collType + 
              (1 | ID) + (1 | item), 
              data = alsla, method = "LRT")
summary(m1_humans)

m1_humans_l1 <- mixed(RT ~ collType +
              (1 | ID) + (1 | item),
              data = alsla %>% 
              filter(l1 == "English"), method = "LRT")
m1_humans_l1
summary(m1_humans_l1)

m1_humans_l2 <- mixed(RT ~ collType +
              (1 | ID) + (1 | item),
              data = alsla %>% 
              filter(l1 == "Portuguese"), method = "LRT")
m1_humans_l2
summary(m1_humans_l2)

m1_humans_high <- mixed(RT ~ collType + 
              (1 | ID) + (1 | item), 
              data = alsla %>% 
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
results_df$space <- as.factor(results_df$space)
results_df$freq <- as.factor(results_df$freq)
unique(results_df$space)
 #relevel factors
results_df$space <- relevel(results_df$space, ref = "Portuguese")
results_df$freq <- relevel(results_df$freq, ref = "Portuguese")


results_plot <- ggbarplot(results_df,
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
alsla$collType <- as.factor(alsla$collType)
alsla$proficiency <- as.factor(alsla$proficiency)
levels(alsla$collType) <- c("Baseline", "Congruent",
                            "Incongruent", "Productive")
levels(alsla$proficiency) <- c("L1", "High", "Intermediate")
# reorder factor levels
alsla$collType <- factor(alsla$collType,
                  levels = c("Productive", "Congruent",
                            "Incongruent", "Baseline"))

alsla$proficiency <- factor(alsla$proficiency, levels = c("L1", "High", "Intermediate"))

#relevel factors
alsla$collType <- relevel(alsla$collType, ref = "Productive")




human_results_plot <- ggbarplot(alsla,
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

hist(results_df$rt, breaks = 1000, col = "grey", main = "Reaction Times", xlab = "Reaction Time (ms)", ylab = "Frequency") + theme_bw() + facet_grid(freq ~ space)


sim_plot <- ggviolin(results_df %>% filter(space != "English Aligned"), x = "collType", y ="rt",
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


human_l1 <- filter(alsla, proficiency == "native")
nrow(human_l1)
human_l2 <- filter(alsla, proficiency != "native")


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