## DOCUMENT DETAILS ----------------------------------------------------------

# Project: CDT in NLP Individual Project
# Working Title: Investigating Collocational Processing with Minerva2
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
library(performance)

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
#library(rstatix) 

# set seed for reproducibility

set.seed(123)

# set options

options(ggpubr.theme = "bw", digits = 3)

# load data
stimuli <- read.csv("stimuli.csv", header = TRUE)
alsla <- read.csv("ALSLA-results.csv", header = TRUE)
alsla$l1 <- ifelse(alsla$l1 == "EN", "English", "Portuguese")

## Data Wrangling ------------------------------------------------------------#

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

# barplots with bootstrapped confidence intervals
ci_fun <- function(y) {
  boot_ci <- boot(y, statistic = mean, R = 1000)
  return(boot.ci(boot_ci, index = 1)$bca[4:5])
}

ci <- tapply(alsla$RT, alsla$collType, ci_fun)
ci_df <- data.frame(x = names(ci), lower = ci[,1], upper = ci[,2])

alsla_plot <-  ggbarplot(alsla, 
                     x = "collType", y = "RT",
                     facet.by = "l1",
                     #title = "Mean response time by collocation type for L1 English (n=99)",
                     add = "mean_ci", 
                     fill = "collType",
                     #facet.by = "item",
                     xlab = "Experimental Condition",
                     ylab = "Response Time (ms)",
                     ggtheme = theme_bw(),
                     label = TRUE,
                     lab.vjust = -4,
                     font.title = c(22, "bold"),
                     font.x = c("22", "bold"),
                     font.y = c("22", "bold"),
                     palette = c("#005876", "#D50032",
                                  "#EBA70E", "#81B920",
                                  "#00AFBB", "#4A7875",
                                  "#E7B800"),
                     panel.labs.background = list(color = "black",
                                                  fill = "white"),
                          panel.labs.font = list(size = 20, 
                                                  face = "bold"),) +
                    font("xy.text", size = 15) + 
                    theme(legend.position = "none")
alsla_plot

m1_l1 <- mixed(RT ~ collType + age + 
              (1 + collType | ID) + 
              (1 | item), 
              data = alsla %>% filter(l1 == "English"),
              family = inverse.gaussian(link = "identity"), method = "LRT")
m1_l1
summary(m1_l1)


m1_l2 <- mixed(RT ~ collType + age + 
              (1 + collType | ID) + 
              (1 | item), 
              data = alsla %>% filter(l1 != "English"),
              family = inverse.gaussian(link = "identity"), method = "LRT")
m1_l2
summary(m1_l2)

# summary statistics for l1

alsla %>% group_by(l1) %>% summarise(mean = mean(RT), sd = sd(RT), n = n())
# group by item and collType and get a count
stimuli_df <- alsla %>% group_by(item, collType) %>% summarise(n = n())
head(stimuli_df)

## l1 simulation dataset: add collType column

# manipulate strings in item column to have a space instead of a .
l1$item <- gsub("\\.", " ", l1$item)

# add collType from stimuli_df to l1 by matching to item column
l1$collType <- stimuli_df$collType[match(l1$item, stimuli_df$item)]
l1$fcoll <- stimuli_df$collType[match(l1$item, stimuli_df$item)]
head(l1)

l1$collType <- factor(l1$collType)

levels(l1$collType) # check levels

#rename factor levels
levels(l1$collType) <- c("Baseline", "Congruent", "Incongruent", "Productive")
head(l1)
#relevel factors

l1$collType <- relevel(l1$collType, ref = "Productive")
# check to see if there are less than 90 responses to any items

l1 <- l1 %>%
  group_by(id) %>%
  filter(n() == 90) %>%
  ungroup()

 nrow(l1) # alles gut

n_unique(l1$id) # 99 simulations

hist(l1$rt) # visual check, all good

# count if rt = 450
sum(l1$rt == 450) # 178 responses are "time-outs"

# remove time-outs
# l1 <- l1 %>% filter(rt != 450)

# rescale simulated rts to match alsla exp
l1$rescaled_rt <- l1$rt*10

## Summary Statistics --------------------------------------------------------#

# barplot with bootstrap confidence intervals

l1barplot <- ggbarplot(l1, x = "collType", y = "rescaled_rt",
                     title = "Mean simulated response time by collocation type for L1 English (n=99)",
                     add = "mean_ci", 
                     fill = "collType",
                     #facet.by = "item",
                     xlab = "Collocation Type",
                     ylab = "Response Time (ms)",
                     ggtheme = theme_bw(),
                     label = TRUE,
                     lab.vjust = -4,
                     font.title = c(22, "bold"),
                     font.x = c("22", "bold"),
                     font.y = c("22", "bold"),
                     palette = c("#005876", "#D50032",
                                  "#EBA70E", "#81B920",
                                  "#00AFBB", "#4A7875",
                                  "#E7B800")) +
                    font("xy.text", size = 15) +
                    stat_summary(fun.data = mean_cl_boot(), 
                    geom = "errorbar", width = 0.2, size = 1.5, 
                    color = "black")
l1barplot

l2barplot <- ggbarplot(l2, 
                     x = "collType", y = "rescaled_rt",
                     title = "Mean simulated response time by collocation type (n=99)",
                     add = "mean_ci", fill = "collType",
                     #facet.by = "item",
                     xlab = "Collocation Type",
                     ylab = "Response Time (ms)",
                     ggtheme = theme_bw(),
                     label = TRUE,
                     lab.vjust = -4,
                     font.title = c(22, "bold"),
                     font.x = c("22", "bold"),
                     font.y = c("22", "bold"),
                     palette = c("#005876", "#D50032",
                                  "#EBA70E", "#81B920",
                                  "#00AFBB", "#4A7875",
                                  "#E7B800")) +
                    font("xy.text", size = 15)

#-----------------------------------------------------------------------------#

#------------------------- EXPERIMENT ONE -------------------------------------#


## Data Wrangling -------------------------------------------------------------


m1 <- glmer(rescaled_rt ~ collType + (1 | id) + (1 | item), 
            data = l1 %>% filter(collType != "Baseline"), 
            family = inverse.gaussian(link = "identity"), nAGQ = 0,
                    control=glmerControl(optimizer ="bobyqa"))
m1

summary(m1)
m1.glmer <- glmer(rescaled_rt ~ collType + (1 | id) + (1 | item), data = l1 %>% filter(collType != "Baseline"), family = gaussian)
summary(m1)
ranef(m1)

plot(m1)

 plot_model(m1)


m2 <- lmer(rescaled_rt ~ collType + fcoll + (1 | item), data = l1)
summary(m2)
colnames(l1)
