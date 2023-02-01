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

en_en <- read.csv("results\\l1-results-stimuli-lang_en-freq_en-concat.csv", 
                    header = TRUE)
en_en$space <- "English"
en_en$freq <- "English"

en_pt <- read.csv("results\\l1-results-stimuli-lang_en-freq_pt-concat.csv", 
                    header = TRUE)
en_pt$space <- "English"
en_pt$freq <- "Portuguese"

en_al_pt <- read.csv("results\\l1-results-stimuli-lang_en_aligned-freq_pt-concat.csv", 
                    header = TRUE)
en_al_pt$space <- "English Aligned"
en_al_pt$freq <- "Portuguese"

pt_en <- read.csv("results\\l1-results-stimuli-lang_pt-freq_en-concat.csv", 
                    header = TRUE)
pt_en$space <- "Portuguese"
pt_en$freq <- "English"

pt_pt <- read.csv("results\\l1-results-stimuli-lang_pt-freq_pt-concat.csv", 
                    header = TRUE)
pt_pt$space <- "Portuguese"
pt_pt$freq <- "Portuguese"


## Data Wrangling ------------------------------------------------------------#

results_df <- rbind(en_en, en_pt, en_al_pt, pt_en, pt_pt)
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


# barplots with bootstrapped confidence intervals

results_plot <-  ggbarplot(results_df, 
                     x = "collType", y = "rt",
                     #facet.by = "space",
                     title = "Simulating mean RTs by Collocation Type",
                     #add = "mean_se", 
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
                                                  face = "bold")) +
                    font("xy.text", size = 20) + facet_grid(rows = vars(freq),
                                                            cols = vars(space))
results_plot



m1_en_en <- lmer(rt ~ collType + 
              (1 | id) + (1 | item), 
              data = en_en)

m1_en_en
summary(m1_en_en)


m1_en_pt <- lmer(rt ~ collType + 
              (1 | id) + (1 | item), 
              data = en_pt)
summary(m1_en_pt)
