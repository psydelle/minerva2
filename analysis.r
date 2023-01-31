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
#library(afex) 
#library(sjPlot) 
#library(lme4)
#library(rstatix) 

# set seed for reproducibility

set.seed(123)

# set options

options(ggpubr.theme = "bw", digits = 3) 

# load data
stimuli <- read.csv("stimuli.csv", header = TRUE)
nrow(stimuli) # 90 items
alsla <- read.csv("ALSLA-results.csv", header = TRUE)
nrow(alsla) # 28802 responses
l1 <- read.csv("l1-sim-results.csv", header = TRUE)
nrow(l1) # 8911 responses

## Data Wrangling ------------------------------------------------------------#

## ALSLA Dataset: select columns of interest

df <- select(alsla, c("ID", "l1", "item", "itemType", "collType", "RT", ))

# keep only EN from the l1 column
df <- df %>% filter(l1 == "EN")

n_unique(df$ID) # 99 participants
hist(df$RT) # visual check, all good

# group by item and collType and get a count
stimuli_df <- df %>% group_by(item, collType) %>% summarise(n = n())


## l1 simulation dataset: add collType column

# manipulate strings in item column to have a space instead of a .
l1$item <- gsub("\\.", " ", l1$item)

# add collType from stimuli_df to l1 by matching to item column
l1$collType <- stimuli_df$collType[match(l1$item, stimuli_df$item)]
view(l1)

l1$collType <- factor(l1$collType)

levels(l1$collType) # check levels

#rename factor levels
levels(l1$collType) <- c("Baseline", "Congruent", "Incongruent", "Productive")

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

hist(l1$rescaled_rt) # visual check, all good



## Summary Statistics --------------------------------------------------------#

# barplot with t test
barplot <- ggbarplot(l1 %>% filter(collType != "Baseline"), 
                     x = "collType", y = "rescaled_rt",
                     title = "Mean simulated response time by collocation type (n=99)",
                     add = "mean_se", fill = "collType",
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
barplot



#-----------------------------------------------------------------------------#

#------------------------- EXPERIMENT ONE -------------------------------------#


## Data Wrangling -------------------------------------------------------------


