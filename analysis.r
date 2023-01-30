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
library(stargazer) # for latex tables of model summaries
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

alsla <- read.csv("ALSLA-results.csv", header = TRUE)

l1 <- read.csv("l1-results.csv", header = FALSE)
nrow(l1) # 8934 responses
# Data Wrangling -------------------------------------------------------------#

# select columns of interest

df <- select(alsla, c("ID","l1", "item", "itemType", "collType", "RT", ))

# keep only EN from the l1 column
df <- df %>% filter(l1 == "EN")

n_unique(df$ID) # 99 participants
hist(df$RT) # visual check, all good

# group by item and collType and get a count
stimuli_df <- df %>% group_by(item, collType) %>% summarise(n = n())

head(stimuli_df)


# remove seeds that did not complete the experiment
# (i.e. less than 90 responses to items)
colnames(l1) <- c("X", "lang", "id", "seed", "item", "tensor", "rt")

l1 <- l1 %>%
  group_by(id) %>%
  filter(n() == 90) %>%
  ungroup()

 nrow(l1)



colnames(l1) <- c("X", "lang", "id", "seed", "item", "tensor", "rt")
skim(l1)
n_unique(l1$id) # 99 simulations
hist(l1$rt) # visual check: lots of time-outs that we deleted from alsla exp

# count if rt = 450
sum(l1$rt == 450) # 940 responses are "time-outs"

# remove time-outs
# l1 <- l1 %>% filter(rt != 450)

# rescale simulated rts to match alsla exp
l1$rescaled_rt <- l1$rt*10
hist(l1$rescaled_rt) # visual check, all good

# eliminate rts below 450ms

#l1 <- l1 %>% filter(rescaled_rt > 450)
hist(l1$rescaled_rt) # visual check, all good
nrow(l1) # 7986 responses

# manipulate srtings in item column to have a space instead of a .
l1$item <- gsub("\\.", " ", l1$item)

# add itemType from stimuli_df to l1 by matching to item column

l1$collType <- stimuli_df$collType[match(l1$item, stimuli_df$item)]


barplot <- ggbarplot(l1, x = "collType", y = "rescaled_rt", 
                     add = "mean_se", fill = "collType", palette = "jco",
                     xlab = "Collocation Type", ylab = "Response Time (ms)",
                     ggtheme = theme_bw())
barplot

#-----------------------------------------------------------------------------#

#------------------------- EXPERIMENT ONE -------------------------------------#

skim(l1)

## Data Wrangling -------------------------------------------------------------


