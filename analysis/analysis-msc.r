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


# set seed for reproducibility

set.seed(123)

# set options

options(ggpubr.theme = "bw", digits = 3) 

# load data
fdstimuli <- read.csv("FinalDataset.csv", header = TRUE)
nrow(fdstimuli) # 90 items
head(fdstimuli)
alsla <- read.csv("ALSLA-results.csv", header = TRUE)
nrow(alsla) # 28802 responses
fdl1 <- read.csv("l1-results-FinalDataset.csv", header = TRUE)
nrow(fdl1)
colnames(fdl1)
skim(fdl1)
## Data Wrangling ------------------------------------------------------------#

## ALSLA Dataset: select columns of interest

df <- select(alsla, c("ID", "l1", "item", "itemType", "collType", "RT", ))

# keep only EN from the l1 column
df <- df %>% filter(fdl1 == "EN")

n_unique(df$ID) # 99 participants
hist(df$RT) # visual check, all good

# group by item and collType and get a count
fdstimuli_df <- df %>% group_by(item, collType) %>% summarise(n = n())


## l1 simulation dataset: add collType column

# manipulate strings in item column to have a space instead of a .
fdl1$item <- gsub("\\.", " ", fdl1$item)

# add collType from fdstimuli_df to l1 by matching to item column
fdl1$collType <- fdstimuli_df$isCollocation[match(fdl1$item, fdstimuli_df$item)]
head(fdl1)

fdl1$collType <- factor(fdl1$collType)

levels(fdl1$collType) # check levels

#rename factor levels
levels(fdl1$collType) <- c("Baseline", "Congruent", "Incongruent", "Productive")

#relevel factors

fdl1$collType <- relevel(fdl1$collType, ref = "Productive")
# check to see if there are less than 90 responses to any items

fdl1 <- fdl1 %>%
  group_by(id) %>%
  filter(n() == 90) %>%
  ungroup()

 nrow(fdl1) # alles gut

n_unique(fdl1$id) # 99 simulations

hist(fdl1$rt) # visual check, all good

# count if rt = 450
sum(fdl1$rt == 450) # 178 responses are "time-outs"

# remove time-outs
# fdl1 <- fdl1 %>% filter(rt != 450)

# rescale simulated rts to match alsla exp
fdl1$rescaled_rt <- fdl1$rt*10

hist(fdl1$rescaled_rt) # visual check, all good



## Summary Statistics --------------------------------------------------------#

# barplot with t test
barplot <- ggbarplot(fdl1 %>% filter(collType != "Baseline"), 
                     x = "collType", y = "rescaled_rt",
                     title = "Mean simulated response time by collocation type (n=99)",
                     add = "mean_se", fill = "collType",
                     facet.by = "item",
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


