# %%

## Set-Up ---------------------------------------------------------------------
# set current working directory
setwd("./results")

# load packages
library(tidyverse) # data wrangling
library(skimr) # summary statistics
library(xtable) # for latex tables
library(patchwork) # for combining plots
library(lme4) # for mixed effects models
library(brms)
library(fitdistrplus)
library(ggsignif)

# set theme for ggplot
theme_set(theme_bw())

# use cores for parallel processing
options(mc.cores = parallel::detectCores())

# colors
options(ggplot2.discrete.fill = c("#00AFBB", "#E7B800", "#FC4E07", "#646262"))
options(digits = 3)

# set seed for reproducibility
set.seed(0976)

# %%



# %%
## Load Data -------------------------------------------------------------------

# load data
human <- read_csv("experiment-data.csv")
stimuli <- read_csv("..\\data\\stimuli_idioms_clean_annotated1.csv")


# capitalize first letter in column names of stimuli
colnames(stimuli) <- tools::toTitleCase(colnames(stimuli))
colnames(stimuli)

# keep necessary columns
stimuli <- stimuli %>%
    dplyr::select(c(Stimuli_grammatical, Verb, Noun, Fitem, Score, Item))

human <- human %>%
    dplyr::select(-c(dataType, Handedness, Vision, LanguagePathology, foldb))
# rename columns
stimuli <- stimuli %>%
    rename(
        Stimuli = Stimuli_grammatical,
        Frequency = Fitem
    )

colnames(human)

human <- human %>% rename(Stimuli = Item)

# merge data
human <- human %>% left_join(stimuli, by = "Stimuli")
skim(human) # only 89 IDs, should be 90 as verified on prolific

# %%

# %%

# Data Cleaning ----------------------------------------------------------------
# find duplicates
human %>%
    group_by(ID) %>%
    summarise(n = n()) %>%
    filter(n > 164) # one duplicate ID, check fold and remove second entry

dupes <- human %>%
    filter(ID == "607ee4f932bfb9ddf3da6d83") # 2 entries for this ID

unique(dupes$Fold) # in fold 2 and 3, removing from fold 3 and recollect data

human <- human %>%
    filter(!(ID == "607ee4f932bfb9ddf3da6d83" & Fold == 3))

# check for missing data
human %>%
    group_by(ID, Fold) %>%
    summarise(missing = sum(is.na(RT))) %>%
    filter(missing > 10)  # 5 participants with missing data

# 1 ID with 82 missing values from fold 3, remove and recollect data
human <- human %>% filter(ID != "5d7ff8bcb9c215001ce3298d")
# 1 ID with 32 missing values from fold 2, remove and recollect data
human <- human %>% filter(ID != "5e1f1ec9debba10112ac5733")

# boxplots by participant to check for RT outliers
human %>% ggplot(aes(x = ID, y = RT)) +
    geom_boxplot() +
    theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
    facet_wrap(~Fold, ncol = 3, scales = "free")


# count unique IDs per fold
human %>%
    group_by(Fold) %>%
    summarise(n = n_distinct(ID))
# we now have 62 IDs per fold

# final checks
n_obs <- nrow(human)
sprintf("Number of observations: %.0f", n_obs) # 30668
n_participants <- n_distinct(human$ID) # 186
sprintf("Number of participants: %.0f", n_participants)
n_obs_participant <- n_obs / n_participants # 164
sprintf("Number of observations/participant: %.0f", n_obs_participant)

# mean age and sd
mean_age <- mean(human$Age, na.rm = TRUE)
sd_age <- sd(human$Age, na.rm = TRUE)

sprintf("Mean age: %.2f", mean_age)
sprintf("SD age: %.2f", sd_age)

# n unique sex

human %>%
    group_by(Sex) %>%
    summarise(n = n_distinct(ID))

# n unique items
n_unique(human$Item) # 90

human %>% group_by(Condition) %>% summarise(n = n_distinct(Item))
# %%

# %%
## Dealing with Outliers -------------------------------------------------------

# outlier removal
human$outliers <- NA

# start with minimal a priori trimming
# remove too fast RTs and NAs
human$outliers[human$RT < 450] <- "Too fast"
human$outliers[is.na(human$RT)] <- "Missing"
table(human$outliers) # 9 missing values, 33 too fast

# remove too slow RTs 3.5 SD from the mean
mean_rt <- mean(human$RT, na.rm = TRUE)
sd_rt <- sd(human$RT, na.rm = TRUE)
human$outliers[human$RT > (mean_rt + 3.5 * sd_rt)] <- "Too slow"
sum(!is.na(human$outliers))
# percentage of outliers
percentage_outliers <- (sum(!is.na(human$outliers)) / n_obs) * 100
sprintf("Percentage of outliers: %.3f", percentage_outliers)


human <- human %>% filter(is.na(outliers))

n_obs <- nrow(human) # 14704
n_obs

# add accuracy column
human$Accuracy <- ifelse(human$Response == human$Correct, 1, 0)

# participant means
participant_means <- human %>%
    group_by(ID) %>%
    summarise(
        mAccuracy = mean(Accuracy),
        mRT = mean(RT)
    )

# add means to df
human <- human %>% left_join(participant_means, by = "ID")
head(human)
skim(human)

# find IDs with low accuracy
human %>%
    group_by(ID, Fold) %>%
    summarise(mean_accuracy = mean(Accuracy)) %>%
    filter(mean_accuracy < 0.7)

# item means
item_means <- human %>%
    group_by(Item) %>%
    summarise(
        iAccuracy = mean(Accuracy),
        iRT = mean(RT)
    )

# add means_item to df
human <- human %>% left_join(item_means, by = "Item")

# find items with low accuracy
bad_items <- human %>%
    filter(Condition != "Baseline") %>% # exclude baselines
    group_by(Condition, Item, Verb) %>%
    summarise(mean_accuracy = mean(Accuracy)) %>%
    filter(mean_accuracy < 0.7) %>%
    arrange(mean_accuracy, by_group = TRUE)

bad_items

skim(human)

# %%

# %%
# filter out participants with low accuracy
# human <- human %>% filter(mAccuracy >= 0.7)

# # filter out items with low accuracy
# human <- human %>% filter(iAccuracy >= 0.7)

# filter out verbs with low accuracy
human <- human %>% filter(Verb != "silence")
human <- human %>% filter(Verb != "muzzle")
human <- human %>% filter(Verb != "pad")
human <- human %>% filter(Verb != "slap")

n_unique(human$Item)

# %%

# %%
## Plots for Rts

# relevel factors
human$Condition <- relevel(as.factor(human$Condition), ref = "Prod")

human$Condition <- dplyr::recode(human$Condition,
"Prod" = "Productive",
"Collocation" = "Collocation",
"Idiom" = "Idiom",
"Baseline" = "Baseline")


# barplots for RTs by condition
human %>%
    filter(Accuracy == 1) %>% # only correct trials
    ggplot(aes(x = Condition, y = RT, fill = Condition)) +
    geom_bar(stat = "summary", fun = "mean", position = "dodge", color = "black", linewidth = 0.8) +
    geom_errorbar(stat = "summary", fun.data = "mean_cl_boot", position = position_dodge(width = 0.90), width = 0.25, linewidth = 0.8) +
    labs(title = "Human Judgements", y = "Mean RT (ms)", x = "Condition") +
    theme(
        title = element_text(size = 40, face = "bold"),
        axis.line = element_line(colour = "black", linewidth = 0.8, lineend = "round"),
        axis.title = element_text(size = 40, face = "bold"),
        axis.text = element_text(size = 30, face = "bold"),
        legend.title = element_text(size = 35, face = "bold"),
        legend.text = element_text(size = 30, face = "bold"),
        legend.position = "right",
        strip.text.x = element_text(size = 30, face = "bold"),
        strip.text.y = element_text(size = 30, face = "bold"),
        strip.background = element_rect(colour = "black", fill = "white", linewidth = 0.8),
        panel.background = element_rect(colour = "black", fill = "white", linewidth = 0.8),
        panel.grid.major = element_line(colour = "grey", linewidth = 0.5),
        panel.spacing = unit(0, "points"),
        plot.background = element_rect(colour = "white", fill = "white")
    )

ggsave("human_rt_barplot.png", width = 20, height = 15, units = "in")
# boxplots for RTs by condition
human %>%
    filter(Accuracy == 1) %>% # only correct trials
    ggplot(aes(x = Condition, y = RT)) +
    geom_boxplot() +
    labs(y = "RT (ms)", x = "Condition") +
    theme(axis.text.x = element_text(angle = 90, hjust = 1))


# # rename prod to productive
# human$Condition <- ifelse(human$Condition == "Prod", "Productive", human$Condition)

# # relevel factor for mixed effects model
# human$Condition <- relevel(as.factor(human$Condition), ref = "Idiom")

# %%

# %%

# filter out verbs with low accuracy
human <- human %>% filter(Verb != "silence")
human <- human %>% filter(Verb != "muzzle")
human <- human %>% filter(Verb != "pad")
human <- human %>% filter(Verb != "slap")

# null model
glmm_rt_null <- glmer(RT ~ 1 + (1 | ID) + (1 | Verb),
    data = human %>%
        filter(Condition != "Baseline") %>%
        filter(Accuracy == 1),
    family = Gamma(link = "log")
)

summary(glmm_rt_null)

# run a glmm with gamma distribution
glmm_rt <- glmer(RT ~ Condition + (1 | ID) + (1 | Verb),
    data = human %>%
        filter(Condition != "Baseline") %>%
        filter(Accuracy == 1),
    family = Gamma(link = "log")
)

summary(glmm_rt)

glmm_rt_max <- glmer(RT ~ Condition + scale(Frequency) + (1 | ID) + (1 | Verb),
    data = human %>%
         filter(Condition != "Baseline") %>%
        filter(Accuracy == 1),
    family = Gamma(link = "log")
)


summary(glmm_rt_max)
anova(glmm_rt_null, glmm_rt, glmm_rt_max)

# power_glmm <- simr::powerSim(glmm_rt)



# %%


# %%

human %>%
    filter(Condition != "Baseline") %>%
    filter(Response == "y") %>% # only correct trials
    ggplot(aes(x = Condition, y = RT)) +
    geom_boxplot() +
    labs(y = "RT (ms)", x = "Condition") +
    theme(axis.text.x = element_text(angle = 90, hjust = 1))

human %>%
    filter(Condition != "Baseline") %>%
    filter(RT >= 2000) %>% # only correct trials
    ggplot(aes(x = Condition, y = RT)) +
    geom_boxplot() +
    labs(y = "RT (ms)", x = "Condition") +
    theme(axis.text.x = element_text(angle = 90, hjust = 1))

# barplots for RTs by condition
human %>%
    filter(Condition != "Baseline") %>%
    filter(Accuracy == 1) %>% # only correct trials
    ggplot(aes(x = Condition, y = RT)) +
    geom_bar(stat = "summary", fun = "mean") +
    geom_errorbar(stat = "summary", fun.data = mean_cl_boot, width = 0.2) +
    labs(y = "Mean RT (ms)", x = "Condition") +
    theme(axis.text.x = element_text(angle = 90, hjust = 1))


# boxplots for RTs by condition
human %>%
    filter(Condition != "Baseline") %>%
    filter(Accuracy == 1) %>%
    filter(RT < 2000) %>% # only correct trials
    ggplot(aes(x = Condition, y = RT)) +
    geom_boxplot() +
    labs(y = "RT (ms)", x = "Condition") +
    theme(axis.text.x = element_text(angle = 90, hjust = 1))
# %%


# corr between acceptability and fitem

# %% 

# Frequency Stats

human %>%
    group_by(Condition) %>%
    summarise(mean_frequency = mean(Frequency, na.rm = TRUE),
        sd_frequency = sd(Frequency, na.rm = TRUE))

# anova for frequency
aov_frequency <- aov(Frequency ~ Condition, data = human)
summary(aov_frequency)

emmeans::emmeans(aov_frequency, pairwise ~ Condition)

#boxplot for frequency by condition
human %>%
    ggplot(aes(x = Condition, y = log10(Frequency))) +
    geom_boxplot() +
    labs(y = "Frequency", x = "Condition") +
    theme(axis.text.x = element_text(angle = 90, hjust = 1))


