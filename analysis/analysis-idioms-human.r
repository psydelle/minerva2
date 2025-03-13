# %%
getwd()
# load packages
library(tidyverse) # data wrangling
# library(ggpattern) # for pattern fills
library(skimr) # summary statistics
library(xtable) # for latex tables
# library(patchwork) # for combining plots
library(lme4) # for mixed effects models


# set theme for ggplot
theme_set(theme_bw())

# use cores for parallel processing
options(mc.cores = parallel::detectCores())

# colors
options(ggplot2.discrete.fill = c("#387ADF", "#FF4500", "#50C4ED", "#bab9b9"))

# options(ggplot2.discrete.fill = c("#00AFBB", "#E7B800", "#FC4E07", "#646262"))
options(digits = 3)

# set seed for reproducibility
set.seed(0976)

# %%
# load data
human <- read_csv("results\\experiment-data.csv", col_names = TRUE)
stimuli <- read_csv("data\\stimuli_idioms_clean_annotated1.csv", col_names = TRUE)

# capitalize first letter in column names of stimuli
colnames(stimuli) <- tools::toTitleCase(colnames(stimuli))
colnames(stimuli)

# keep necessary columns
stimuli <- stimuli %>%
    dplyr::select(c(Stimuli_grammatical, Verb, Noun, Fitem, Score, Item))

# rename columns
stimuli <- stimuli %>%
    rename(
        Stimuli = Stimuli_grammatical,
        Frequency = Fitem
    )

human <- human %>% rename(Stimuli = Item)

# merge data
human <- human %>% left_join(stimuli, by = "Stimuli")

# remove unnecessary columns
human <- human %>%
    dplyr::select(-c(dataType, Handedness, Vision, LanguagePathology, foldb, Status, PilotComments, Custom_study_tncs_accepted_at, Primary_language, Ethnicity_simplified, Country_of_birth, Country_of_residence, Nationality, Prolific_Age))

# add accuracy column
human$Accuracy <- ifelse(human$Response == human$Correct, 1, 0)

skim(human)
# %%

human$Condition <- dplyr::recode(human$Condition,
    "Prod" = "Compositional",
    "Collocation" = "Collocation",
    "Idiom" = "Idiom",
    "Baseline" = "Baseline"
)

unique(human$Condition)
# relevel factors
human$Condition <- relevel(as.factor(human$Condition), ref = "Compositional")

# reorder factor levels
human$Condition <- factor(human$Condition, levels = c("Compositional", "Collocation", "Idiom", "Baseline"))

# %%

# check for missing data
human %>%
    group_by(ID, Fold) %>%
    summarise(missing = sum(is.na(RT))) %>%
    filter(missing > 10) # 2 participants with missing data

# 1 ID with 82 missing values from fold 3, remove and recollect data
human <- human %>% filter(ID != "5d7ff8bcb9c215001ce3298d")
# 1 ID with 32 missing values from fold 2, remove and recollect data
human <- human %>% filter(ID != "5e1f1ec9debba10112ac5733")


# find duplicates
human %>%
    group_by(ID) %>%
    summarise(n = n()) %>%
    filter(n > 164) # one duplicate ID, check fold and remove second entry

dupes <- human %>%
    filter(ID == "607ee4f932bfb9ddf3da6d83") # multiple entries for this ID

# check for differences
unique(dupes$Fold) # in fold 2 and 3, removing from fold 3 and recollect data

# the same participant took the test twice in fold 2 and 3, this is due to me not
# filtering in Prolific during the second round of data collection. Therefore,
# I will remove the second entry from fold 3.

human <- human %>%
    filter(!(ID == "607ee4f932bfb9ddf3da6d83" & Fold == 3))

# the same participant had their data saved twice in fold 2
# i'm not sure what caused this, but I will remove the second entry
# everything except the time-taken was the same

human <- human %>%
    filter(!(ID == "607ee4f932bfb9ddf3da6d83" & Time_taken == 360))

# check for duplicate rows
human %>% filter(duplicated(human)) # no duplicates

nrow(human) # 30504

# count unique IDs per fold
human %>%
    group_by(Fold) %>%
    summarise(n = n_distinct(ID)) # 62

# count unique items per fold
human %>%
    group_by(Fold) %>%
    filter(Condition != "Baseline") %>%
    summarise(n = n_distinct(Item)) # 82

# count all items per fold
human %>%
    group_by(Fold) %>%
    filter(Condition != "Baseline") %>%
    summarise(n = n()) # 5084

# %%
# final checks
n_obs <- nrow(human)
sprintf("Number of observations: %.0f", n_obs) # 30504
n_participants <- n_distinct(human$ID) # 186
sprintf("Number of participants: %.0f", n_participants)
n_items <- n_distinct(human$Item) # 247
sprintf("Number of items: %.0f", n_items)

n_obs_participant <- n_obs / n_participants # 164
sprintf("Number of observations/participant: %.000f", n_obs_participant)
n_baselines <- nrow(human %>% filter(Condition == "Baseline")) # 15252
sprintf("Number of baselines: %.0f", n_baselines)
n_incorrect <- nrow(human %>% filter(Accuracy == 0)) # 2752
sprintf("Number of incorrect trials: %.0f", n_incorrect)
n_obs_item <- n_obs / n_items # 123
sprintf("Number of observations/item: %.000f", n_obs_item)

# incorrect trials by condition
human %>%
    filter(Accuracy == 0) %>%
    group_by(Condition) %>%
    summarise(n = n())
# incorrect trials by verb
human %>%
    filter(Accuracy == 0) %>%
    group_by(Verb) %>%
    summarise(n = n()) %>%
    arrange(desc(n))

human$RT <- as.numeric(human$RT)


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
n_distinct(human$Item) # 247 (contains NA for all Baselines)

items <- human %>%
    select(Item) %>%
    unique()


# %%

# outlier removal
human$outliers <- NA

# start with minimal a priori trimming
# remove too fast RTs and NAs
human$outliers[human$RT < 450] <- "Too fast"
human$outliers[is.na(human$RT)] <- "Missing"
human$outliers[human$RT >= 8000] <- "Too slow"
table(human$outliers) # 16 missing values, 48 too fast

# remove too slow RTs 3.5 SD from the mean
mean_rt <- mean(human$RT, na.rm = TRUE)
sd_rt <- sd(human$RT, na.rm = TRUE)
human$outliers[human$RT > (mean_rt + 3.5 * sd_rt)] <- "Too slow"
sum(!is.na(human$outliers)) # 454
# percentage of outliers
percentage_outliers <- (sum(!is.na(human$outliers)) / n_obs) * 100
sprintf("Percentage of outliers: %.3f", percentage_outliers)

# filter out outliers
human <- human %>% filter(is.na(outliers))

n_obs <- nrow(human) # 30050
sprintf("Number of observations: %.0f", n_obs)

correct_trials <- human %>% filter(Accuracy == 1)
nrow(correct_trials) # 27429

# participant means
participant_means <- human %>%
    group_by(ID) %>%
    summarise(
        mAccuracy = mean(Accuracy),
        mRT = mean(RT)
    )

# add means to df
human <- human %>% left_join(participant_means, by = "ID")

# find IDs with low accuracy
human %>%
    group_by(ID, Fold) %>%
    summarise(mean_accuracy = mean(Accuracy)) %>%
    filter(mean_accuracy < 0.7)

# there are no participants with accuracy <50%, moving on

# %%
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

# there are 4 items with <50%V accuracy,
# all 4 have denominal verbs
# 3 are in the collocation condition
# 1 is in the idiomatic condition
# we'll run analyses with and without these items
# but we'll report the results without these items

# comment out the following lines to keep these items
# %%
# filter out verbs with low accuracy
human <- human %>% filter(Verb != "silence")
human <- human %>% filter(Verb != "muzzle")
human <- human %>% filter(Verb != "pad")
human <- human %>% filter(Verb != "slap")




# human <- human %>% filter(Verb != "diffuse") # wrong spelling, but exists in corpus
# human <- human %>% filter(Verb != "break") # wrong spelling, but exists in corpus

# filter out baselines
human <- human %>% filter(Condition != "Baseline")
nrow(human) # 14348
human %>%
    filter(Accuracy == 1) %>%
    nrow() # 13369


mean(human$RT, digits = 3) # 1011
sd(human$RT) # 387


# write to csv
write_csv(human, "human_clean_for_acl_withbreakdiffuse.csv")

# %%

# descriptive statistics by condition
human %>%
    group_by(Condition) %>%
    filter(Accuracy == 1) %>% # only correct trials
    summarise(
        mean_RT = mean(RT, na.rm = TRUE, digits = 3),
        sd_RT = sd(RT, na.rm = TRUE),
        min_RT = min(RT, na.rm = TRUE),
        max_RT = max(RT, na.rm = TRUE)
    ) %>%
    xtable(digits = 3)

# descriptive statistics by condition for accuracy
human %>%
    group_by(Condition) %>%
    summarise(
        mean_accuracy = mean(Accuracy, na.rm = TRUE, digits = 3),
        sd_accuracy = sd(Accuracy, na.rm = TRUE),
        min_accuracy = min(Accuracy, na.rm = TRUE),
        max_accuracy = max(Accuracy, na.rm = TRUE)
    ) %>%
    xtable(digits = 3)


rt_summary <- human %>%
    filter(Accuracy == 1) %>%
    summarise(min_RT = min(RT), max_RT = max(RT))


# barplots for RTs by condition
plot <- human %>%
    filter(Accuracy == 1) %>% # only correct trials
    filter(Condition != "Baseline") %>% # exclude baselines
    ggplot(aes(x = Condition, y = RT, fill = Condition)) +
    geom_bar(stat = "summary", fun = "mean", position = "dodge", color = "black", linewidth = 0.8) +
    geom_errorbar(stat = "summary", fun.data = "mean_cl_boot", position = position_dodge(width = 0.90), width = 0.25, linewidth = 0.8) +
    geom_text(
        stat = "summary", fun = mean, aes(label = round(..y.., 1)),
        vjust = -1.3, size = 10, fontface = "bold"
    ) +
    labs(y = "Mean RT (ms)") +
    theme_bw() +
    theme(
        title = element_text(size = 50, face = "bold"),
        axis.line = element_line(colour = "black", linewidth = 0.8, lineend = "round"),
        axis.title = element_text(size = 50, face = "bold"),
        axis.title.x = element_blank(),
        axis.text = element_text(size = 40, face = "bold"),
        # axis.text.x = element_blank(),
        axis.text.x = element_text(angle = 0),
        legend.title = element_blank(),
        legend.text = element_text(size = 45, face = "bold"),
        legend.position = "none",
        strip.text.x = element_text(size = 45, face = "bold"),
        strip.text.y = element_text(size = 45, face = "bold"),
        strip.background = element_rect(colour = "black", fill = "white", linewidth = 0.8),
        panel.background = element_rect(colour = "black", fill = "white", linewidth = 0.8),
        panel.grid.major = element_line(colour = "grey", linewidth = 0.5),
        panel.spacing = unit(0, "points"),
        plot.background = element_rect(colour = "white", fill = "white"),
    )

plot <- plot + coord_cartesian(ylim = c(500, 1100))

plot

ggsave("human_rt_plot.png", plot, width = 15, height = 10, units = "in", dpi = 300)


# corr timeouts

# corr.test(human$iRT, human$Timeouts, method = "kendall")



# %%

# %%


# set reference level for Condition to idiom

human$Condition <- relevel(as.factor(human$Condition), ref = "Idiom")

# # null model
# glmm_rt_null <- glmer(RT ~ 1 + (1 | ID) + (1 | Verb),
#     data = human %>%
#         filter(Condition != "Baseline") %>%
#         filter(Accuracy == 1),
#     family = Gamma(link = "identity")
# )

# summary(glmm_rt_null)

# # run a glmm with gamma distribution
# glmm_rt <- glmer(RT ~ Condition + (1 | ID) + (1 | Verb),
#     data = human %>%
#         filter(Condition != "Baseline") %>%
#         filter(Accuracy == 1),
#     family = Gamma(link = "identity")
# )

# summary(glmm_rt)


glmm_rt_max <- glmer(RT ~ Condition + scale(Frequency) + (1 | ID) + (1 | Verb),
    data = human %>%
        filter(Condition != "Baseline") %>%
        filter(Accuracy == 1),
    family = Gamma(link = "identity")
)

summary(glmm_rt_max)
anova(glmm_rt_null, glmm_rt, glmm_rt_max)

# use stargazer to create a latex table

stargazer::stargazer(glmm_rt_null, glmm_rt, glmm_rt_max, type = "latex", title = "RTs with low acc items", align = TRUE, label = "tab:rt", header = TRUE, digits = 3)


# %%
# accuracy model

# null model

glmer_accuracy_null <- glmer(Accuracy ~ 1 + (1 | ID) + (1 | Verb),
    data = human,
    family = binomial(link = "logit")
)

summary(glmer_accuracy_null)

# run a glmm with binomial distribution
glmer_accuracy <- glmer(Accuracy ~ Condition + (1 | ID) + (1 | Verb),
    data = human,
    family = binomial(link = "logit")
)

summary(glmer_accuracy)

glmer_accuracy_max <- glmer(Accuracy ~ Condition + scale(Frequency) + (1 | ID) + (1 | Verb),
    data = human,
    family = binomial(link = "logit")
)

summary(glmer_accuracy_max)

anova(glmer_accuracy_null, glmer_accuracy, glmer_accuracy_max)

# %%

timeouts <- read_csv("timeout-counts.csv", col_names = TRUE)
timeouts <- timeouts %>% rename(Stimuli = Item)


# corr with taus

taus <- read_csv("tau_by_item_by_experiment.csv", col_names = TRUE)

# add taus to human
human <- human %>% left_join(taus, by = "Item")

head(human)
is.na(human$Mean_Tau.x) %>% sum() # 0


# correlation between RT and tau

fs <- cor.test(human$RT, human$"Frequency & Semantics", method = "kendall")
fonly <- cor.test(human$RT, human$"Frequency-only", method = "kendall")
sonly <- cor.test(human$RT, human$"Semantics-only", method = "kendall")

anova(fs, fonly, sonly)


# correlation between RT and tau (last four columns are the taus, there are 36 columns in total)


# rename column Frequency & Semantics to tau_fs

human <- human %>% rename(tau_fs = "Frequency & Semantics")
human <- human %>% rename(tau_f = "Frequency-only")
human <- human %>% rename(tau_s = "Semantics-only")



cor.test(human$iRT, human$Timeouts, method = "kendall")

# model with irt~timeouts
glmm_rt_timeouts <- lm(iRT ~ 1 + Timeouts,
    data = human
)

summary(glmm_rt_timeouts)

# plot
plot <- ggplot(human, aes(x = Timeouts, y = iRT)) +
    geom_point() +
    geom_smooth(method = "lm", se = FALSE) +
    theme_bw()

rt_tau_model_fs <- glmer(RT ~ 1 + tau_fs + (1 | ID) + (1 | Verb),
    data = human %>%
        filter(Accuracy == 1),
    family = Gamma(link = "identity")
)

summary(rt_tau_model_fs)

rt_tau_model_f <- glmer(RT ~ 1 + tau_f + (1 | ID) + (1 | Verb),
    data = human %>%
        filter(Accuracy == 1),
    family = Gamma(link = "identity")
)

summary(rt_tau_model_f)

rt_tau_model_s <- glmer(RT ~ 1 + tau_s + (1 | ID) + (1 | Verb),
    data = human %>%
        filter(Accuracy == 1),
    family = Gamma(link = "identity")
)

summary(rt_tau_model_s)


anova(rt_tau_model_fs, rt_tau_model_f, rt_tau_model_s)

stargazer::stargazer(rt_tau_model_fs, rt_tau_model_f, rt_tau_model_s, type = "text", title = "RTs with taus", align = TRUE, label = "tab:rt_tau", header = TRUE, digits = 3)

# correlation between RT and tau by condition
human %>%
    group_by(Condition) %>%
    summarise(correlation = cor(iRT, Mean_Tau.x, method = "kendall"))


# model with tau
glmm_rt_tau <- glmer(RT ~ 1 + Mean_Tau.x + (1 | ID) + (1 | Verb),
    data = human %>%
        filter(Accuracy == 1),
    family = Gamma(link = "identity")
)

summary(glmm_rt_tau)
# %%

# Frequency Stats

human %>%
    group_by(Condition) %>%
    summarise(
        mean_frequency = mean(Frequency, na.rm = TRUE),
        sd_frequency = sd(Frequency, na.rm = TRUE),
        min_frequency = min(Frequency, na.rm = TRUE),
        max_frequency = max(Frequency, na.rm = TRUE)
    ) %>%
    xtable(digits = 3)

# anova for frequency
aov_frequency <- aov(Frequency ~ Condition, data = human)
summary(aov_frequency)

emmeans::emmeans(aov_frequency, pairwise ~ Condition)

stats::aov(RT ~ Condition + scale(Frequency) + Error(Verb), data = human %>% filter(Accuracy == 1) %>% filter(Frequency > 20000)) %>%
    summary()
