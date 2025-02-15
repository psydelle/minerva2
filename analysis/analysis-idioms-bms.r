# %%

## Set-Up ---------------------------------------------------------------------
# set current working directory
setwd("./results")
# install.packages(c("tidyverse",
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
library(pander) # for publication-ready tables
library(xtable) # for latex tables
# library(patchwork) # for combining plots
library(lme4) # for mixed effects models
library(brms)


# set theme for ggplot
theme_set(theme_bw())

# use cores for parallel processing
options(mc.cores = parallel::detectCores())

# colors
# options(ggplot2.discrete.fill = c("#00AFBB", "#E7B800", "#FC4E07"))
options(ggplot2.discrete.fill = c("#387ADF", "#FF4500", "#50C4ED", "#646262"))

options(digits = 3)

# set seed for reproducibility
set.seed(0976)

# %%

minerva <- read_csv("results\\minerva_full_results.csv")

minerva <- minerva %>% dplyr::select(c(participant, item, type, verb, rt, minerva_k, forget_prob, fitem, score, is_noise_embeddings, is_equal_frequency, embedding_model)) # nolint

# convert all columns except tau, fitem, and score to factor
minerva <- minerva %>%
    mutate_at(vars(-rt, -fitem, -score, -minerva_k, -forget_prob), as.factor)


# rename type levels using dplyr

minerva$type <- dplyr::recode(minerva$type,
    "prod" = "Compositional",
    "collocation" = "Collocation",
    "idiom" = "Idiom"
)

# relevel factors for type
minerva$type <- relevel(minerva$type, ref = "Compositional")


# check unique values for factor variables
unique(is.na(minerva$rt)) # check for missing values
n_distinct(minerva$participant) # check unique participant

type <- unique(minerva$type) # check unique type
type

unique(minerva$minerva_k) # check unique k
unique(minerva$forget_prob) # check unique forget_prob
unique(minerva$is_noise_embeddings) # check semantic condition
unique(minerva$is_equal_frequency) # check frequency condition
unique(minerva$embedding_model) # check unique embedding model

# add a column for embedding type
minerva <- minerva %>%
    mutate(Embedding = ifelse(embedding_model == "sbert", "Contextual", "Non-contextual")) # nolint

# add a column for experiment type

minerva <- minerva %>%
    mutate(
        Experiment = ifelse(is_noise_embeddings == "TRUE" & is_equal_frequency == "FALSE", "Frequency-only", # nolint: line_length_linter.
            ifelse(is_equal_frequency == "TRUE" & is_noise_embeddings == "FALSE", "Semantics-only", # nolint: line_length_linter.
                ifelse(is_noise_embeddings == "FALSE" & is_equal_frequency == "FALSE", "Frequency & Semantics", "Null Model") # nolint: line_length_linter.
            )
        ) # nolint
    )

unique(minerva$Experiment) # check unique experiment

# rename columns
minerva <- minerva %>% rename(
    "ID" = participant,
    "Item" = item,
    "Condition" = type,
    "Tau" = rt,
    "K" = minerva_k,
    "Forget" = forget_prob,
    "Frequency" = fitem,
    "Score" = score,
    "Model" = embedding_model,
    "Verb" = verb
)


# print col names
colnames(minerva)

# drop columns
minerva <- minerva %>% dplyr::select(-c(is_noise_embeddings, is_equal_frequency))

# filter out verbs with low accuracy as we did in human experiments
minerva <- minerva %>% filter(Verb != "silence")
minerva <- minerva %>% filter(Verb != "muzzle")
minerva <- minerva %>% filter(Verb != "pad")
minerva <- minerva %>% filter(Verb != "slap")

# from here on , we only deal with contextual embeddings
minerva <- minerva %>% filter(Model == "sbert")

# write to csv
write_csv(minerva, "minerva_contextual_clean.csv")

# %%

# check experiment data

n_id <- n_distinct(minerva$ID) # number of participants should be 300
sprintf("Number of participants: %s", n_id)

n_condition <- n_distinct(minerva$Condition) # number of conditions should be 3
sprintf("Number of experimental conditions: %s", n_condition)

n_item <- n_distinct(minerva$Item) # number of items should be 246
sprintf("Number of experimental items: %s", n_item)

n_k <- n_distinct(minerva$K) # number of ks should be 6
sprintf("Number of different ks: %s", n_k)

n_forget <- n_distinct(minerva$Forget) # number of forget probabilities should be 5
sprintf("Number of different forget probabilities: %s", n_forget)

# %%

#  create a new df with mean tau for each item by experiment

# tau_by_item <- minerva %>%
#     group_by(Item, Experiment) %>%
#     summarise(mean_tau = mean(Tau)) %>%
#     ungroup()

# # pivot the df to wide format
# tau_by_item_wide <- tau_by_item %>%
#     pivot_wider(names_from = Experiment, values_from = mean_tau)


# write_csv(tau_by_item_wide, "tau_by_item_by_experiment.csv")


# plot barplot of mean tau by condition for each experiment

minerva %>%
    ggplot(aes(x = Condition, y = Tau, fill = Condition)) +
    geom_bar(stat = "summary", fun = "mean", position = "dodge", color = "black", linewidth = 0.8) +
    geom_errorbar(stat = "summary", fun.data = "mean_cl_boot", position = position_dodge(width = 0.90), width = 0.25, linewidth = 0.8) +
    geom_text(stat = "summary", fun = mean,  aes(label = round(..y.., 1)), 
              vjust = -1.3, size = 10, fontface = "bold") + labs(y = "Mean RT (ms)") +
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
        ) + facet_wrap(~Experiment, ncol = 4, scales = "free")

# %%

k99_forget08 <- bind_rows(
    main %>% filter(K == 0.99, Forget == 0.8),
    frequency %>% filter(K == 0.99, Forget == 0.8),
    semantics %>% filter(K == 0.99, Forget == 0.8)
)

# boxplot
k99_forget08 %>%
    filter(Tau < 300) %>%
    ggplot(aes(x = Experiment, y = Tau, fill = Condition)) +
    geom_boxplot() +
    theme_minimal() +
    labs(title = "k = 0.99, forget = 0.8", x = "Condition", y = "RT") +
    theme(legend.position = "right") +
    facet_wrap(~Model, ncol = 2)



# %%

# %%

# table of number of timeouts per condition for each k and forget probability combination

timeouts <- minerva %>%
    # filter(Model == "sbert") %>%
    filter(Experiment != "Null Model") %>%
    filter(Tau == 300) %>%
    group_by(K, Forget, Experiment, Condition, Embedding, Item) %>%
    summarise(n_timeouts = n())

timeouts

# plot of number of timeouts per condition for each k and forget probability combination
for (i in unique(timeouts$Experiment)) {
    plot <- timeouts %>%
    filter(K == 0.99, Forget == 0.6) %>%
        filter(Experiment == i) %>%
        ggplot(aes(x = Embedding, y = n_timeouts, fill = Condition)) +
    geom_bar(stat = "summary", fun = "mean", position = "dodge", color = "black", linewidth = 0.8) +
    geom_errorbar(stat = "summary", fun.data = "mean_cl_boot", position = position_dodge(width = 0.90), width = 0.25, linewidth = 0.8) +
        theme_minimal() +
        labs(title = paste0("Number of Timeouts per Condition"), subtitle = i, x = "Embedding Type", y = "Number of Timeouts") +
        facet_grid(K ~ Forget, space = "free") +
        theme(
            title = element_text(size = 60, face = "bold"),
            axis.line = element_line(colour = "black", linewidth = 0.8, lineend = "round"),
            axis.title = element_text(size = 60, face = "bold"),
            axis.text = element_text(size = 50, face = "bold"),
            # axis.text.x = element_blank(),
            axis.text.x = element_text(angle = 45, hjust = 1),
            legend.title = element_text(size = 55, face = "bold"),
            legend.text = element_text(size = 50, face = "bold"),
            legend.position = "right",
            strip.text.x = element_text(size = 50, face = "bold"),
            strip.text.y = element_text(size = 50, face = "bold"),
            strip.background = element_rect(colour = "black", fill = "white", linewidth = 0.8),
            panel.background = element_rect(colour = "black", fill = "white", linewidth = 0.8),
            panel.grid.major = element_line(colour = "grey", linewidth = 0.5),
            panel.spacing = unit(0, "points"),
            plot.background = element_rect(colour = "white", fill = "white")
        )
    print(plot)
    ggsave(paste0("timeouts_", i, ".png"), plot, width = 35, height = 25, units = "in", dpi = 300)
}

timeouts %>%
    filter(K == 0.99, Forget == 0.6) %>%
    ggplot(aes(x = Condition, y = n_timeouts, fill = Condition)) +
    geom_bar(stat = "summary", fun = "mean", position = "dodge", color = "black", linewidth = 0.8) +
    geom_errorbar(stat = "summary", fun.data = "mean_cl_boot", position = position_dodge(width = 0.90), width = 0.25, linewidth = 0.8) +
    theme_minimal() +
    labs(title = "Average Timeouts per Condition", subtitle = "K = 0.99; Forget Probability = 0.6", x = "Condition", y = "Number of Timeouts") +
    facet_grid(Experiment ~ Embedding, space = "free") +
    theme(
        title = element_text(size = 60, face = "bold"),
        axis.line = element_line(colour = "black", linewidth = 0.8, lineend = "round"),
        axis.title = element_text(size = 60, face = "bold"),
        axis.text = element_text(size = 50, face = "bold"),
        axis.text.x = element_blank(),
        # axis.text.x = element_text(angle = 45, hjust = 1),
        legend.title = element_text(size = 55, face = "bold"),
        legend.text = element_text(size = 50, face = "bold"),
        legend.position = "right",
        strip.text.x = element_text(size = 50, face = "bold"),
        strip.text.y = element_text(size = 50, face = "bold"),
        strip.background = element_rect(colour = "black", fill = "white", linewidth = 0.8),
        panel.background = element_rect(colour = "black", fill = "white", linewidth = 0.8),
        panel.grid.major = element_line(colour = "grey", linewidth = 0.5),
        panel.spacing = unit(0, "points"),
        plot.background = element_rect(colour = "white", fill = "white")
    )

ggsave("plot_k99f06_only_timeouts.png", width = 30, height = 28, units = "in", dpi = 300)
# %%

# %%
# bind all data together
k98_forget08 <- bind_rows(
    main %>% filter(K == 0.98, Forget == 0.8),
    frequency %>% filter(K == 0.98, Forget == 0.8),
    semantics %>% filter(K == 0.98, Forget == 0.8)
)

k99_forget06 <- bind_rows(
    main %>% filter(K == 0.99, Forget == 0.6),
    frequency %>% filter(K == 0.99, Forget == 0.6),
    semantics %>% filter(K == 0.99, Forget == 0.6)
)

# boxplot
k98_forget08 %>%
    # filter(Tau < 300) %>%
    ggplot(aes(x = Experiment, y = Tau, fill = Condition)) +
    geom_boxplot() +
    theme_minimal() +
    labs(title = "k = 0.98, forget = 0.6", x = "Condition", y = "RT") +
    theme(legend.position = "right") +
    facet_wrap(~Model, ncol = 2)

# bar plot with 95% confidence interval

k98_forget08 %>%
filter(Tau < 300) %>%
    ggplot(aes(x = Experiment, y = Tau, fill = Condition)) +
    geom_bar(stat = "summary", fun = "mean", position = "dodge") +
    geom_errorbar(stat = "summary", fun.data = "mean_cl_boot", position = position_dodge(width = 0.90), width = 0.25) +
    theme_minimal() +
    labs(title = "k = 0.98, forget = 0.6", x = "Condition", y = "RT") +
    theme(legend.position = "right") +
    facet_wrap(~Model, ncol = 2)

# %%


# %%

# run linear mixed effects model for combinations of k and forget, sbert

# set ref level for condition to idioms
k98_forget08$Condition <- relevel(k98_forget08$Condition, ref = "Idiom")

m1 <- glmer(Tau ~ Condition + (1 | ID) + (1 | Verb),
    data = k98_forget08 %>% 
    filter(Model == "sbert") %>%
    filter(Experiment == "Frequency & Semantics"),
    family = Gamma(link = "identity"),
    control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
)
summary(m1)

m2 <- glmer(Tau ~ Condition + (1 | ID),
    data = k99_forget06 %>% filter(Model == "sbert") %>% filter(Experiment == "Frequency-only"),
    family = Gamma(link = "log"), control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
)

summary(m2)

m3 <- glmer(Tau ~ Condition + (1 | ID),
    data = k99_forget06 %>% filter(Model == "sbert") %>% filter(Experiment == "Semantics-only"),
    family = Gamma(link = "log"), control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
)

summary(m3)
# bernoulli family for binary outcome

minerva$Timeout <- ifelse(minerva$Tau == 300, 1, 0)

# proportion of timeouts
prop.table(table(minerva$Timeout))


m4 <- glmer(cbind(Timeout, 1 - Timeout) ~ Condition + scale(log10(Frequency)) + (1 | ID),
    data = k99_forget06 %>% filter(Model == "sbert") %>% filter(Experiment == "Frequency & Semantics"),
    family = binomial(link = "logit"), control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
)

summary(m4)

m5 <- glmer(cbind(Timeout, 1 - Timeout) ~ Condition + scale(log10(Frequency)) + (1 | ID),
    data = k99_forget06 %>% filter(Model == "sbert") %>% filter(Experiment == "Semantics-only"),
    family = binomial(link = "logit"), control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
)

summary(m5)
# %%


# %%

# run linear mixed effects model for combinations of k and forget, fasttext

m4 <- glmer(Tau ~ Condition + (1 | ID),
    data = k99_forget06 %>% filter(Model == "fasttext") %>% filter(!(verb %in% outliers$verb)) %>% filter(Experiment == "Frequency & Semantics"),
    family = Gamma(link = "log"), control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
)

summary(m4)

m5 <- glmer(Tau ~ Condition + (1 | ID),
    data = k99_forget06 %>% filter(Model == "fasttext") %>% filter(!(verb %in% outliers$verb)) %>% filter(Experiment == "Frequency-only"),
    family = Gamma(link = "log"), control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
)

summary(m5)

m6 <- glmer(Tau ~ Condition + (1 | ID),
    data = k99_forget06 %>% filter(Model == "fasttext") %>% filter(!(verb %in% outliers$verb)) %>% filter(Experiment == "Semantics-only"),
    family = Gamma(link = "log"), control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
)

summary(m6)

# %%

# %%
# brms model

fit1 <- brm(Tau ~ Condition + (1 | ID),
    data = main %>% filter(Model == "sbert") %>% filter(!(verb %in% outliers$verb)) %>% filter(K == 0.99 & Forget == 0.8),
    family = Gamma(link = "log"), cores = 6
)

summary(fit1)
posterior_summary(fit1)

hypothesis(fit1)
# plot the model
plot(fit1)

# save the model
save(fit1, file = "fit1.rda")


fit2 <- brm(Tau ~ Condition + (1 | ID),
    data = frequency %>% filter(Model == "sbert") %>% filter(!(verb %in% outliers$verb)) %>% filter(K == 0.99 & Forget == 0.8),
    family = Gamma(link = "log"), cores = 6
)

summary(fit2)

# plot the model
plot(fit2)

# save the model
save(fit2, file = "fit2.rda")




# %%

# %%

# bar plot with 95% confidence interval

k99_forget06 %>%
    filter(!(verb %in% outliers$verb)) %>% # filter out outliers
    ggplot(aes(x = Experiment, y = Tau, fill = Condition)) +
    geom_bar(stat = "summary", fun = "mean", position = "dodge") +
    geom_errorbar(stat = "summary", fun.data = "mean_cl_boot", position = position_dodge(width = 0.90), width = 0.25) +
    theme_minimal() +
    labs(title = "k = 0.99, forget = 0.6", x = "Condition", y = "TAU") +
    facet_wrap(~Model, ncol = 2) +
    theme(legend.position = "right")
# %%

# %%

# plots for k = 0.99, forget = 0.6  for all experiments
plot <- k99_forget06 %>%
    # filter(!(Verb %in% outliers$Verb)) %>%
    # filter(Tau != 300) %>% # filter out timeouts
    ggplot(aes(x = Condition, y = Tau, fill = Condition)) +
    geom_bar(stat = "summary", fun = "mean", position = "dodge", color = "black", linewidth = 0.8) +
    geom_errorbar(stat = "summary", fun.data = "mean_cl_boot", position = position_dodge(width = 0.90), width = 0.25, linewidth = 0.8) +
    facet_grid(Experiment ~ Embedding, space = "fixed", scales = "free") +
    theme_minimal() +
    labs(title = "Mean Tau per Condition (with Timeouts)", subtitle = "K = 0.99; Forget Probability = 0.6", x = "Condition", y = "Tau") +
    theme(
        title = element_text(size = 60, face = "bold"),
        axis.line = element_line(colour = "black", linewidth = 0.8, lineend = "round"),
        axis.title = element_text(size = 60, face = "bold"),
        axis.text = element_text(size = 50, face = "bold"),
        axis.text.x = element_blank(),
        # legend.title = element_text(size = 55, face = "bold"),
        legend.title = element_blank(),
        legend.text = element_text(size = 50, face = "bold"),
        legend.position = "bottom",
        strip.text.x = element_text(size = 50, face = "bold"),
        strip.text.y = element_text(size = 50, face = "bold"),
        strip.background = element_rect(colour = "black", fill = "white", linewidth = 0.8),
        panel.background = element_rect(colour = "black", fill = "white", linewidth = 0.8),
        panel.grid.major = element_line(colour = "grey", linewidth = 0.5),
        panel.spacing = unit(0, "points"),
        plot.background = element_rect(colour = "white", fill = "white")
    )
print(plot)

ggsave("plot_k99f06_with_timeouts.png", plot, width = 35, height = 30, units = "in", dpi = 300)


# %%


# %%

# all plots

# for (i in unique(k99_forget08$Experiment)) {
#     print(i)
#     plot <- k99_forget08 %>%
#         filter(Experiment == i) %>%
#         filter(!(Verb %in% outliers$Verb)) %>%
#         filter(K >= 0.98 & Forget >= 0.4) %>% # filter out low k and forget
#         ggplot(aes(x = Embedding, y = Tau, fill = Condition)) +
#         geom_bar(stat = "summary", fun = "mean", position = "dodge", color = "black", linewidth = 0.8) +
#         geom_errorbar(stat = "summary", fun.data = "mean_cl_boot", position = position_dodge(width = 0.90), width = 0.25, linewidth = 0.8) +
#         facet_grid(Forget ~ K, as.table = TRUE, space = "free_y") +
#         theme_minimal() +
#         labs(title = i, subtitle = "No Frequency Outliers", x = "Embedding Type", y = "Tau") +
#         theme(
#             title = element_text(size = 60, face = "bold"),
#             axis.line = element_line(colour = "black", linewidth = 0.8, lineend = "round"),
#             axis.title = element_text(size = 60, face = "bold"),
#             axis.text = element_text(size = 50, face = "bold"),
#             legend.title = element_text(size = 55, face = "bold"),
#             legend.text = element_text(size = 50, face = "bold"),
#             legend.position = "right",
#             strip.text.x = element_text(size = 50, face = "bold"),
#             strip.text.y = element_text(size = 50, face = "bold"),
#             strip.background = element_rect(colour = "black", fill = "white", linewidth = 0.8),
#             panel.background = element_rect(colour = "black", fill = "white", linewidth = 0.8),
#             panel.grid.major = element_line(colour = "grey", linewidth = 0.5),
#             panel.spacing = unit(0, "points"),
#             plot.background = element_rect(colour = "white", fill = "white")
#         ) +
#         ggsignif::geom_signif(comparisons = list(c("Productive", "Collocation"), c("Productive", "Idiom"), c("Collocation", "Idiom")), map_signif_level = TRUE, textsize = 10, vjust = -0.5, hjust = 0.5)

#     print(plot)
#     ggsave(paste0("plot_k99f08_no_outliers_", i, ".png"), plot, width = 25, height = 25, units = "in", dpi = 300)
# }

# %%
# check how well tau fits gamma distribution
summary(fitdist(minerva$tau, "gamma"))
# plot(fitdist(minerva$tau, "gamma"))


fit1 <- brm(bf(RT ~ Tau, shape ~ (1 | ID)),
    data = k99_forget08 %>% filter(ID == 1), family = Gamma(link = "identity"), cores = 2, iter = 1000, chains = 2, seed= 123)

# # summary(fit1)

# # save the model
# save(fit1, file = "fit1.rda")

# Link is set to log because of negative initial values, below for details
# https://discourse.mc-stan.org/t/fitting-gamma-model-with-brms/10728
# %%

library(tidyverse)
annotated <- read_csv("C:\\Users\\psyde\\OneDrive\\Documents\\GitHub\\minerva2\\data\\stimuli_idioms_clean_annotated1.csv")
colnames(annotated)
# remove trailing whitespace
annotated <- annotated %>%
    mutate(stimuli_grammatical = str_trim(stimuli_grammatical, side = "right"))

head(annotated$stimuli_grammatical)

annotated$correct_ajt_response <- "y"
unique(annotated$correct_ajt_response)
# drop columns and write to json

annotated <- annotated %>% dplyr::select(c(item, type, fold, stimuli_grammatical, stimuli_plural, correct_ajt_response))
head(annotated)


# write to json

jsonlite::write_json(annotated, "C:\\Users\\psyde\\OneDrive\\Documents\\GitHub\\minerva2\\data\\stimuli_idioms_clean_annotated.json")

# %%
