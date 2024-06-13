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
library(brms)
# load data using relative path
minerva <- read_csv("minerva_full_results.csv")



# %%

# %%

# check data and data types
n <- n_distinct(minerva$participant) # number of participants should be 99
n_item <- n_distinct(minerva$item) # number of items should be 246
n_k <- n_distinct(minerva$minerva_k) # number of k
unique(minerva$minerva_k) # check unique k

# convert all columns except act and rt to factor
minerva <- minerva %>%
    mutate_at(vars(-act, -rt, -fitem, -score), as.factor)

# relevel factors for type
minerva$type <- relevel(minerva$type, ref = "prod")

# convert doubles to integers


# rename type levels using dplyr

minerva$type <- dplyr::recode(minerva$type,
    "prod" = "Productive",
    "collocation" = "Collocation",
    "idiom" = "Idiom"
)

unique(is.na(minerva$rt)) # check for missing values

# check data types
# skim(minerva)

# no missing values everything looks good

# %%

# %%
# parametric bootstrap

# Define a function to calculate mean and confidence interval
mean_ci <- function(x) {
  bs_result <- boot::boot(x, function(x, i) mean(x[i]), R = 1000)
  ci <- quantile(bs_result$t, c(0.025, 0.975))
  return(c(mean = mean(x), lower_ci = ci[1], upper_ci = ci[2]))
}

# Create a data frame with the results
conditions <- c("Productive", "Collocation", "Idiom")
k <- c(0.95, 0.96, 0.97, 0.98, 0.99)
forget_prob <- c(0.0, 0.2, 0.4, 0.6, 0.8)

results <- expand.grid(conditions, k, forget_prob)

# run the bootstrap for each condition and wrtie to the results data frame
for (i in 1:nrow(results)) {
  condition <- results[i, 1]
  k <- results[i, 2]
  forget_prob <- results[i, 3]
  data <- minerva %>% filter(type == condition & minerva_k == k & forget_prob == forget_prob)
  results[i, 4:6] <- mean_ci(data$rt)
}

head(results)

  data <- minerva %>% filter(type == condition & minerva_k == k & forget_prob == forget_prob)


# %%
# %%

# Descriptive Statistics -------------------------------------------------------

# plot mean rt for type facted by minerva_k, forget_prob
plot <- minerva %>%
    filter(is_noise_embeddings == "FALSE" & is_equal_frequency == "FALSE") %>%
    #filter outliers of fitem > 3 sd from the mean
    filter(fitem < mean(fitem, na.rm = TRUE) + 3 * sd(fitem, na.rm = TRUE) | fitem > mean(fitem, na.rm = TRUE) + 3 * sd(fitem, na.rm = TRUE)) %>%
    group_by(type, minerva_k, forget_prob) %>%
    summarise(mean_rt = mean(rt, na.rm = TRUE)) %>%
    ggplot(aes(x = type, y = mean_rt, fill = type)) +
    geom_bar(stat = "identity", position = "dodge") +
    labs(
        title = "Mean Tau by Minerva K and Forget Probability",
        subtitle = "Excluding Outliers, Noise Embeddings and Equal Frequency",
        x = "Item Type",
        y = "Tau",
        fill = "Item Type"
    ) +
    facet_grid(minerva_k ~ forget_prob) +
    theme_bw() +
    theme(
        title = element_text(size = 18, face = "bold"),
        axis.title = element_text(size = 14, face = "bold"),
        axis.text.y = element_text(size = 14, face = "bold"),
        axis.text.x = element_blank(),
        legend.title = element_text(size = 16, face = "bold"),
        legend.text = element_text(size = 14, face = "bold"),
        legend.position = "bottom",
        strip.text.x = element_text(size = 14, face = "bold"),
        strip.text.y = element_text(size = 14, face = "bold"),
        strip.background = element_rect(colour = "black", fill = "#ffffff")
    )
# add bootstraped confidence intervals
plot <- plot + geom_errorbar(aes(ymin = mean_rt - sd(rt, na.rm = TRUE) / sqrt(length(rt)), ymax = mean_rt + sd(rt, na.rm = TRUE) / sqrt(length(rt))), width = 0.2, position = position_dodge(0.9))
plot

ggsave("mean_rt_by_minerva_k_forget_prob_no_outliers.png", plot, width = 12, height = 8)
# %%


# %%

# Descriptive Statistics -------------------------------------------------------

# plot mean rt for type facted by minerva_k, forget_prob with noise embeddings only
plot <- minerva %>%
    filter(is_noise_embeddings == "TRUE" & is_equal_frequency == "FALSE") %>%
    group_by(type, minerva_k, forget_prob) %>%
    summarise(mean_rt = mean(rt, na.rm = TRUE)) %>%
    ggplot(aes(x = type, y = mean_rt, fill = type)) +
    geom_bar(stat = "identity", position = "dodge") +
    labs(
        title = "Mean Tau by Minerva K and Forget Probability",
        subtitle = "Only Noise Embeddings (no Equal Frequency)",
        x = "Item Type",
        y = "Tau",
        fill = "Item Type"
    ) +
    facet_grid(minerva_k ~ forget_prob) +
    theme_bw() +
    theme(
        title = element_text(size = 18, face = "bold"),
        axis.title = element_text(size = 14, face = "bold"),
        axis.text.y = element_text(size = 14, face = "bold"),
        axis.text.x = element_blank(),
        legend.title = element_text(size = 16, face = "bold"),
        legend.text = element_text(size = 14, face = "bold"),
        legend.position = "bottom",
        strip.text.x = element_text(size = 14, face = "bold"),
        strip.text.y = element_text(size = 14, face = "bold"),
        strip.background = element_rect(colour = "black", fill = "#ffffff")
    )

plot

ggsave("mean_rt_by_minerva_k_forget_prob_noise.png", plot, width = 12, height = 8)
# %%

# %%

# plot mean rt for type facted by minerva_k, forget_prob with equal frequency only

plot <- minerva %>%
    filter(is_noise_embeddings == "FALSE" & is_equal_frequency == "TRUE") %>%
    group_by(type, minerva_k, forget_prob) %>%
    summarise(mean_rt = mean(rt, na.rm = TRUE)) %>%
    ggplot(aes(x = type, y = mean_rt, fill = type)) +
    geom_bar(stat = "identity", position = "dodge") +
    labs(
        title = "Mean Tau by Minerva K and Forget Probability",
        subtitle = "Only Equal Frequency (no Noise Embeddings)",
        x = "Item Type",
        y = "Tau",
        fill = "Item Type"
    ) +
    facet_grid(minerva_k ~ forget_prob) +
    theme_bw() +
    theme(
        title = element_text(size = 18, face = "bold"),
        axis.title = element_text(size = 14, face = "bold"),
        axis.text.y = element_text(size = 14, face = "bold"),
        axis.text.x = element_blank(),
        legend.title = element_text(size = 16, face = "bold"),
        legend.text = element_text(size = 14, face = "bold"),
        legend.position = "bottom",
        strip.text.x = element_text(size = 14, face = "bold"),
        strip.text.y = element_text(size = 14, face = "bold"),
        strip.background = element_rect(colour = "black", fill = "#ffffff")
    )

plot

ggsave("mean_rt_by_minerva_k_forget_prob_equal.png", plot, width = 12, height = 8)
# %%

# %%
# plot mean rt for type facted by minerva_k, forget_prob with equal frequency and noise embeddings

plot <- minerva %>%
    filter(is_noise_embeddings == "TRUE" & is_equal_frequency == "TRUE") %>%
    group_by(type, minerva_k, forget_prob) %>%
    summarise(mean_rt = mean(rt, na.rm = TRUE)) %>%
    ggplot(aes(x = type, y = mean_rt, fill = type)) +
    geom_bar(stat = "identity", position = "dodge") +
    labs(
        title = "Mean Tau by Minerva K and Forget Probability",
        subtitle = "Equal Frequency and Noise Embeddings",
        x = "Item Type",
        y = "Tau",
        fill = "Item Type"
    ) +
    facet_grid(minerva_k ~ forget_prob) +
    theme_bw() +
    theme(
        title = element_text(size = 18, face = "bold"),
        axis.title = element_text(size = 14, face = "bold"),
        axis.text.y = element_text(size = 14, face = "bold"),
        axis.text.x = element_blank(),
        legend.title = element_text(size = 16, face = "bold"),
        legend.text = element_text(size = 14, face = "bold"),
        legend.position = "bottom",
        strip.text.x = element_text(size = 14, face = "bold"),
        strip.text.y = element_text(size = 14, face = "bold"),
        strip.background = element_rect(colour = "black", fill = "#ffffff")
    )

plot

ggsave("mean_rt_by_minerva_k_forget_prob_equal_noise.png", plot, width = 12, height = 8)

# %%

# %%
# plot frequency of type facted by minerva_k, forget_prob

plot <- minerva %>%
    group_by(type) %>%
    summarise(mean_freq = mean(fitem)) %>%
    ggplot(aes(x = type, y = mean_freq, fill = type)) +
    geom_bar(stat = "identity", position = "dodge") +
    labs(
        title = "Mean Frequency",
        subtitle = "Excluding Noise Embeddings and Equal Frequency",
        x = "Item Type",
        y = "Frequency",
        fill = "Item Type"
    ) +
    theme_bw() +
    theme(
        title = element_text(size = 18, face = "bold"),
        axis.title = element_text(size = 14, face = "bold"),
        axis.text.y = element_text(size = 14, face = "bold"),
        axis.text.x = element_blank(),
        legend.title = element_text(size = 16, face = "bold"),
        legend.text = element_text(size = 14, face = "bold"),
        legend.position = "bottom"
    )

plot

ggsave("mean_freq_by_type.png", plot, width = 12, height = 8)

# %%

# %%

# plot histogram of fitem by item overlap

plot <- minerva %>%
    # drop all columns except item and fitem
    select(item, fitem, type) %>%
    # unique rows
    distinct() %>%
    ggplot(aes(x = log10(fitem), fill = type, color = type)) +
    geom_histogram(binwidth = 0.25) +
    labs(
        title = "Frequency of Items",
        x = "Frequency",
        y = "Count"
    ) +
    theme_bw() +
    theme(
        title = element_text(size = 18, face = "bold"),
        axis.title = element_text(size = 14, face = "bold"),
        axis.text.y = element_text(size = 14, face = "bold"),
        axis.text.x = element_text(size = 14, face = "bold")
    )

plot

ggsave("hist_fitem.png", plot, width = 12, height = 8)

# %%

# %%

# linear mixed effects model ---------------------------------------------------

# run a linear model for each combination of minerva_k and forget_prob
# and save the results to a list

# create a list to store the results
lm_results <- list()

# loop through each combination of minerva_k and forget_prob
for (k in unique(minerva$minerva_k)) {
    for (fp in unique(minerva$forget_prob)) {
        # filter the data
        data <- minerva %>%
            filter(minerva_k == k & forget_prob == fp)
        # run the model
        model <- lmer(rt ~ type + scale(log10(fitem)) + (1 | participant), data = data %>% filter(is_noise_embeddings == "FALSE" & is_equal_frequency == "FALSE"))
        # save the results
        lm_results[[paste(k, fp, sep = "_")]] <- summary(model)
    }
}

print(lm_results)

# %%

# %%
# bayesian model averaging -----------------------------------------------------

# create a list to store the results
bma_results <- list()

# loop through each combination of minerva_k and forget_prob

for (k in unique(minerva$minerva_k)) {
    for (fp in unique(minerva$forget_prob)) {
        # filter the data
        data <- minerva %>%
            filter(minerva_k == k & forget_prob == fp)
        # run the model
        model <- brm(rt ~ type + scale(log10(fitem)) + (1 | participant), family = Gamma(link = "identity"), data = data %>% filter(is_noise_embeddings == "FALSE" & is_equal_frequency == "FALSE"))
        # save the results
        bma_results[[paste(k, fp, sep = "_")]] <- summary(model)
    }
}

# %%

# %%
