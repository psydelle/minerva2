# %% Setup --------------------------------------------------------------------

library(tidyverse) 
library(skimr) 
library(ggpubr) 
library(pander) 
library(xtable) 
library(patchwork) 
library(car) 
library(afex) 
library(sjPlot) 
library(lme4) 
library(boot) 
library(emmeans) 
library(BMS) 
library(brms)

# Load data
minerva <- read_csv("results/minerva_full_results.csv")

# %% Data Cleaning ------------------------------------------------------------

# Check distinct values
n <- n_distinct(minerva$participant)  # Should be 99
n_item <- n_distinct(minerva$item)    # Should be 246

# Convert all categorical variables (except specified ones) to factors
minerva <- minerva %>% 
    mutate(across(-c(act, rt, fitem, score), as.factor))

# Rename type levels
minerva$type <- recode(minerva$type, 
    "prod" = "Compositional", 
    "collocation" = "Collocation", 
    "idiom" = "Idiom"
)

# Set reference level
minerva$type <- relevel(minerva$type, ref = "Compositional")

# Check missing values
if (any(is.na(minerva$rt))) {
    warning("Missing values detected in RT column.")
}

# %% Descriptive Statistics ---------------------------------------------------

# General function to create plots
plot_rt <- function(data, title_suffix, filename) {
    p <- data %>%
        group_by(type, minerva_k, forget_prob) %>%
        summarise(mean_rt = mean(rt, na.rm = TRUE), .groups = "drop") %>%
        ggplot(aes(x = type, y = mean_rt, fill = type)) +
        geom_bar(stat = "identity", position = "dodge") +
        geom_errorbar(aes(ymin = mean_rt - sd(rt, na.rm = TRUE) / sqrt(n()), 
                          ymax = mean_rt + sd(rt, na.rm = TRUE) / sqrt(n())), 
                      width = 0.2, position = position_dodge(0.9)) +
        labs(
            title = paste("Mean Tau by Minerva K and Forget Probability", title_suffix),
            x = "Item Type", y = "Tau", fill = "Item Type"
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
            legend.position = "bottom"
        )

    ggsave(filename, p, width = 12, height = 8)
}

# Generate multiple plots dynamically
plot_rt(minerva %>% filter(is_noise_embeddings == "FALSE", is_equal_frequency == "FALSE"), 
        "Excluding Outliers, Noise Embeddings and Equal Frequency", 
        "mean_rt_no_outliers.png")

plot_rt(minerva %>% filter(is_noise_embeddings == "TRUE", is_equal_frequency == "FALSE"), 
        "Only Noise Embeddings", 
        "mean_rt_noise.png")

plot_rt(minerva %>% filter(is_noise_embeddings == "FALSE", is_equal_frequency == "TRUE"), 
        "Only Equal Frequency", 
        "mean_rt_equal.png")

plot_rt(minerva %>% filter(is_noise_embeddings == "TRUE", is_equal_frequency == "TRUE"), 
        "Equal Frequency and Noise Embeddings", 
        "mean_rt_equal_noise.png")

# %% Model Fitting ------------------------------------------------------------

# Function to fit linear mixed-effects models
fit_lmm <- function(data) {
    lmer(rt ~ type + scale(log10(fitem)) + (1 | participant), data = data)
}

# Nest data for model fitting
lm_results <- minerva %>% 
    filter(is_noise_embeddings == "FALSE", is_equal_frequency == "FALSE") %>% 
    group_by(minerva_k, forget_prob) %>% 
    nest() %>% 
    mutate(model = map(data, fit_lmm), 
           summary = map(model, summary))

# Print summary of models
print(lm_results$summary)