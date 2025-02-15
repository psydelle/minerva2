## DOCUMENT DETAILS ----------------------------------------------------------

# Project: Analogy2024 Workshop
# Title: Stats for ProbMod simulations
# Author: Sydelle de Souza
# Date: 2024/07/18

#-----------------------------------------------------------------------------#

## COMMENTS -------------------------------------------------------------------

# This script is for the analysis of the simulations of the probability model
# we need to merge the data from the simulations and then run the analysis
# we will perform correlation analyses and regression analyses

#-----------------------------------------------------------------------------#

## ACKNOWLEDGEMENTS  ----------------------------------------------------------

#
#

#-----------------------------------------------------------------------------#

## Set-Up ---------------------------------------------------------------------
# %%
# load libraries
library(tidyverse) # data wrangling
library(skimr) # summary statistics
library(ggpubr) # for publication-ready plots
library(patchwork)

# colors
options(ggplot2.discrete.fill = c("#387ADF", "#FF4500", "#50C4ED", "#646262"))
options(digits = 3)

# %%

#-----------------------------------------------------------------------------#
# %%
## Data -----------------------------------------------------------------------
# load and merge multiple csvs
# list all files in the data/processed directory
list.files(path = "data/processed", pattern = "*.csv")

# read in all files (there should be 4)
files <- list.files(path = "data/processed", pattern = "*.csv", full.names = TRUE) # nolint: line_length_linter.
head(files)

# open files and read in data
data <- files %>%
    map_dfr(read_csv)

# change NA to None and capitalize the other values
data$noisy_probes <- ifelse(is.na(data$noisy_probes), "Unnoised",
    ifelse(data$noisy_probes == "nouns", "Noisy Nouns", data$noisy_probes)
)

# load stimuli file and merge by item
stimuli <- read_csv("data/stimuli_idioms_clean.csv")

# merge data with stimuli
data <- data %>%
    left_join(stimuli, by = c("item" = "item"))

head(data)
# %%
#-----------------------------------------------------------------------------#

## Exploratory Data Analysis --------------------------------------------------

# convert type and noisy_probes to factors
data$type <- as.factor(data$type)
data$noisy_probes <- as.factor(data$noisy_probes)

# factor levels for type

data$type <- ifelse(data$type == "prod", "Productive", str_to_title(data$type))
unique(data$type)

data$type <- factor(data$type, levels = c("Productive", "Collocation", "Idiom"))
data$noisy_probes <- factor(data$noisy_probes, levels = c("Unnoised", "Noisy Nouns", "verbs", "both"))

nrow(data)

# boxplot of utilities by type faceted by verb with scales free

plot <- data %>%
    group_by(type) %>%
    filter(noisy_probes == "Unnoised" | noisy_probes == "Noisy Nouns") %>%
    ggplot(aes(x = type, y = utility, fill = type)) +
    geom_bar(stat = "summary", fun = "mean") +
    geom_errorbar(stat = "summary", fun.data = "mean_cl_boot", width = 0.3, linewidth = 1) +
    labs(
        subtitle = "Probability of Recognizing a Probe",
        x = "Condition",
        y = "p(Probe | Sum(instances in matrix))",
        fill = "Condition"
    ) +
    facet_wrap(~noisy_probes, ncol = 2) +
    theme_bw() +
    theme(
            title = element_text(size = 50, face = "bold"),
            axis.line = element_line(colour = "black", linewidth = 0.8, lineend = "round"),
            axis.title = element_text(size = 50, face = "bold"),
            axis.text = element_text(size = 40, face = "bold"),
            axis.text.x = element_blank(),
            # axis.text.x = element_text(angle = 45, hjust = 1),
            legend.title = element_blank(),
            legend.text = element_text(size = 45, face = "bold"),
            legend.position = "bottom",
            strip.text.x = element_text(size = 45, face = "bold"),
            strip.text.y = element_text(size = 45, face = "bold"),
            strip.background = element_rect(colour = "black", fill = "white", linewidth = 0.8),
            panel.background = element_rect(colour = "black", fill = "white", linewidth = 0.8),
            panel.grid.major = element_line(colour = "grey", linewidth = 0.5),
            panel.spacing = unit(0, "points"),
            plot.background = element_rect(colour = "white", fill = "white")
        )

plot

ggsave("figures/utility_plots.png", plot, width = 16, height = 20)

data %>%
    filter(noisy_probes == "Unnoised") %>%
    ggplot(aes(x = type, y = utility, fill = type)) +
    geom_boxplot() +
    # facet_wrap(~verb, scales = "free") +
    theme_minimal() +
    labs(
        title = "Utilities of Item by Item Type",
        subtitle = "Unnoised",
        x = "Type",
        y = "Utility"
    )

noised <- data %>%
    filter(noisy_probes == "Noisy Nouns") %>%
    ggplot(aes(x = type, y = utility, fill = type)) +
    geom_boxplot() +
    # facet_wrap(~verb, scales = "free") +
    theme_minimal() +
    theme(legend.position = "none") +
    labs(
        title = "Utilities of Item by Item Type",
        subtitle = "Unnoised",
        x = "Type",
        y = "Utility"
    )


    ggsave("figures/utility_plots.png", width = 35, height = 30, dpi = 300, legend())


# barplot of utilities by type faceted by verb with scales free

data %>%
    filter(noisy_probes == "Unnoised" | noisy_probes == "Noisy Nouns") %>%
    ggplot(aes(x = type, y = utility, fill = type)) +
    geom_bar(stat = "summary", fun = "mean") +
    geom_errorbar(stat = "summary", fun.data = "mean_cl_normal", width = 0.2) +
    facet_grid(cols = type, scales = "free") +
    theme(legend.position = "none") +
    labs(
        title = "Global Utilities of Item by Item Type",
        subtitle = "Unnoised",
        x = "Condition",
        y = "Utility"
    )


    # free_y = TRUE, # set rows and columns for



