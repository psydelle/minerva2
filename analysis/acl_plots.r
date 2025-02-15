#%%

library(tidyverse) # data wrangling
library(skimr) # summary statistics
library(pander) # for publication-ready tables
library(xtable) # for latex tables
# library(patchwork) # for combining plots
library(lme4) # for mixed effects models

# set theme for ggplot
theme_set(theme_bw())

# use cores for parallel processing
options(mc.cores = parallel::detectCores())

# options(ggplot2.discrete.fill = c("#00AFBB", "#E7B800", "#FC4E07"))
options(ggplot2.discrete.fill = c("#387ADF", "#FF4500", "#50C4ED", "#646262"))

options(digits = 3)

# set seed for reproducibility
set.seed(0976)

# read data
minerva <- read_csv("minerva_contextual_clean.csv")

#%%
# check experiment data

n_id <- n_distinct(minerva$ID) # number of participants should be 300
sprintf("Number of participants: %s", n_id)

n_condition <- n_distinct(minerva$Condition) # number of conditions should be 3
sprintf("Number of experimental conditions: %s", n_condition)

n_item <- n_distinct(minerva$Item) # number of items should be 234 we removed 12 items from the original list
sprintf("Number of experimental items: %s", n_item)

n_k <- n_distinct(minerva$K) # number of ks should be 6
sprintf("Number of different ks: %s", n_k)

n_forget <- n_distinct(minerva$Forget) # number of forget probabilities should be 5
sprintf("Number of different forget probabilities: %s", n_forget)

experiments <- unique(minerva$Experiment) 

# relevel the condition
minerva$Condition <- factor(minerva$Condition, levels = c("Compositional", "Collocation", "Idiom"))
minerva$Experiment <- factor(minerva$Experiment, levels = c(experiments))


#%%
plot <-  minerva %>%
    filter(Experiment == "Null Model") %>%
    group_by(Condition, K, Forget) %>%
    ggplot(aes(x = Condition, y = Tau, fill = Condition)) +
    geom_bar(stat = "summary", fun = "mean", position = "dodge", color = "black", linewidth = 0.8) +
    geom_errorbar(stat = "summary", fun.data = "mean_cl_boot", position = position_dodge(width = 0.90), width = 0.25, linewidth = 0.8) +
    labs(title = "Null Model", y = "Tau", subtitle = "With Contextual Embeddings") +
    theme_bw() +
    theme(
            title = element_text(size = 26, face = "bold"),
            axis.line = element_line(colour = "black", linewidth = 0.8, lineend = "round"),
            axis.title = element_text(size = 18, face = "bold"),
            axis.title.x = element_blank(),
            axis.text.y = element_text(size = 20, face = "bold"),
            axis.text.x = element_blank(),
            legend.title = element_blank(),
            legend.text = element_text(size = 20, face = "bold"),
            legend.position = "bottom",
            strip.text.x = element_text(size = 18, face = "bold"),
            strip.text.y = element_text(size = 18, face = "bold"),
            strip.background = element_rect(colour = "black", fill = "white", linewidth = 0.8),
            panel.background = element_rect(colour = "black", fill = "white", linewidth = 0.8),
            panel.grid.major = element_line(colour = "grey", linewidth = 0.5),
            panel.spacing = unit(0, "points"),
            plot.background = element_rect(colour = "white", fill = "white"),
        ) +   facet_grid(K ~ Forget, scales = "free_y")

ggsave("barplot-null-contextual.png", plot, width = 30, height = 50, units = "cm")


#%%

k99f08 <- minerva %>%
    filter(K == 0.99, Forget == 0.8) %>% filter(Experiment != "Null Model")
    
    
plot <-  k99f08 %>%
    group_by(Condition, Experiment) %>%
    filter(Tau <300) %>%
    ggplot(aes(x = Condition, y = Tau, fill = Condition)) +
    geom_bar(stat = "summary", fun = "mean", position = "dodge", color = "black", linewidth = 0.8) +
    geom_errorbar(stat = "summary", fun.data = "mean_cl_boot", position = position_dodge(width = 0.90), width = 0.25, linewidth = 0.8) +
    labs(y = "Tau") +
    theme_bw() +
    theme(
            title = element_text(size = 26, face = "bold"),
            axis.line = element_line(colour = "black", linewidth = 0.8, lineend = "round"),
            axis.title = element_text(size = 18, face = "bold"),
            axis.title.x = element_blank(),
            axis.title.y = element_text(size = 22, face = "bold"),
            axis.text.y = element_text(size = 22, face = "bold"),
            axis.text.x = element_blank(),
            legend.title = element_blank(),
            legend.text = element_text(size = 20, face = "bold"),
            legend.position = "bottom",
            strip.text.x = element_text(size = 18, face = "bold"),
            strip.text.y = element_text(size = 18, face = "bold"),
            strip.background = element_rect(colour = "black", fill = "white", linewidth = 0.8),
            panel.background = element_rect(colour = "black", fill = "white", linewidth = 0.8),
            panel.grid.major = element_line(colour = "grey", linewidth = 0.5),
            panel.spacing = unit(0, "points"),
            plot.background = element_rect(colour = "white", fill = "white"),
        ) +   facet_wrap(~Experiment, scales = "free_y", ncol = 4)

        plot

ggsave("barplot-k99f08-notimeouts-contextual.png", plot, width = 30, height = 20, units = "cm")

# set reference level for the condition to idiom
k99f08$Condition <- relevel(k99f08$Condition, ref = "Compositional")
glmm_tau <- glmer(Tau ~ Condition + scale(Frequency) + (1|ID), data = k99f08 %>% filter(Experiment == "Frequency & Semantics"), , family = Gamma(link = "identity"))
summary(glmm_tau) 

stargazer::stargazer(glmm_tau, type = "latex", ci = TRUE, ci.level = 0.95, title = "GLMM for Tau with K = 0.99 and Forget = 0.8")

summary(lm(Frequency ~ Condition, data = k99f08 %>% filter(Experiment == "Frequency & Semantics")))

# descriptive statistics for frequency
k99f08 %>%
    group_by(Condition) %>%
    summarise(mean = mean(Frequency), sd = sd(Frequency), median = median(Frequency), min = min(Frequency), max = max(Frequency), n = n()) %>%
    xtable()



# reorder level for the condition to compositional, collocation, idiom

k99f08$Condition <- factor(k99f08$Condition, levels = c("Compositional", "Collocation", "Idiom"))


# plots for counts of timeouts
plot <-  k99f08 %>%
    group_by(Condition, Experiment) %>%
    filter(Tau == 300) %>%
    ggplot(aes(x = Condition, fill = Condition)) +
    geom_bar(stat = "count", position = "dodge", color = "black", linewidth = 0.8) +
    labs(y = "Count") +
    theme_bw() +
    theme(
            title = element_text(size = 26, face = "bold"),
            axis.line = element_line(colour = "black", linewidth = 0.8, lineend = "round"),
            axis.title = element_text(size = 18, face = "bold"),
            axis.title.x = element_blank(),
            axis.title.y = element_text(size = 22, face = "bold"),
            axis.text.y = element_text(size = 22, face = "bold"),
            axis.text.x = element_blank(),
            legend.title = element_blank(),
            legend.text = element_text(size = 20, face = "bold"),
            legend.position = "bottom",
            strip.text.x = element_text(size = 18, face = "bold"),
            strip.text.y = element_text(size = 18, face = "bold"),
            strip.background = element_rect(colour = "black", fill = "white", linewidth = 0.8),
            panel.background = element_rect(colour = "black", fill = "white", linewidth = 0.8),
            panel.grid.major = element_line(colour = "grey", linewidth = 0.5),
            panel.spacing = unit(0, "points"),
            plot.background = element_rect(colour = "white", fill = "white"),
        ) +   facet_wrap(~Experiment, scales = "free_y", ncol = 4)

        plot

ggsave("barplot-k99f08-timeouts-count-contextual.png", plot, width = 30, height = 20, units = "cm")

