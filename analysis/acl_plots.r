# %%

library(tidyverse) # data wrangling
library(skimr) # summary statistics
library(pander) # for publication-ready tables
library(xtable) # for latex tables
library(patchwork) # for combining plots
library(lme4) # for mixed effects models
library(performance)
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


# copy irt and iaccuracy columns to minerva by item
# minerva <- minerva %>%
#     left_join(human %>% select(Item, iRT, iAccuracy), by = "Item")


# %%
# options(ggplot2.discrete.fill = c("#00AFBB", "#E7B800", "#FC4E07"))
options(ggplot2.discrete.fill = c("#387ADF", "#FF4500", "#50C4ED", "#646262"))
plot_theme <- theme(
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
    plot.background = element_rect(colour = "white", fill = "white")
)



# %%
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

# subset data for k = 0.99 and forget = 0.8
k99f08 <- minerva %>%
    filter(K == 0.99, Forget == 0.8) %>%
    filter(Experiment != "Null Model")

# make a timeout column and set it to 0 then sum the number of timeouts per item
k99f08$Timeouts <- 0
k99f08$Timeouts[k99f08$Tau == 300] <- 1


# write a csv with noo of timeouts per item by experiment
timeouts <- k99f08 %>%
    group_by(Item, Condition, Experiment) %>%
    summarise(Timeouts = sum(Timeouts))

write_csv(timeouts, "timeout-counts.csv")

# print no of idioms with timeouts
timeouts %>%
    filter(Condition == "Idiom") %>%
    filter(Experiment == "Frequency & Semantics") %>%
    group_by(Item) %>%
    summarise(Timeouts = sum(Timeouts)) %>%
    arrange(desc(Timeouts)) %>%
    xtable()

# %%
# set reference level for the condition to idiom
k99f08$Condition <- relevel(k99f08$Condition, ref = "Idiom")
fs_no_timeout_glmm_tau <- glm(Tau ~ Condition + scale(Frequency), data = k99f08 %>% filter(Tau < 300) %>% filter(Experiment == "Frequency & Semantics"), , family = Gamma(link = "identity"))
summary(fs_no_timeout_glmm_tau)

fs_timeout_glmm_tau <- glm(Tau ~ Condition + scale(Frequency), data = k99f08 %>% filter(Experiment == "Frequency & Semantics"), , family = Gamma(link = "identity"))
summary(fs_timeout_glmm_tau)

# latex table
stargazer::stargazer(fs_no_timeout_glmm_tau, fs_timeout_glmm_tau, type = "latex", title = "GLM results for Frequency & Semantics  with K = 0.99 and Forget = 0.8")

s_no_timeout_glmm_tau <- glm(Tau ~ Condition + scale(Frequency), data = k99f08 %>% filter(Tau < 300) %>% filter(Experiment == "Semantics-only"), , family = Gamma(link = "identity"))
summary(s_no_timeout_glmm_tau)

s_timeout_glmm_tau <- glm(Tau ~ Condition + scale(Frequency), data = k99f08 %>% filter(Experiment == "Semantics-only"), , family = Gamma(link = "identity"))
summary(s_timeout_glmm_tau)

stargazer::stargazer(s_no_timeout_glmm_tau, s_timeout_glmm_tau, type = "latex", title = "GLM results for Semantics Only with K = 0.99 and Forget = 0.8")

f_no_timeout_glmm_tau <- glm(Tau ~ Condition + scale(Frequency), data = k99f08 %>% filter(Tau < 300) %>% filter(Experiment == "Frequency-only"), , family = Gamma(link = "identity"))
summary(f_no_timeout_glmm_tau)

f_timeout_glmm_tau <- glm(Tau ~ Condition, data = k99f08 %>% filter(Experiment == "Frequency-only"), , family = Gamma(link = "identity"))
summary(f_timeout_glmm_tau)

stargazer::stargazer(f_no_timeout_glmm_tau, f_timeout_glmm_tau, type = "latex", title = "GLM results for Frequency Only with K = 0.99 and Forget = 0.8")

# %%


summary(lm(Frequency ~ Condition, data = k99f08 %>% filter(Experiment == "Frequency & Semantics")))

model_performance(f_no_timeout_glmm_tau)
# descriptive statistics for frequency
k99f08 %>%
    group_by(Condition) %>%
    summarise(mean = mean(Frequency), sd = sd(Frequency), median = median(Frequency), min = min(Frequency), max = max(Frequency), n = n()) %>%
    xtable()

# %%

# reorder level for the condition to compositional, collocation, idiom
k99f08$Condition <- factor(k99f08$Condition, levels = c("Compositional", "Collocation", "Idiom"))


# plots for tau. I need mean taus for each experiment with and without timeouts. Each experiement should be saved as a separate plot
plot_fs_notimeout <- k99f08 %>%
    filter(Tau < 300) %>%
    filter(Experiment == "Frequency & Semantics") %>%
    group_by(Condition) %>%
    ggplot(aes(x = Condition, y = Tau, fill = Condition)) +
    geom_bar(stat = "summary", fun = "mean", position = "dodge", color = "black", linewidth = 0.8) +
    geom_errorbar(stat = "summary", fun.data = "mean_cl_boot", position = position_dodge(width = 0.90), width = 0.25, linewidth = 0.8) +
    labs(title = "Frequency & Semantics", y = "Tau", subtitle = "Successful Retrievals") +
    theme_bw() +
    plot_theme

plot_fs_notimeout

plot_fs_timeout <- k99f08 %>%
    filter(Experiment == "Frequency & Semantics") %>%
    group_by(Condition) %>%
    ggplot(aes(x = Condition, y = Tau, fill = Condition)) +
    geom_bar(stat = "summary", fun = "mean", position = "dodge", color = "black", linewidth = 0.8) +
    geom_errorbar(stat = "summary", fun.data = "mean_cl_boot", position = position_dodge(width = 0.90), width = 0.25, linewidth = 0.8) +
    labs(y = "Tau", subtitle = "All Retrievals") +
    theme_bw() +
    plot_theme

plot_fs_timeout

# share y axis
plot_fs_notimeout + plot_fs_timeout + plot_layout(guides = "collect", axis_titles = "collect", axes = "collect_y") & theme(legend.position = "bottom")

ggsave("barplot-fs-contextual.png", width = 30, height = 20, units = "cm")

# %%
# plot for semantics only

plot_s_notimeout <- k99f08 %>%
    filter(Tau < 300) %>%
    filter(Experiment == "Semantics-only") %>%
    group_by(Condition) %>%
    ggplot(aes(x = Condition, y = Tau, fill = Condition)) +
    geom_bar(stat = "summary", fun = "mean", position = "dodge", color = "black", linewidth = 0.8) +
    geom_errorbar(stat = "summary", fun.data = "mean_cl_boot", position = position_dodge(width = 0.90), width = 0.25, linewidth = 0.8) +
    labs(title = "Semantics Only", y = "Tau", subtitle = "Successful Retrievals") +
    theme_bw() +
    plot_theme

plot_s_notimeout

plot_s_timeout <- k99f08 %>%
    filter(Experiment == "Semantics-only") %>%
    group_by(Condition) %>%
    ggplot(aes(x = Condition, y = Tau, fill = Condition)) +
    geom_bar(stat = "summary", fun = "mean", position = "dodge", color = "black", linewidth = 0.8) +
    geom_errorbar(stat = "summary", fun.data = "mean_cl_boot", position = position_dodge(width = 0.90), width = 0.25, linewidth = 0.8) +
    labs(y = "Tau", subtitle = "All Retrievals") +
    theme_bw() +
    plot_theme

plot_s_timeout

# share y axis
plot_s_notimeout + plot_s_timeout + plot_layout(guides = "collect", axis_titles = "collect", axes = "collect_y") & theme(legend.position = "bottom")

ggsave("barplot-sonly-contextual.png", width = 30, height = 20, units = "cm")


# %%

# plots for frequency only

plot_f_notimeout <- k99f08 %>%
    filter(Tau < 300) %>%
    filter(Experiment == "Frequency-only") %>%
    group_by(Condition) %>%
    ggplot(aes(x = Condition, y = Tau, fill = Condition)) +
    geom_bar(stat = "summary", fun = "mean", position = "dodge", color = "black", linewidth = 0.8) +
    geom_errorbar(stat = "summary", fun.data = "mean_cl_boot", position = position_dodge(width = 0.90), width = 0.25, linewidth = 0.8) +
    labs(title = "Frequency Only", y = "Tau", subtitle = "Successful Retrievals") +
    theme_bw() +
    plot_theme

plot_f_notimeout

plot_f_timeout <- k99f08 %>%
    filter(Experiment == "Frequency-only") %>%
    group_by(Condition) %>%
    ggplot(aes(x = Condition, y = Tau, fill = Condition)) +
    geom_bar(stat = "summary", fun = "mean", position = "dodge", color = "black", linewidth = 0.8) +
    geom_errorbar(stat = "summary", fun.data = "mean_cl_boot", position = position_dodge(width = 0.90), width = 0.25, linewidth = 0.8) +
    labs(y = "Tau", subtitle = "All Retrievals") +
    theme_bw() +
    plot_theme

plot_f_timeout

# share y axis
plot_f_notimeout + plot_f_timeout + plot_layout(guides = "collect", axis_titles = "collect", axes = "collect_y") & theme(legend.position = "bottom")

ggsave("barplot-fonly-contextual.png", width = 30, height = 20, units = "cm")

# %%

# percentage timeouts per condition and experiment
k99f08 %>%
    group_by(Condition, Experiment) %>%
    summarise(percentage = sum(Tau >= 300) / n()) %>%
    xtable()

# plot percentage timeouts

# plot for semantics only

plot_s_notimeout <- k99f08 %>%
    filter(Tau < 300) %>%
    filter(Experiment == "Semantics-only") %>%
    group_by(Condition) %>%
    ggplot(aes(x = Condition, y = Tau, fill = Condition)) +
    geom_bar(stat = "summary", fun = "mean", position = "dodge", color = "black", linewidth = 0.8) +
    geom_errorbar(stat = "summary", fun.data = "mean_cl_boot", position = position_dodge(width = 0.90), width = 0.25, linewidth = 0.8) +
    #    geom_text(stat = "summary", aes(label = round(after_stat(y),2), group = Condition), vjust = -1, size = 8, position = position_dodge(width = 0.2)) +
    labs(title = "Semantics Only", y = "Tau", subtitle = "Successful Retrievals") +
    theme_bw() +
    plot_theme

plot_s_notimeout

plot_s_timeout <- k99f08 %>%
    filter(Experiment == "Semantics-only") %>%
    group_by(Condition) %>%
    ggplot(aes(x = Condition, y = Tau, fill = Condition)) +
    geom_bar(stat = "summary", fun = "mean", position = "dodge", color = "black", linewidth = 0.8) +
    geom_errorbar(stat = "summary", fun.data = "mean_cl_boot", position = position_dodge(width = 0.90), width = 0.25, linewidth = 0.8) +
    labs(y = "Tau", subtitle = "All Retrievals") +
    theme_bw() +
    plot_theme

plot_s_timeout

# share y axis
plot_s_notimeout + plot_s_timeout + plot_layout(guides = "collect", axis_titles = "collect", axes = "collect_y") & theme(legend.position = "bottom")

ggsave("barplot-sonly-contextual.png", width = 30, height = 30, units = "cm")


ggsave("barplot-k99f08-percentage-timeouts-contextual.png", plot, width = 30, height = 20, units = "cm")

# plots for counts of timeouts
plot <- k99f08 %>%
    group_by(Condition, Experiment) %>%
    filter(Tau == 300) %>%
    ggplot(aes(x = Condition, fill = Condition)) +
    geom_bar(stat = "count", position = "dodge", color = "black", linewidth = 0.8) +
    labs(y = "Count") +
    theme_bw() +
    plot_theme +
    facet_wrap(~Experiment, ncol = 4)

plot

ggsave("barplot-k99f08-timeouts-count-contextual.png", plot, width = 30, height = 20, units = "cm")

# count of timeouts by item write to csv

k99f08 %>%
    filter(Tau == 300) %>%
    group_by(Item, Condition, Experiment) %>%
    summarise(Timeouts = n()) %>%
    xtable()

# scatterplot frequency by number of timeouts by condition for experiment 1
plot <- k99f08 %>%
    filter(Experiment == "Frequency & Semantics") %>%
    group_by(Condition, Item) %>%
    summarise(Frequency = mean(Frequency), Timeouts = sum(Tau == 300)) %>%
    ggplot(aes(x = log(Frequency), y = Timeouts, color = Condition)) +
    geom_point() +
    geom_smooth(method = "lm", se = FALSE) +
    labs(title = "Frequency & Semantics", x = "Frequency", y = "Timeouts") +
    theme_bw() +
    plot_theme +
    theme(axis.text.x = element_text(size = 20, face = "bold"), axis.text.y = element_text(size = 20, face = "bold"))


# plot vertical line at 27123
plot + geom_vline(xintercept = log(27123), linetype = "dashed", color = "red")

plot

# find most frequent items with timeouts

k99f08 %>%
    filter(Experiment == "Frequency & Semantics") %>%
    filter(Tau == 300) %>%
    group_by(Item, Frequency) %>%
    summarise(Timeouts = n()) %>%
    arrange(desc(Frequency)) %>%
    head(10) %>%
    xtable()


#



# %%
# Ablation Station


# read data
minerva <- read_csv("ablation-results\\clean\\mean_sweep.csv")
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


# %%
plot <- minerva %>%
    filter(Experiment == "Frequency & Semantics") %>%
    group_by(Condition, K, Forget) %>%
    ggplot(aes(x = Condition, y = Tau, fill = Condition)) +
    geom_bar(stat = "summary", fun = "mean", position = "dodge", color = "black", linewidth = 0.8) +
    geom_errorbar(stat = "summary", fun.data = "mean_cl_boot", position = position_dodge(width = 0.90), width = 0.25, linewidth = 0.8) +
    labs(title = "Frequency & Semantics", y = "Tau", subtitle = "Mean Across Verb and Noun") +
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
    ) +
    facet_grid(K ~ Forget, scales = "free_y")

ggsave("ablation-results\\plots\\barplot-fs-contextual-mean.png", plot, width = 30, height = 50, units = "cm")


# %%
plot <- minerva %>%
    filter(Experiment == "Frequency-only") %>%
    group_by(Condition, K, Forget) %>%
    ggplot(aes(x = Condition, y = Tau, fill = Condition)) +
    geom_bar(stat = "summary", fun = "mean", position = "dodge", color = "black", linewidth = 0.8) +
    geom_errorbar(stat = "summary", fun.data = "mean_cl_boot", position = position_dodge(width = 0.90), width = 0.25, linewidth = 0.8) +
    labs(title = "Frequency Only", y = "Tau", subtitle = "Mean Across Verb and Noun") +
    theme_bw() +
    plot_theme +
    facet_grid(K ~ Forget, scales = "free_y")

plot

ggsave("ablation-results\\plots\\barplot-fonly-contextual-mean.png", plot, width = 30, height = 50, units = "cm")

# %%
plot <- minerva %>%
    filter(Experiment == "Semantics-only") %>%
    group_by(Condition, K, Forget) %>%
    ggplot(aes(x = Condition, y = Tau, fill = Condition)) +
    geom_bar(stat = "summary", fun = "mean", position = "dodge", color = "black", linewidth = 0.8) +
    geom_errorbar(stat = "summary", fun.data = "mean_cl_boot", position = position_dodge(width = 0.90), width = 0.25, linewidth = 0.8) +
    labs(title = "Semantics Only", y = "Tau", subtitle = "Mean Across Verb and Noun") +
    theme_bw() +
    plot_theme +
    facet_grid(K ~ Forget, scales = "free_y")

ggsave("ablation-results\\plots\\barplot-sonly-contextual-mean.png", plot, width = 30, height = 50, units = "cm")


plot <- minerva %>%
    filter(Experiment == "Null Model") %>%
    group_by(Condition, K, Forget) %>%
    ggplot(aes(x = Condition, y = Tau, fill = Condition)) +
    geom_bar(stat = "summary", fun = "mean", position = "dodge", color = "black", linewidth = 0.8) +
    geom_errorbar(stat = "summary", fun.data = "mean_cl_boot", position = position_dodge(width = 0.90), width = 0.25, linewidth = 0.8) +
    labs(title = "Null Model", y = "Tau", subtitle = "With Noise Embeddings") +
    theme_bw() +
    plot_theme +
    facet_grid(K ~ Forget, scales = "free_y")

ggsave("barplot-null-contextual.png", plot, width = 30, height = 50, units = "cm")


# %%


plot <- k99f08 %>%
    group_by(Condition, Experiment) %>%
    # filter(Tau <300) %>%
    ggplot(aes(x = Condition, y = Tau, fill = Condition)) +
    geom_bar(stat = "summary", fun = "mean", position = "dodge", color = "black", linewidth = 0.8) +
    geom_errorbar(stat = "summary", fun.data = "mean_cl_boot", position = position_dodge(width = 0.90), width = 0.25, linewidth = 0.8) +
    labs(y = "Tau", subtitle = "Successful & Failed Retrievals") +
    theme_bw() +
    plot_theme +
    facet_wrap(~Experiment, ncol = 4)

plot

ggsave("barplot-k99f08-timeouts-contextual.png", plot, width = 30, height = 20, units = "cm")

# %%

# #%%
# plot <-  minerva %>%
#     filter(Experiment == "Null Model") %>%
#     group_by(Condition, K, Forget) %>%
#     ggplot(aes(x = Condition, y = Tau, fill = Condition)) +
#     geom_bar(stat = "summary", fun = "mean", position = "dodge", color = "black", linewidth = 0.8) +
#     geom_errorbar(stat = "summary", fun.data = "mean_cl_boot", position = position_dodge(width = 0.90), width = 0.25, linewidth = 0.8) +
#     labs(title = "Null Model", y = "Tau", subtitle = "With Noise Embeddings") +
#     theme_bw() +
#     theme(
#             title = element_text(size = 26, face = "bold"),
#             axis.line = element_line(colour = "black", linewidth = 0.8, lineend = "round"),
#             axis.title = element_text(size = 18, face = "bold"),
#             axis.title.x = element_blank(),
#             axis.text.y = element_text(size = 20, face = "bold"),
#             axis.text.x = element_blank(),
#             legend.title = element_blank(),
#             legend.text = element_text(size = 20, face = "bold"),
#             legend.position = "bottom",
#             strip.text.x = element_text(size = 18, face = "bold"),
#             strip.text.y = element_text(size = 18, face = "bold"),
#             strip.background = element_rect(colour = "black", fill = "white", linewidth = 0.8),
#             panel.background = element_rect(colour = "black", fill = "white", linewidth = 0.8),
#             panel.grid.major = element_line(colour = "grey", linewidth = 0.5),
#             panel.spacing = unit(0, "points"),
#             plot.background = element_rect(colour = "white", fill = "white"),
#         ) +   facet_grid(K ~ Forget, scales = "free_y")

# ggsave("ablation-results\\plots\\barplot-null-contextual.png", plot, width = 30, height = 50, units = "cm")
