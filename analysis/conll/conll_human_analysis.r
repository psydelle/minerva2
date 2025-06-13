# Load tidyverse packages
getwd()
library(tidyverse)
options(digits = 3)
options(ggplot2.discrete.fill = c("#387ADF", "#FF4500", "#50C4ED"))
palette <- c("#387ADF", "#FF4500", "#50C4ED")
plot_theme <- theme(
  title = element_text(face = "bold"),
  axis.line = element_line(colour = "black", lineend = "round"),
  axis.title = element_text(face = "bold"),
  axis.title.x = element_blank(),
  axis.title.y = element_text(face = "bold"),
  axis.text.y = element_text(face = "bold"),
  axis.text.x = element_text(face = "bold"),
  legend.title = element_blank(),
  legend.text = element_text(face = "bold"),
  legend.position = "bottom",
  strip.text.x = element_text(face = "bold"),
  strip.text.y = element_text(face = "bold"),
  strip.background = element_rect(colour = "black", fill = "white", linewidth = 0.8),
  panel.background = element_rect(colour = "black", fill = "white", linewidth = 0.8),
  panel.grid.major = element_line(colour = "grey", linewidth = 0.5),
  panel.spacing = unit(0, "points"),
  plot.background = element_rect(colour = "white", fill = "white")
)

# Load additional packages
library(lme4)
library(xtable)
library(gtsummary)
library(psych)
library(lmerTest)
library(brms)


# Load the data from csv
human <- read_csv("analysis\\conll\\human_clean_for_acl_withbreakdiffuse.csv", show_col_types = FALSE)
# read data
minerva <- read_csv("minerva_contextual_clean.csv")
# SUBSET HYPEPARAMETERS OF INTEREST
k99f08 <- minerva %>% filter(K == 0.99 & Forget == 0.8)

# add col timeouts to k99f08
k99f08$Failures <- 0
k99f08$Successes <- 0
k99f08$Failures[k99f08$Tau == 300] <- 1
k99f08$Successes[k99f08$Tau < 300] <- 1


# set levels for condition


# reorder factor levels
human$Condition <- as.factor(human$Condition)
human$Condition <- factor(human$Condition, levels = c("Compositional", "Collocation", "Idiom"))
k99f08 <- k99f08 %>% mutate(Condition = factor(Condition, levels = c("Compositional", "Collocation", "Idiom")))

# set reference level for condition to Idiom
human$Condition <- relevel(human$Condition, ref = "Idiom")
k99f08$Condition <- relevel(k99f08$Condition, ref = "Idiom")

# filter out verbs with low accuracy
human <- human %>% filter(Verb != "silence")
human <- human %>% filter(Verb != "muzzle")
human <- human %>% filter(Verb != "pad")
human <- human %>% filter(Verb != "slap")

# do the same for minerva
k99f08 <- k99f08 %>% filter(Verb != "silence")
k99f08 <- k99f08 %>% filter(Verb != "muzzle")
k99f08 <- k99f08 %>% filter(Verb != "pad")
k99f08 <- k99f08 %>% filter(Verb != "slap")


human_model <- glmer(RT ~ Condition + scale(Frequency) + (1 | ID) + (1 | Verb), data = human %>% filter(Accuracy == 1), family = Gamma(link = "identity"))
summary(human_model)

human_model_acc <- glmer(Accuracy ~ Condition + scale(Frequency) + (1 | ID) + (1 | Verb), data = human, family = binomial(link = "logit"))
summary(human_model_acc)

stargazer::stargazer(human_model_acc, type = "latex")

emmeans::emmeans(human_model_acc, pairwise ~ Condition, adjust = "tukey", type = "response")

# number of items above 28500 frequency by condition table

human %>%
  filter(Frequency > 28000) %>%
  group_by(Condition) %>%
  distinct(Item) %>%
  summarise(n = n()) %>%
  xtable(caption = "Number of Items above 28500 Frequency by Condition", digits = 2)

# human rt by condition table
human %>%
  filter(Accuracy == 1) %>%
  group_by(Condition) %>%
  summarise(mean_rt = mean(RT), sd_rt = sd(RT), n = n()) %>%
  xtable(caption = "Human RT by Condition", digits = 2)

# human accuracy by condition table
human %>%
  group_by(Condition) %>%
  summarise(mean_accuracy = mean(Accuracy), sd_accuracy = sd(Accuracy), n = n()) %>%
  mutate(se = sd_accuracy / sqrt(n)) %>%
  xtable(caption = "Human Accuracy by Condition", digits = 2)

# frequency by condition table

human %>%
  group_by(Condition) %>%
  distinct(Frequency) %>%
  summarise(mean_frequency = mean(Frequency), sd_frequency = sd(Frequency), n = n()) %>%
  mutate(se = sd_frequency / sqrt(n)) %>%
  xtable(caption = "Human Frequency by Condition", digits = 2)

minerva_model <- glmer(Tau ~ Condition + scale(Frequency) + (1 | Verb), data = k99f08 %>% filter(Experiment == "Frequency & Semantics") %>% filter(Tau < 300), family = Gamma(link = "identity"))
summary(minerva_model)

stargazer::stargazer(human_model, minerva_model, type = "latex")

k99f08 <- k99f08 %>%
  mutate(Frequency_centered = Frequency - mean(Frequency, na.rm = TRUE))

m_to <- glm(Successes ~ Condition + Frequency_centered, data = k99f08 %>% filter(Experiment == "Frequency & Semantics"), family = binomial(link = "logit"))
summary(m_to)

# exponentiate coefs to get odds ratios
exp(coef(m_to))


m_to_scale <- glm(Successes ~ Condition + scale(Frequency), data = k99f08 %>% filter(Experiment == "Frequency & Semantics"), family = binomial(link = "logit"))
summary(m_to_scale)

m_to_no_freq <- glm(Successes ~ Condition, data = k99f08 %>% filter(Experiment == "Frequency & Semantics"), family = binomial(link = "logit"))
summary(m_to_no_freq)

# anova
anova(m_to_no_freq, m_to_scale)

pairs(emm)

m_to_log <- glm(Successes ~ Condition + log(Frequency), data = k99f08 %>% filter(Experiment == "Frequency & Semantics"), family = binomial(link = "logit"))
summary(m_to_log)

human_model_freq <- lmer(RT ~ Condition + scale(Frequency) + (1 | ID) + (1 | Verb), data = human %>% filter(Accuracy == 1) %>% filter(Frequency > 28000))
summary(human_model_freq)


# succesful retrieval by condition

k99f08 %>%
  filter(Experiment == "Frequency & Semantics") %>%
  group_by(Condition) %>%
  summarise(successes = sum(Successes), failures = sum(Failures)) %>%
  mutate(total = successes + failures, success_rate = successes / total)


# no of items successfully retrieved by condition

k99f08 %>%
  filter(Experiment == "Frequency & Semantics") %>%
  group_by(Condition, Item) %>%
  summarise(successes = sum(Successes), failures = sum(Failures)) %>%
  mutate(total = successes + failures, success_rate = successes / total) %>%
  filter(success_rate > 0) %>%
  ggplot(aes(x = Condition, y = success_rate, fill = Condition)) +
  geom_boxplot() +
  geom_jitter(width = 0.2, height = 0, alpha = 0.5) +
  plot_theme +
  theme(legend.position = "none") +
  labs(title = "Success Rate by Condition", x = "Condition", y = "Success Rate")

# minerva accuracy by item

# find items with low accuracy
bad_items <- k99f08 %>%
  filter(Experiment == "Frequency & Semantics") %>%
  group_by(Condition, Item, Verb) %>%
  summarise(mean_accuracy = mean(Successes)) %>%
  filter(mean_accuracy < 0.2) %>%
  arrange(mean_accuracy, by_group = TRUE)

bad_items

table(bad_items$Condition)
table(bad_items$Verb) %>% sort(decreasing = TRUE)
# plot model

plot(human_model_freq)

k99f08_highfreq_nooutliers <- k99f08 %>%
  filter(Experiment == "Frequency & Semantics") %>%
  filter(Frequency > 28500) %>%
  filter(Tau < 300)
head(k99f08_highfreq_nooutliers)
minerva_model_freq <- lm(Tau ~ Condition + scale(Frequency), data = k99f08_highfreq_nooutliers)
summary(minerva_model_freq)

# scatterplot of tau
k99f08_highfreq_nooutliers %>%
  filter(Experiment == "Frequency & Semantics") %>%
  filter(Frequency > 28500) %>%
  ggplot(aes(x = Frequency, y = Tau, color = Condition)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", se = FALSE, color = "black", linewidth = 1) +
  plot_theme +
  theme(legend.position = "none") +
  labs(title = "Scatterplot of Tau and Frequency", x = "Frequency", y = "Tau") +
  scale_color_manual(values = palette)

# scatterplot x-axis frequency y-axis no of timeouts by condition

k99f08 %>%
  filter(Experiment == "Frequency & Semantics") %>%
  ggplot(aes(x = Frequency, y = Successes, color = Condition)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", se = FALSE, color = "black", linewidth = 1) +
  plot_theme +
  theme(legend.position = "none") +
  labs(title = "Scatterplot of Successes and Frequency", x = "Frequency", y = "Successes") +
  scale_color_manual(values = palette)



m_human <- glmer(RT ~ Condition + log(Frequency) + (1 | ID) + (1 | Verb), data = human %>% filter(Accuracy == 1), family = Gamma(link = "identity"))

summary(m_human)

m1 <- glmer(RT ~ Condition * scale(Frequency) + (1 | ID), data = human %>% filter(Accuracy == 1) %>% filter(Frequency > 28500), family = Gamma(link = "identity"))
summary(m1)

m2 <- glmer(RT ~ scale(Frequency) + (1 | ID), data = human %>% filter(Accuracy == 1) %>% filter(Frequency > 28500), family = Gamma(link = "identity"))
summary(m2)

anova(m2, m1)


m3 <- lmer(Tau ~ Condition + scale(Frequency) + (1 | ID), data = k99f08 %>% filter(Experiment == "Frequency & Semantics") %>% filter(Frequency > 28500))
summary(m3)

m4 <- lm(Tau ~ scale(Frequency), data = k99f08 %>% filter(Experiment == "Frequency & Semantics") %>% filter(Frequency > 28500))
summary(m4)

anova(m4, m3)


# anova for m3
summary(aov(Tau ~ Condition + scale(Frequency) + Error(ID / Verb), data = k99f08 %>% filter(Experiment == "Frequency & Semantics") %>% filter(Frequency > 40000)))


# plot sd of Tau by Id
k99f08 %>%
  filter(Experiment == "Frequency & Semantics") %>%
  filter(Frequency > 27123) %>%
  group_by(ID) %>%
  summarise(sd = sd(Tau)) %>%
  ggplot(aes(x = ID, y = sd)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE)


# plot sd of Tau for each ID by item

k99f08 %>%
  filter(Experiment == "Frequency & Semantics") %>%
  group_by(Item, Condition) %>%
  summarise(sd = sd(Tau)) %>%
  ggplot(aes(x = Item, y = sd, color = Condition)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  theme(legend.position = "none")


# plot sd of Tau for each ID
k99f08 %>%
  filter(Experiment == "Frequency & Semantics") %>%
  group_by(ID) %>%
  summarise(sd = sd(Tau)) %>%
  ggplot(aes(x = ID, y = sd)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE)


m4 <- lmer(Tau ~ scale(Frequency) + (1 | ID), data = k99f08 %>% filter(Experiment == "Frequency & Semantics") %>% filter(Frequency > 28500))

summary(m4)

m5 <- lm(Tau ~ Condition * scale(Frequency), data = k99f08 %>% filter(Experiment == "Frequency & Semantics") %>% filter(Frequency > 28500))

summary(m5)
#----------------------------------------------------------------------------
# order factor levels
human$Condition <- factor(human$Condition, levels = c("Compositional", "Collocation", "Idiom"))
k99f08 <- k99f08 %>% mutate(Condition = factor(Condition, levels = c("Compositional", "Collocation", "Idiom")))



human %>% ggplot(aes(x = scale(Frequency), y = RT, color = Condition)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", se = FALSE, color = "black", linewidth = 1) +
  plot_theme +
  theme(legend.position = "none") +
  theme(axis.text.x = element_text()) +
  labs(title = "Scatterplot of RT and Frequency", x = "log(Frequency)", y = "Reaction Time (ms)") +
  scale_color_manual(values = palette)


# Calculate quantiles for both RT and Frequency within each Condition
quantiles_df <- human %>%
  group_by(Condition) %>%
  summarise(
    quantile = seq(0.1, 0.9, by = 0.1),
    RT_quantile = quantile(RT, probs = seq(0.1, 0.9, by = 0.1)),
    Frequency_quantile = quantile(Frequency, probs = seq(0.1, 0.9, by = 0.1)),
    .groups = "drop"
  )

# Plot with Conditions on x-axis, RT on y-axis, and lines for each quantile
quantiles_human <- ggplot(quantiles_df, aes(x = Condition, y = RT_quantile, group = quantile, color = "black")) +
  geom_line(linewidth = 1, color = "black") + # Draw lines connecting the same quantiles across Conditions
  geom_point(shape = 21, size = 4, color = "black", aes(fill = Condition)) + # Add points at each quantile
  labs(
    x = "Condition",
    y = "Reaction Time (ms)",
    color = "Quantile",
  ) +
  plot_theme + # set palette for condition
  scale_color_manual(values = palette) +
  theme(legend.position = "none") +
  theme(axis.text.x = element_text(size = 20, face = "bold"))

quantiles_human

ggsave("analysis\\conll\\quantile_plot_human.png", width = 6, height = 4, dpi = 300)


# Plot with Conditions on x-axis, RT on y-axis, and lines for each quantile
ggplot(quantiles_df, aes(x = Condition, y = Frequency_quantile, group = quantile, color = factor(quantile))) +
  geom_line(linewidth = 1, color = "black") + # Draw lines connecting the same quantiles across Conditions
  geom_point(shape = 21, size = 4, color = "black", aes(fill = Condition)) + # Add points at each quantile
  labs(
    x = "Condition",
    y = "Frequency",
    color = "Quantile",
  ) +
  plot_theme + # set palette for condition
  scale_color_manual(values = palette) +
  theme(legend.position = "right")

ggsave("analysis\\conll\\quantile_plot_frequency.png", width = 6, height = 4, dpi = 300)

quantiles_minerva <- k99f08 %>%
  filter(Experiment == "Frequency & Semantics") %>%
  filter(Tau < 300) %>%
  group_by(Condition) %>%
  summarise(
    quantile = seq(0.1, 0.9, by = 0.1),
    Tau_quantile = quantile(Tau, probs = seq(0.1, 0.9, by = 0.1)),
    Frequency_quantile = quantile(Frequency, probs = seq(0.1, 0.9, by = 0.1)),
    .groups = "drop"
  )


# Plot with Conditions on x-axis, RT on y-axis, and lines for each quantile
ggplot(quantiles_minerva, aes(x = Condition, y = Tau_quantile, group = quantile, color = "black")) +
  geom_line(linewidth = 1, color = "black") + # Draw lines connecting the same quantiles across Conditions
  geom_point(shape = 21, size = 3, color = "black", aes(fill = Condition)) + # Add points at each quantile
  labs(
    x = "Condition",
    y = "Tau",
    color = "Quantile",
  ) +
  plot_theme + # set palette for condition
  scale_color_manual(values = palette) +
  theme(legend.position = "none")

ggsave("analysis\\conll\\quantile_plot_minerva.png", width = 6, height = 4, dpi = 300)


human_rt <- human %>%
  filter(Accuracy == 1) %>% # only correct trials
  filter(Condition != "Baseline") %>% # exclude baselines
  ggplot(aes(x = Condition, y = RT, fill = Condition)) +
  geom_bar(stat = "summary", fun = "mean", position = "dodge", color = "black", linewidth = 0.8) +
  geom_errorbar(stat = "summary", fun.data = "mean_cl_boot", position = position_dodge(width = 0.90), width = 0.25, linewidth = 0.8) +
  geom_text(
    stat = "summary", fun = mean, aes(label = round(..y.., 1)),
    vjust = 4, size = 10, fontface = "bold"
  ) +
  labs(y = "Reaction Time (ms)") +
  theme_bw() +
  plot_theme +
  theme(axis.text.x = element_text(size = 20, face = "bold")) +
  coord_cartesian(ylim = c(750, 1050))

human_rt

human_rt + quantiles_human + plot_layout(guides = "collect", axis_titles = "collect", axes = "collect_x") & theme(legend.position = "none")

ggsave("analysis\\conll\\human_rt.png", width = 15, height = 10, units = "in", dpi = 300)
