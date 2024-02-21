import pandas as pd

df = pd.read_csv("data/stimuli_idioms.csv")
# drop rows where verb is "defy", "freeze" or "polish"
df = df[~df.verb.isin(["defy", "freeze", "polish"])]
verb_g = df.groupby("verb")
filt = verb_g.filter(lambda g: not (g.noun == "none").any())

filt.to_csv("data/stimuli_idioms_clean.csv", index=False)
