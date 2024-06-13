import pandas as pd

# read csv
df = pd.read_csv("data\minerva-baselines.csv")
stimuli = pd.read_csv("data\stimuli_idioms_clean.csv")

# check if duplicate verbs and nouns are present in both dataframes
df_verb = df['verb'].unique()
df_noun = df['noun'].unique()
stimuli_verb = stimuli['verb'].unique()
stimuli_noun = stimuli['noun'].unique()

# check if there are any duplicate verbs and nouns
print("Duplicate verbs: ", set(df_verb).intersection(stimuli_verb))
print("Duplicate nouns: ", set(df_noun).intersection(stimuli_noun))

#drop first column
df = df.drop(df.columns[0], axis=1)


# write to json
df.to_json("data\minerva-baselines.json", orient='records')





















# # Read the JSON file into a DataFrame
# df = pd.read_json("data\minerva-baselines.json")

# # add a new column to the DataFrame called item_grammatical by concat verb and noun
# df['item_grammatical'] = df['verb'] + " " + df['noun']

# # write to csv
# df.to_csv("data\minerva-baselines.csv")
