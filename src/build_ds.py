import pandas as pd
import math
import os
import numpy

# read in the dataset
df = pd.read_csv("data/PhD_DS_Exp1.csv") 

# remove trailing whitespace in all columns
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x) 

# add string to all empty cells
df = df.fillna('none')

# extract string before whitespace from coll column and add to new column called verb
df['verb'] = df['coll'].str.split().str[0]

# rename columns 
df = df.rename(columns={'coll': 'item', 'coll_freq': 'fitem', 'coll_score': 'logDice'})

# add verb to idiom and to prod
df['idiom'] = df['verb'] + ' ' + df['idiom']
df['prod'] = df['verb'] + ' ' + df['prod']


#view idiom and prod columns
print(df[['idiom', 'prod']])

print(df.head())


# create new df using columns from old df with idiom in colname
idiom = df[[ 'idiom', 'idiom_freq', 'idiom_score']]
#create a new column called type and add the str idiom to it
idiom['type'] = 'idiom'
# rename columns
idiom = idiom.rename(columns={'idiom': 'item', 'idiom_freq': 'fitem', 'idiom_score': 'score'})

# create new df using columns from old df with prod in colname
prod = df[['prod', 'prod_freq', 'prod_score']]
#create a new column called type and add the str idiom to it
prod['type'] = 'prod'
# rename columns
prod = prod.rename(columns={'prod': 'item', 'prod_freq': 'fitem', 'prod_score': 'score'})

# create new df using columns from old df with item in colname
item = df[['item', 'fitem', 'logDice']]
#create a new column called type and add the str idiom to it
item['type'] = 'collocation'
# rename columns
item = item.rename(columns={'logDice': 'score'})

# join dfs to item by adding them to the bottom with colnames from item


new_df = pd.concat([idiom, prod, item])

# split the item into two columns, but keep it in the df
new_df['verb'] = new_df['item'].str.split().str[0]
new_df['noun'] = new_df['item'].str.split().str[1]

# write new df to csv
new_df.to_csv('data/stimuli_idioms.csv', index=False)
