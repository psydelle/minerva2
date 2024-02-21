# Minerva2 experiments

## Preparing data
1. Ensure you have a `.env` with the properties `SE_USERNAME` and `SE_API_KEY`, containing your
SketchEngine username and API key, respectively.

2. Clean the stimulus corpus
```bash
python src/prep_dataset.py
```

3. Build context sentences for each stimulus
```bash
python src/find_kwics_for_csv.py -f data/stimuli_idioms_clean.csv -o data/stimuli_idioms_kwics.json
```

## Running experiments
```bash
python run_all_experiments...
```


## Miscellanea
Cleaning the original `stimuli_idioms.csv`:
```python
import pandas as pd

verb_g = pd.read_csv("data/stimuli_idioms.csv").groupby("verb")
filt = verb_g.filter(lambda g: not (g.noun == "none").any())
filt.to_csv("data/stimuli_idioms_clean.csv", index=False)
```
