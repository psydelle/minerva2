
1. People show pattern for AJT
2. If so, this is a function of frequency and item encoding (semantics)
3. Therefore, we use minerva which is sensitive to freq and semantics

Computational findings:
1. Minerva distribution for successful retrievals is a bad match to human RTs
=> it's not just memory
2. Fewer errors for i. than c and p
2.1 Look for hyperparameter config. in which the model's error rate for idioms matched that of humans (very low)
2.2 For this config, does the tau distribution for successful retrievals match human RTs? (probably no)
2.3 For this config, do idioms have the lowest tau for successful retrievals? (probably yes)
=> We can match the idiom data with just a memory model. However, can't match other item types. We posit that for other item types, humans have other processes which 1) interpret the item and 2) yield longer reaction times. This is supported by longer human RTs for these item types.

Further hypotheses:
3. For config of (2), better performance for idioms (in RT and error rate) is driven by semantics, not frequency. Prediction: this will also hold true for humans. 
3.1 Evidence: In both humans and Minerva, controlling for frequency, idioms have lower RTs than c and p.



Conclusions:
1. Idioms are more modelable as memory process than c. and prod.
