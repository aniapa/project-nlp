Baseline project: https://github.com/rnoxy/pytorch-nn-baseline
Data: http://2018.poleval.pl/index.php/tasks/, task3
Baseline model: 
3-gram created using `lmplz` tool from https://github.com/kpu/kenlm.
`lmplz -o 3 -S 80% /tmp <data/task3_train.txt > models/3gram.arpa`
This model is almost 9GB - too large to put in this repository. 
The calculated perplexity equals `140.0513`.
Fraction of OOV in the test corpus: `0.0039265789502006525`.
This is different than the count of another model submitted for the PolEval, which was equal to `0.00868323449305339` (https://github.com/kwrobel-nlp/lm/).