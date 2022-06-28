# Polish Language Modeling - Report
### Kamil Michalak, Jonatan Hrynko, Anna Pacanowska
## Project repository
https://github.com/aniapa/project-nlp

## Project description
In the 2018 PolEval, one of the tasks (http://2018.poleval.pl/index.php/tasks/ - task 3) was creating a language model for Polish. Since that time there were significant advances in NLP, so we wanted to try to get better results by using the newest techniques, such as transformers.
## Data
We used training and test datasets provided by 2018 PolEval.
## 3-gram
As a baseline model, we used a simple 3-gram.
We created it using lmplz tool from <https://github.com/kpu/kenlm>.
`lmplz -o 3 -S 80% /tmp <data/task3_train.txt > models/3gram.arpa`
The calculated perplexity equals 140.0513.
Fraction of OOV in the test corpus: 0.0039265789502006525.
This is different than the count of another model submitted for the PolEval, which was equal to 0.00868323449305339 (https://github.com/kwrobel-nlp/lm/).
## RNN
We implemented basic RNN using `torch.nn.RNN` module. We tried 2 metods of tokenization
* word tokenization with `gensim.utils.simple_preprocess`
* subword tokenization with SentencePiece (with Unigram algorithm)

Word tokenization worked fine for small dataset (1000 rows), underlined sentence is from training set
![](https://i.imgur.com/yddceId.png)

For larger dataset (1M rows) vocabulary size was too large (0.5M words) for our GPU memory, so we couldn't train it.

Subword tokenization, with vocabulary size: 8000, solved memory problem.

RNN parameters:
```
RNN(
  (embedding): Embedding(8000, 100, padding_idx=0)
  (rnn): RNN(100, 128, num_layers=2, batch_first=True)
  (fc): Linear(in_features=128, out_features=8000,     bias=True)
)
```
We used 1M rows for traning and 100000 rows for testing.

Training parameters:
* optimizer: Adam
* learning rate: 0.001
* batch size: 20
* epochs: 30

We achieved the best result after 9 epochs.

**Perplexity:**
* Training set: 127.9
* Test set: 132.5

![](https://i.imgur.com/4eEm5cv.png)


## GPT2 
### From scratch
We used huggingface implementation (https://huggingface.co/docs/transformers/model_doc/gpt2). 
First, we tried training a model from a scratch. As expected, untrained model had huge perplexity and generated random texts:
![](https://i.imgur.com/gtyI8K0.png)

![](https://i.imgur.com/e08sf1R.png)

After training on the first 200000 lines of the training dataset for 75000 steps, it improved:
![](https://i.imgur.com/sIAXMup.png)
### Pretrained
First, we tried this model: https://huggingface.co/flax-community/papuGaPT2
It gave some nice results:
![](https://i.imgur.com/0S19CbQ.png)
![](https://i.imgur.com/zP1J0Za.png)

Then we tried to fine-tune it using the same subset for 28000 (afterwards the loss stopped decreasing, so we stopped the runtime). Surprisingly, instead of improving, the model performance significantly decreased:
![](https://i.imgur.com/sZ3Sjr3.png)


## Problems
Lack of resources was one of the biggest problems. The datasets from PolEval turned out to be huge - it was not possible to run the computations on the entire files. We had additional problems with training GPT2 locally, so we had to use Colab. We probably should have thought about that earlier and make sure to get access to some better computations.
The perplexity from GPT2 is not comparable to the ones from PolEval -
it uses a byte-pair encoding tokenizer, which does not map one word to one token. The vocabulary size is completely different because of this, as well as different datasets used.
## Conclusions
Training language models, especially using transformers takes a lot of resources. It is also not always obvious how to compare different models.
