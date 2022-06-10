import torch
import kenlm
import torchvision.datasets
from torch.utils.data import DataLoader


def add_spaces(file_in, file_out):
    '''
        Original data are full sentences.
        KenLM requires adding space after each word.
        We treat punctuation marks as words. 
    '''
    marks = '!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~' # Does not treat ... as one mark.
    f = open(file_in)
    text = f.read()
    for mark in marks:
        text = text.replace(mark, ' '+mark+' ')
    lines = text.split('\n')
    for i, line in enumerate(lines):
        lines[i] = ' '.join(line.split()) # remove multiple spaces
    text = '\n'.join(lines)
    f.close()
    f = open(file_out, "w")
    f.write(text)
    f.close()


def get_kenlm_model(model_path):
    return kenlm.Model(model_path)

def kenlm_model_perplexity(test_file, model):
    '''
        Calculate the perplexity of the KenLM model on the test_file.
    '''
    f = open(test_file)
    text = f.read()
    f.close()
    return model.perplexity(text)


def kenlm_model_oov_fraction(test_file, model):
    '''
        Calculate the number of oov divided by the number of all words in the corpus.
    '''
    f = open(test_file)
    text = f.read()
    f.close()
    oov_count = 0
    count = 0
    for score in model.full_scores(text):
        if score[2]:
            oov_count += 1
        count += 1
    return oov_count/count