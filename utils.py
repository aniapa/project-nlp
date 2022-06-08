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


def score_kenlm_model(sentence, model_path):
    model = kenlm.Model(model_path)
    print(model.score(sentence))