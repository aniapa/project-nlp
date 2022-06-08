import torch

from utils import add_spaces, score_kenlm_model


def main():
    score_kenlm_model('very simple sentance .', 'kenlm/lm/test.arpa')


if __name__ == "__main__":
    main()