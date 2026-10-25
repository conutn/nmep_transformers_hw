from .tokenizer import Tokenizer

import torch


class CharacterTokenizer(Tokenizer):
    def __init__(self, verbose: bool = False):
        """
        Initializes the CharacterTokenizer class for French to English translation.
        If verbose is True, prints out the vocabulary.

        We ignore capitalization.

        Implement the remaining parts of __init__ by building the vocab.
        Implement the two functions you defined in Tokenizer here. Once you are
        done, you should pass all the tests in test_character_tokenizer.py.
        """
        super().__init__()


        # Normally, we iterate through the dataset and find all unique characters. To simplify things,
        # we will use a fixed set of characters that we know will be present in the dataset.
        self.characters = """\t\n !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]_`abcdefghijklmnopqrstuvwxyz{|}\x80\xa0¡£§©«\xad°±²³´µ·¸¹º»¼½¾¿ÀÁÂÄÅÆÇÈÉÊÌÍÎÏÐÑÓÔÖØÙÚÜÝÞßàáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþāăąĆćĉČčďĐđēėęěğĢīİıķļĽľŁłńņňōŐőŒœŕŘřŚśŞşŠšŢţťūůųŵźŻżŽžǎǐǔșʻ˗˚ΑΒΕΙΚΜΝΟΠΡΤΥάέήίαβγδεζηικλμνξοπρςστυφχψωόύώЄАДМНОПТУабвгдежзийклмнопрстучшъыьюяєіїᵉᵐ\u200b–—―‘’“”„•…″€−○♦✓�"""
        if verbose:
            print("Vocabulary:", self.vocab)

        self.vocab = {v: k for k, v in enumerate(self.characters)}

    def encode(self, text: str) -> torch.Tensor:
        lst = [*map(self.vocab.__getitem__, text.lower())]
        return torch.tensor(lst)

    def decode(self, tokens: torch.Tensor) -> str:
        lst = tokens.tolist()
        return "".join(map(self.characters.__getitem__, lst))
