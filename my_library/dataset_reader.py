from typing import Iterator, List, Dict

from allennlp.data import DatasetReader, TokenIndexer
from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import Token


@DatasetReader.register('lab4')
class PosDatasetReader(DatasetReader):
    """
    DatasetReader for tagging data, one word per line, like:

        Khoury Arabic
        Nahas Arabic
        Daher Arabic
        Gerges Arabic
        Nazari Arabic

    """

    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def text_to_instance(self, tokens: List[Token], language: str = None) -> Instance:
        word = TextField(tokens, self.token_indexers)
        fields = {"characters": word}

        if language is not None:
            fields['language'] = LabelField(language)

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            for line in f:
                word = line.split(" ")[0]
                lang = line.split(" ")[1]
                yield self.text_to_instance([Token(char) for char in word], lang)


if __name__ == '__main__':
    reader = PosDatasetReader()
    train_dataset = reader.read('/home/eerik/Dropbox/workspace/phd/nlp/lab_4/data/names/train_set.txt')
