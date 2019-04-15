from typing import Dict

import torch
import torch.nn.functional as F
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy

torch.manual_seed(1)


@Model.register("lab4")
class LstmTagger(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=vocab.get_vocab_size('language'))
        self.accuracy = CategoricalAccuracy()
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, characters: Dict[str, torch.Tensor],
                language: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(characters)
        embeddings = self.word_embeddings(characters)
        encoder_out = self.encoder(embeddings, mask)

        tag_logits = self.hidden2tag(encoder_out)
        class_probabilities = F.softmax(tag_logits)

        output = {"class_probabilities": class_probabilities}

        if language is not None:
            loss = self.loss(tag_logits, language)
            self.accuracy(tag_logits, language)
            output["loss"] = loss

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}
