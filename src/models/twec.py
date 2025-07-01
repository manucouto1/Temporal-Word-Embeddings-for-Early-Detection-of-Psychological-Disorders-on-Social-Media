from gensim.models.word2vec import Word2Vec, LineSentence, PathLineSentences

from gensim import utils
import os
import numpy as np
import logging
import copy
from gensim.utils import tokenize
import multiprocessing
from tqdm import tqdm
from gensim.models import callbacks
from itertools import chain


class MyCallback(callbacks.CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 0:
            print("Pérdida después de la época {}: {}".format(self.epoch, loss))
        else:
            print(
                "Pérdida después de la época {}: {}".format(
                    self.epoch, loss - self.loss_previous_step
                )
            )
        self.epoch += 1
        self.loss_previous_step = loss


class TWEC:
    """
    Handles alignment between multiple slices of temporal text
    """

    def __init__(
        self,
        size=100,
        sg=0,
        siter=10,
        ns=10,
        window=5,
        alpha=0.025,
        min_count=5,
        workers=2,
        test="test",
        init_mode="hidden",
    ):
        """

        :param size: Number of dimensions. Default is 100.
        :param sg: Neural architecture of Word2vec. Default is CBOW (). If 1, Skip-gram is employed.
        :param siter: Number of static iterations (epochs). Default is 5.
        :param diter: Number of dynamic iterations (epochs). Default is 5.
        :param ns: Number of negative sampling examples. Default is 10, min is 1.
        :param window: Size of the context window (left and right). Default is 5 (5 left + 5 right).
        :param alpha: Initial learning rate. Default is 0.025.
        :param min_count: Min frequency for words over the entire corpus. Default is 5.
        :param workers: Number of worker threads. Default is 2.
        :param test: Folder name of the diachronic corpus files for testing.
        :param init_mode: If \"hidden\" (default), initialize temporal models with hidden embeddings of the context;'
                            'if \"both\", initilize also the word embeddings;'
                            'if \"copy\", temporal models are initiliazed as a copy of the context model
                            (same vocabulary)
        """
        self.size = size
        self.sg = sg
        self.trained_slices = dict()
        self.gvocab = []
        self.epoch = siter
        self.negative = ns
        self.window = window
        self.static_alpha = alpha
        self.dynamic_alpha = alpha
        self.min_count = min_count
        self.workers = multiprocessing.cpu_count() - 3
        self.test = test
        self.init_mode = init_mode
        self.compass: None | Word2Vec = None

    def initialize_from_compass(self, model) -> Word2Vec:
        if self.compass is None:
            raise Exception("Compass model is not initialized")

        if self.init_mode == "copy":
            model = copy.deepcopy(self.compass)
        else:
            if self.compass.layer1_size != self.size:  # type: ignore
                raise Exception("Compass and Slice have different vector sizes")

            if len(model.wv.index_to_key) == 0:
                model.build_vocab(corpus_iterable=self.compass.wv.index_to_key)  # type: ignore

            vocab_m = model.wv.index_to_key

            indices = [
                self.compass.wv.key_to_index[w]
                for w in vocab_m
                if w in self.compass.wv.key_to_index
            ]
            new_syn1neg = np.array([self.compass.syn1neg[index] for index in indices])
            model.syn1neg = new_syn1neg

            if self.init_mode == "both":
                new_syn0 = np.array([self.compass.wv.syn0[index] for index in indices])  # type: ignore
                model.wv.syn0 = new_syn0

        model.learn_hidden = False  # type: ignore
        model.alpha = self.dynamic_alpha
        return model

    def internal_trimming_rule(self, word, count, min_count):
        """
        Internal rule used to trim words
        :param word:
        :return:
        """
        if word in self.gvocab:
            return utils.RULE_KEEP
        else:
            return utils.RULE_DISCARD

    def train_model(self, sentences) -> Word2Vec | None:
        model = None
        if self.compass is None or self.init_mode != "copy":
            model = Word2Vec(
                sg=self.sg,
                vector_size=self.size,
                alpha=self.static_alpha,
                negative=self.negative,
                window=self.window,
                min_count=self.min_count,
                workers=self.workers,
            )
            model.build_vocab(
                corpus_iterable=sentences,
                trim_rule=self.internal_trimming_rule
                if self.compass is not None
                else None,
            )

        if self.compass is not None:
            model = self.initialize_from_compass(model)
            model.train(
                corpus_iterable=sentences,
                total_words=sum([len(s) for s in sentences]),
                epochs=self.epoch,
                compute_loss=True,
            )
        else:
            model.train(  # type: ignore
                corpus_iterable=sentences,
                total_words=sum([len(s) for s in sentences]),
                epochs=self.epoch,
                compute_loss=True,
                callbacks=[MyCallback()],
            )

        return model

    def train_compass(self, chunks):
        texts = list(chain(*chunks))
        sentences = [
            list(tokenize(str(text), lowercase=True, deacc=True))
            for text in tqdm(texts, desc="Preparing full corpus")
        ]
        print("Training the compass.")
        self.compass = self.train_model(sentences)
        self.gvocab = self.compass.wv.index_to_key  # type: ignore

    def train_slice(self, chunks):
        if self.compass is None:
            return Exception("Missing Compass")

        sentences = [
            list(tokenize(str(text), lowercase=True, deacc=True)) for text in chunks
        ]
        model = self.train_model(sentences)
        return model

    # FINE TUNNING VARIATION

    def finetune_model(self, sentences, pretrained_path):
        model = None
        if self.compass is None or self.init_mode != "copy":
            model = Word2Vec(
                sg=self.sg,
                vector_size=self.size,
                alpha=self.static_alpha,
                negative=self.negative,
                window=self.window,
                min_count=self.min_count,
                workers=self.workers,
            )
            model.build_vocab(
                sentences,
                trim_rule=self.internal_trimming_rule
                if self.compass is not None
                else None,
            )
            # model.build_vocab(list(pretrained_model.vocab.keys()), update=True)
            model.intersect_word2vec_format(pretrained_path, binary=True, lockf=1.0)  # type: ignore

        if self.compass is not None:
            model = self.initialize_from_compass(model)

        model.train(  # type: ignore
            sentences,
            total_words=sum([len(s) for s in sentences]),
            epochs=self.epoch,
            compute_loss=True,
        )

        return model

    def finetune_compass(self, compass_text, pre_path, overwrite=False, save=True):
        sentences = PathLineSentences(compass_text)
        sentences.input_files = [
            s for s in sentences.input_files if not os.path.basename(s).startswith(".")
        ]
        logging.info("Finetunning the compass.")
        self.compass = self.finetune_model(sentences, pre_path)

        self.gvocab = self.compass.wv.index_to_key  # type: ignore

    def finetune_slice(self, slice_text, pretrained):
        try:
            if self.compass is None:
                logging.info("Fuck where is the dam compass")
                return Exception("Missing Compass")
            logging.info(
                "Finetunning temporal embeddings: slice {}.".format(slice_text)
            )

            sentences = LineSentence(slice_text)
            model = self.finetune_model(sentences, pretrained)
            return model
        except Exception as fk:
            logging.error("What da > {}".format(fk))
