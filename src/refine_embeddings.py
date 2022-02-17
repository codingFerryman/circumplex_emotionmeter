import codecs

import pandas as pd
import numpy as np
from tqdm import trange
from utils import get_logger
import gensim
import spacy

logger = get_logger("refine_embeddings", True)

# https://aclanthology.org/D17-1056.pdf
# https://github.com/wangjin0818/word_embedding_refine/blob/master/embedding_refine.py


class EmbeddingsCooker:
    def __init__(
            self,
            w2v_model_path,
            lexicon_path,
            topn=10,
            epoch=100,
            beta=0.1,
            gamma=0.1
    ):
        self.w2v_model = gensim.models.Word2Vec.load(w2v_model_path)
        self.embedding_dim = self.w2v_model.vector_size
        logger.info('w2v_model loaded!')

        self.lexicon_path = lexicon_path
        self.max_score = None

        self.topn = topn
        self.epoch = epoch
        self.beta = beta
        self.gamma = gamma

    def load_lexicon(self,
                     path=None,
                     valence_col='ValMn',
                     arousal_col='AroMn',
                     dominance_col='DomMn',
                     **kwargs):
        """
        Import lexicon data
        :param path: the path of the lexicon file
        :param valence_col: the name of valence column
        :param arousal_col: the name of arousal column
        :param dominance_col: the name of dominance column
        :param kwargs: parameters passed to pd.read_csv()
        :return:
        """
        if path is None:
            _path = self.lexicon_path
        else:
            _path = path
        if 'ANEW2017' in _path:
            kwargs['sep'] = '\t'
            kwargs['index_col'] = 0
            self.prefix = 'anew_'
            self.max_score = 9
        elif 'NRC' in _path:
            kwargs['sep'] = '\t'
            kwargs['index_col'] = 0
            kwargs['names'] = ['Word', 'ValMn', 'AroMn', 'DomMn']
            self.prefix = 'nrc_'
            self.max_score = 1
        else:
            raise FileNotFoundError
        self.lexicon_df = pd.read_csv(_path, **kwargs)
        self.valence = self.lexicon_df[valence_col]
        self.arousal = self.lexicon_df[arousal_col]
        self.dominance = self.lexicon_df[dominance_col]
        logger.info(f'Lexicon loaded {_path}')

    def refine_embeddings(self):
        assert self.lexicon_df is not None
        assert self.max_score is not None

        valence_dict = {}
        arousal_dict = {}
        dominance_dict = {}

        self.vector_dict = {}

        for w in self.lexicon_df.index:
            if w in self.w2v_model.wv.index_to_key:
                self.vector_dict[w] = self.w2v_model.wv[w]
                valence_dict[w] = self.valence[w]
                arousal_dict[w] = self.arousal[w]
                dominance_dict[w] = self.dominance[w]
        self._refine_embeddings(valence_dict, self.prefix+'valence')
        self._refine_embeddings(arousal_dict, self.prefix+'arousal')
        self._refine_embeddings(dominance_dict, self.prefix+'dominance')


    def _refine_embeddings(self, lexicon_dict, dimension_name, w2v_model=None):
        if w2v_model is None:
            w2v_model = self.w2v_model.wv
        _words = list(lexicon_dict.keys())

        logger.debug('weight_dict')
        weight_dict = {}
        for i in _words:
            for j in _words:
                weight = self.max_score - abs(lexicon_dict[i] - lexicon_dict[j])
                update_dict(weight_dict, i, j, weight)

        # weighted matrix
        logger.debug('weight_matrix')
        weight_matrix = np.zeros((len(lexicon_dict), len(lexicon_dict)))
        for i in range(len(_words)):
            word_i = _words[i]
            sim_dict = most_similar(word_i, w2v_model, weight_dict, top=self.topn)
            for j in range(len(_words)):
                word_j = _words[j]
                if word_j in sim_dict.keys():
                    weight_matrix[i][j] = sim_dict[word_j]
                    weight_matrix[j][i] = sim_dict[word_j]

        # vertex matrix
        logger.debug('vertex_matrix')
        vertex_matrix = np.zeros((len(_words), self.embedding_dim))
        for i in range(vertex_matrix.shape[0]):
            for j in range(vertex_matrix.shape[1]):
                vector = self.vector_dict[_words[i]]
                vertex_matrix[i, j] = vector[j]

        # starting refinement
        logger.info(f'starting refinement: {dimension_name}')
        # origin_vertex_matrix = vertex_matrix
        # pre_vertex_matrix = vertex_matrix
        pre_distance = 0.0
        # diff = 1.0
        num_word = len(_words)

        tr = trange(self.epoch, desc='cost: ', leave=True)
        for _ in tr:
            pre_vertex_matrix = vertex_matrix.copy()
            for i in range(num_word):
                # denominator = 0.0
                # molecule = 0.0
                tmp_vertex = np.zeros((self.embedding_dim,))
                weight_sum = 0.0
                for j in range(num_word):
                    w_multi_v = weight_matrix[i, j] * pre_vertex_matrix[j]
                    weight_sum = weight_sum + weight_matrix[i, j]
                    tmp_vertex = tmp_vertex + w_multi_v

                molecule = self.gamma * pre_vertex_matrix[i] + self.beta * tmp_vertex
                denominator = self.gamma + self.beta * weight_sum
                delta = molecule / denominator
                vertex_matrix[i] = delta
            distance = vertex_matrix - pre_vertex_matrix
            value = np.dot(distance, distance.T)

            ec_distance = 0.0
            for i in range(self.embedding_dim):
                ec_distance = ec_distance + value[i, i]

            diff = abs(ec_distance - pre_distance)
            tr.set_description("cost: {:.5f}".format(diff), refresh=True)
            pre_distance = ec_distance
        refine_vector_file = dimension_name + '.wv'
        write_vector(refine_vector_file, w2v_model, vertex_matrix, _words)


def update_dict(target_dict, key_a, key_b, val):
    if key_a in target_dict:
        target_dict[key_a].update({key_b: val})
    else:
        target_dict.update({key_a: {key_b: val}})


def write_vector(file_name, w2v_model, vertex_matrix, words):
    with codecs.open(file_name, 'w', 'utf8') as my_file:
        # refine result
        for i in range(len(words)):
            word = words[i]
            vec = vertex_matrix[i]
            my_file.write('%s %s\n' % (word, ' '.join('%f' % val for val in vec)))

        for word in w2v_model.index_to_key:
            if word not in words:
                vec = w2v_model[word]
                my_file.write('%s %s\n' % (word, ' '.join('%f' % val for val in vec)))


def most_similar(word, w2v_model, weight_dict, top=10):
    sim_array = []
    word_array = []

    # get the most similar words from word2vec model
    similar_words = w2v_model.most_similar(word, topn=top)

    i = 0
    for similar_word in similar_words:
        try:
            diff = weight_dict[word][similar_word[0]]
            sim_array.append([i, diff])
        except:
            sim_array.append([i, 0.0])

        word_array.append(similar_word[0])
        i = i + 1

    sim_array = np.array(sim_array)
    sort_index = sim_array[:, 1].argsort(0)
    new_array = sim_array[sort_index][::-1]

    ret_dict = {}
    for i in range(top):
        word = word_array[int(new_array[i][0])]
        ret_dict[word] = 1. / float(i + 1.)

    return ret_dict


if __name__ == '__main__':
    # cooker = EmbeddingsCooker(
    #     w2v_model_path='./w2v.wv',
    #     lexicon_path='../lexicon/ANEW2017/ANEW2017All.txt'
    # )
    # cooker = EmbeddingsCooker(
    #     w2v_model_path='./w2v.wv',
    #     lexicon_path='../lexicon/NRC-VAD-Lexicon-Aug2018Release/NRC-VAD-Lexicon.txt'
    # )
    # cooker.load_lexicon()
    # cooker.refine_embeddings()
    pass
