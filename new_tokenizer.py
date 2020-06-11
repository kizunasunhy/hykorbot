import gluonnlp as nlp
from gluonnlp.data import SentencepieceTokenizer
import json

class MyTokenizer():

    def __init__(self, vocab_file_path):
        self.tokenizer = SentencepieceTokenizer(vocab_file_path)
        self.unknown_token = self.tokenizer.tokens.index("<unk>")
        self.pad_token_id = self.tokenizer.tokens.index("<pad>")
        self.max_len = 1024
        self.max_len_single_sentence = 1024
        self.unexpected_sep_token = ['<pad>', '<unk>']
        self.vocab_b_obj = vocab_b_obj = nlp.vocab.BERTVocab.from_sentencepiece(vocab_file_path,
                                                         mask_token=None,
                                                         sep_token=None,
                                                         cls_token=None,
                                                         unknown_token='<unk>',
                                                         padding_token='<pad>',
                                                         bos_token='<s>',
                                                         eos_token='</s>')
        self.size = len(self.tokenizer.tokens)
    
    def get_vocab_size(self):
        return self.size
    
    def tokenize(self, text):
        if text in self.unexpected_sep_token:
            return [text]
        return self.tokenizer(text)
    
    def convert_tokens_to_ids(self, tokens):
        ids = []
        if isinstance(tokens, str):
            if tokens in self.vocab_b_obj.token_to_idx:
                return self.vocab_b_obj.token_to_idx[tokens]
            else:
                return self.unknown_token
        for token in tokens:
            if token in self.vocab_b_obj.token_to_idx:
                ids.append(self.vocab_b_obj.token_to_idx[token])
            else:
                ids.append(self.unknown_token)
        return ids
    
    def convert_ids_to_tokens(self, ids):
        sentence = ''
        for id_ in ids:
            sentence += self.vocab_b_obj.idx_to_token[id_]
        sentence = sentence.replace('▁', ' ')
        return sentence.strip()
    
    def add_special_tokens(self, new_tokens):
        cnt = 0
        for token_list in new_tokens:
            tokens = new_tokens[token_list]
            if isinstance(tokens, str):
                if tokens not in self.vocab_b_obj.token_to_idx:
                    last_num = len(self.vocab_b_obj.token_to_idx.keys())
                    self.vocab_b_obj.token_to_idx[tokens] = last_num
                    self.vocab_b_obj.idx_to_token.append(tokens)

                    print(tokens, ' token is added in vocab! token_id:', self.vocab_b_obj.token_to_idx[tokens])
                    cnt += 1
                else:
                    print(tokens, ' token is already in vocab, token_id:', self.vocab_b_obj.token_to_idx[tokens])
            else:
                for token in tokens:
                    if token not in self.vocab_b_obj.token_to_idx:
                        last_num = len(self.vocab_b_obj.token_to_idx.keys())
                        self.vocab_b_obj.token_to_idx[token] = last_num
                        self.vocab_b_obj.idx_to_token.append(token)

                        print(token, ' token is added in vocab! token_id:', \
                              self.vocab_b_obj.token_to_idx[token])
                        cnt += 1
                    else:
                        print(token, ' token is already in vocab, token_id:', \
                              self.vocab_b_obj.token_to_idx[token])
        return cnt
