# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
import random
from argparse import ArgumentParser
from pprint import pformat
import warnings

import torch
import torch.nn.functional as F

from transformers import GPT2Config, GPT2DoubleHeadsModel
from train import SPECIAL_TOKENS, build_input_from_segments
from new_tokenizer import MyTokenizer


vocab_file_path = 'model/kogpt2_news_wiki_ko_cased_818bfa919d.spiece'
tokenizer = MyTokenizer(vocab_file_path)
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<s>', 'eos_token': '</s>', 'pad_token': '<pad>',
                         'additional_special_tokens': ['<sent1>', '<sent2>']}

def top_filtering(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def sample_sequence(history, tokenizer, model, args, current_output=[]):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

    for i in range(args.max_length):
        instance = build_input_from_segments([tokenizer.convert_tokens_to_ids(history)], current_output, tokenizer, with_eos=False)
        input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)

        logits = model(input_ids, token_type_ids=token_type_ids)
        if isinstance(logits, tuple):  # for gpt2 and maybe others
            logits = logits[0]
        logits = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)

        if prev.item() in special_tokens_ids:
            current_output.append(prev.item())
            break
        current_output.append(prev.item())

    return current_output


def add_special_tokens_(model, tokenizer):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    print('tokenizer size:', tokenizer.size)
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
    

def run():
    parser = ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=20, help="Maximum length of the output utterances")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(args))
    
    if args.model_checkpoint == "":
        raise ValueError("Requiring a finetuned model_checkpoint")
	
    if args.seed != 0:
    	random.seed(args.seed)
    	torch.random.manual_seed(args.seed)
    	torch.cuda.manual_seed(args.seed)

    logger.info("Get pretrained model and tokenizer")
    config = GPT2Config(vocab_size=50003)
    model = GPT2DoubleHeadsModel(config)
    if args.model_checkpoint:
        print("\nLoad model from", args.model_checkpoint)
        model.load_state_dict(torch.load(args.model_checkpoint), strict=False)

    model.to(args.device)
    add_special_tokens_(model, tokenizer)

    history = ''
    print('\nPlease input a sentece to chat with the chatbot!')
    while True:
        raw_text = input(">>> ")
        while not raw_text:
            print('Prompt should not be empty!')
            raw_text = input(">>> ")
        history = tokenizer.tokenize(raw_text)
        with torch.no_grad():
            out_ids = sample_sequence(history, tokenizer, model, args)
        print(tokenizer.convert_ids_to_tokens(out_ids))

if __name__ == "__main__":
    run()
