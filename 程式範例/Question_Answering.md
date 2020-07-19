#
```
# -*- coding: utf-8 -*-
"""Question_Answering.ipynb

# Let's Start Question Answering

使用pytorch-transformers 1.2.0版
"""

!pip install https://github.com/huggingface/pytorch-transformers/archive/1.2.0.zip

!rm -rf bert-chinese-qa*
!wget -q --no-check-certificate -r 'https://drive.google.com/uc?export=download&id=1GQtGFd-1AvZHZuYckhA3xqvvpDk-x5DW' -O bert-chinese-qa.zip
!unzip bert-chinese-qa.zip -d bert-chinese-qa

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)


def to_list(tensor):
    return tensor.detach().cpu().tolist()
 

def _get_best_indexes(logits, n_best_size=1):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes
 

def evaluate(dataset, model, tokenizer):
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=1)

    # Eval!
    all_results = []
    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      }
            example_indices = batch[3]
            outputs = model(**inputs)
            start_logits = to_list(outputs[0][0])
            end_logits   = to_list(outputs[1][0])
            start_indexes = _get_best_indexes(start_logits)
            end_indexes = _get_best_indexes(end_logits)
    return (start_indexes, end_indexes)

import collections

from torch.utils.data import TensorDataset




def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def convert_examples_to_features(tokenizer, question_text, doc_tokens, max_seq_length=384,
                                 doc_stride=128, max_query_length=64,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=0, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    """Loads a data file into a list of `InputBatch`s."""
    query_tokens = tokenizer.tokenize(question_text)
    if len(query_tokens) > max_query_length:
      query_tokens = query_tokens[0:max_query_length]
    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of the up to our max length with a stride of `doc_stride`.
    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset
        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(all_doc_tokens):
            break
        start_offset += min(length, doc_stride)

    for (doc_span_index, doc_span) in enumerate(doc_spans):
        tokens = []
        token_to_orig_map = {}
        token_is_max_context = {}
        segment_ids = []

        # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
        # Original TF implem also keep the classification token (set to 0) (not sure why...)
        p_mask = []

        # CLS token at the beginning
        if not cls_token_at_end:
            tokens.append(cls_token)
            segment_ids.append(cls_token_segment_id)
            p_mask.append(0)
            cls_index = 0

        # Query
        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(sequence_a_segment_id)
            p_mask.append(1)

        # SEP token
        tokens.append(sep_token)
        segment_ids.append(sequence_a_segment_id)
        p_mask.append(1)

        # Paragraph
        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

            is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                   split_token_index)
            token_is_max_context[len(tokens)] = is_max_context
            tokens.append(all_doc_tokens[split_token_index])
            segment_ids.append(sequence_b_segment_id)
            p_mask.append(0)
        paragraph_len = doc_span.length

        # SEP token
        tokens.append(sep_token)
        segment_ids.append(sequence_b_segment_id)
        p_mask.append(1)

        # CLS token at the end
        if cls_token_at_end:
            tokens.append(cls_token)
            segment_ids.append(cls_token_segment_id)
            p_mask.append(0)
            cls_index = len(tokens) - 1  # Index of classification token

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(pad_token)
            input_mask.append(0 if mask_padding_with_zero else 1)
            segment_ids.append(pad_token_segment_id)
            p_mask.append(1)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
    input_ids = torch.tensor([input_ids], dtype=torch.long)
    input_mask = torch.tensor([input_mask], dtype=torch.long)
    segment_ids = torch.tensor([segment_ids], dtype=torch.long)
    cls_index = torch.tensor([cls_index], dtype=torch.long)
    p_mask = torch.tensor([p_mask], dtype=torch.float)
    example_index = torch.arange(input_ids.size(0), dtype=torch.long)
    data = TensorDataset(input_ids, input_mask, segment_ids,
                            example_index, cls_index, p_mask)


    print("*** Example ***")
    # print("doc_span_index: %s" % (doc_span_index))
    print("tokens: %s" % " ".join(tokens))
    # print("token_to_orig_map: %s" % " ".join([
    #                 "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
    # print("token_is_max_context: %s" % " ".join([
    #                 "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
    #             ]))
    # print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    # print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    # print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

    return data, tokens

# 因為要我們今天要跑的是中文QA 所以只有Bert可以用
import torch
from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForQuestionAnswering, BertTokenizer)


device = torch.device("cuda")

checkpoint = 'bert-chinese-qa'
config_class, model_class, tokenizer_class = BertConfig, BertForQuestionAnswering, BertTokenizer
model = model_class.from_pretrained(checkpoint).to(device)
tokenizer = tokenizer_class.from_pretrained('bert-base-chinese', do_lower_case=True)

context = '國立臺灣大學，簡稱臺大、NTU，是臺灣第一所現代綜合大學，為臺灣學生人數最多的高等教育機構。其始於1928年日治時代中期創校的「臺北帝國大學」[注 1]，1945年中華民國接收臺灣後經改制與兩次易名始用現名。現設有11個學院、3個專業學院，下分54個學系、109個研究所；另設有30餘個各學術領域之國家級或校級研究中心，以及進修推廣部、臺大醫院等附屬機構，是全臺唯一學生人數超過三萬的高等教育學校[11][14]。目前亦為高教深耕計畫中參與全球鏈結全校型計畫的4所學校之一[15][16]。2020年QS世界大學排名位居第69名。此外，臺大擁有臺北市境內的3大校區、以及多處散布於全臺的分支校區與校地，總面積約3萬4千公頃，佔臺灣土地總面積的百分之一。'
question = '國立台灣大學簡稱為什麼'
data, tokens = convert_examples_to_features(tokenizer=tokenizer, question_text=question, doc_tokens=context)
start, end = evaluate(data, model, tokenizer)
"".join(tokens[start[0]: end[0]+1])

question = '台大面積有多大？'
data, tokens = convert_examples_to_features(tokenizer=tokenizer, question_text=question, doc_tokens=context)
start, end = evaluate(data, model, tokenizer)
"".join(tokens[start[0]: end[0]+1])
```
