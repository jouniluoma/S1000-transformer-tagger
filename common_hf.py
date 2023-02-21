import sys
import numpy as np

from collections import deque, namedtuple
from argparse import ArgumentParser

#from config import DEFAULT_SEQ_LEN, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS

from transformers import RobertaTokenizer, RobertaTokenizerFast, BertTokenizer

DEFAULT_BATCH_SIZE=32

Sentences = namedtuple('Sentences', [
    'words', 'tokens', 'labels', 'lengths', 
    'combined_tokens', 'combined_labels','sentence_numbers', 'sentence_starts'
])

def argument_parser(mode='train'):
    argparser = ArgumentParser()

    argparser.add_argument(
        '--input_data', required=True,
        help='Training data'
    )
    argparser.add_argument(
        '--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
        help='Batch size for training'
    )
    argparser.add_argument(
        '--output_spans', default="pubmed-output/output.spans",
        help='File to write predicted spans to'
    )
    argparser.add_argument(
        '--output_tsv', default="pubmed-output/output.tsv",
        help='File to write predicted tsv to'
    )
    argparser.add_argument(
        '--ner_model_dir',
        help='Trained NER model directory'
    )
    argparser.add_argument(
        '--sentences_on_batch', type=int, default=2000,
        help = 'Write tagger output after this number of sentences'
    )

    return argparser

def encode(lines, tokenizer, max_len):
    tids = []
    sids = []
    for line in lines:
        tokens = [tokenizer.cls_token]+line
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        if len(token_ids) < max_len:
            pad_len = max_len - len(token_ids)
            token_ids += tokenizer.convert_tokens_to_ids([tokenizer.pad_token]) * pad_len
            segment_ids += [0] * pad_len
        tids.append(token_ids)
        sids.append(segment_ids)
    return np.array(tids), np.array(sids)

# def label_encode(labels, tag_dict, max_len):
#     encoded = []
#     sample_weights = []
#     for sentence in labels:
#         enc = [tag_dict[i] for i in sentence]
#         enc.insert(0, tag_dict['O'])
#         weight = [0 if i=='[SEP]' else 1 for i in sentence]  #TODO: Is there sep tokens in labels?
#         weight.insert(0,0)
#         if len(enc) < max_len:
#             weight.extend([0]*(max_len-len(enc)))
#             enc.extend([tag_dict['O']]*(max_len-len(enc)))
#         encoded.append(np.array(enc))
#         sample_weights.append(np.array(weight))
#     lab_enc = np.expand_dims(np.stack(encoded, axis=0), axis=-1)
#     weights = np.stack(sample_weights, axis=0)
#     return np.array(lab_enc), np.array(weights)


# def get_labels(label_sequences):
#     unique = set([t for s in label_sequences for t in s])
#     labels = sorted(list(unique), reverse=True)
#     return labels


# def tokenize_and_split(words, word_labels, tokenizer, max_length):
#     unk_token = tokenizer.unk_token
#     # Tokenize each word in sentence, propagate labels
#     tokens, labels, lengths = [], [], []
#     for word, label in zip(words, word_labels):
#         ttt = tokenizer(word, add_special_tokens=False,return_length=True)
#         tokenized = tokenizer.convert_ids_to_tokens(ttt.input_ids)
#         tokens.extend(tokenized)
#         lengths.append(len(tokenized))
#         for i, token in enumerate(tokenized):
#             if i == 0:
#                 labels.append(label)
#             else:
#                 if label.startswith('B'):
#                     labels.append('I'+label[1:])
#                 else:
#                     labels.append(label)

#     # Split into multiple sentences if too long
#     split_tokens, split_labels = [], []
#     start, end = 0, max_length
#     while end < len(tokens):
#         # Avoid splitting inside tokenized word
#         while end > start and tokens[end].startswith('##'):
#             end -= 1
#         if end == start:
#             end = start + max_length    # only continuations
#         split_tokens.append(tokens[start:end])
#         split_labels.append(labels[start:end])
#         start = end
#         end += max_length
#     split_tokens.append(tokens[start:])
#     split_labels.append(labels[start:])

#     return split_tokens, split_labels, lengths

def tokenize_and_split(words, word_labels, tokenizer, max_length):
    unk_token = tokenizer.unk_token
    # Tokenize each word in sentence, propagate labels
    #labels, lengths = [], [], []
    ttt = tokenizer(' '.join(words), add_special_tokens=False,return_length=True)
    tokens = tokenizer.convert_ids_to_tokens(ttt.input_ids)
    lengths = lengths_in_subwords(tokens, tokenizer)
    labels = []
    for length, label in zip(lengths, word_labels):
        if label.startswith('B-'):
            labels.append(label)
            if length > 1:
                ll = 'I-'+label[2:]
                labels.extend((length-1)*[ll])
        else:
            labels.extend(length*[label])


    # Split into multiple sentences if too long
    split_tokens, split_labels = [], []
    start, end = 0, max_length
    while end < len(tokens):
        # Avoid splitting inside tokenized word
        while end > start and (tokens[end].startswith('##') or not tokens[end].startswith('Ġ')):
            end -= 1
        if end == start:
            end = start + max_length    # only continuations
        split_tokens.append(tokens[start:end])
        split_labels.append(labels[start:end])
        start = end
        end += max_length
    split_tokens.append(tokens[start:])
    split_labels.append(labels[start:])

    return split_tokens, split_labels, lengths


def tokenize_and_split_sentences(orig_words, orig_labels, tokenizer, max_length):
    words, labels, lengths = [], [], []
    for w, l in zip(orig_words, orig_labels):
        split_w, split_l, lens = tokenize_and_split(w, l, tokenizer, max_length-2)
        words.extend(split_w)
        labels.extend(split_l)
        lengths.extend(lens)
    return words, labels, lengths


def read_conll(input_file, mode='train'):
    # words and labels are lists of lists, outer for sentences and
    # inner for the words/labels of each sentence.
    words, labels = [], []
    curr_words, curr_labels = [], []
    with open(input_file) as f:
        for line in f:
            line = line.strip()
            if line:
                fields = line.split('\t')
                if len(fields) > 1:
                    curr_words.append(fields[0])
                    if mode != 'test':
                        curr_labels.append(fields[1])
                    else:
                        curr_labels.append('O')
                else:
                    print('ignoring line: {}'.format(line), file=sys.stderr)
                    pass
            elif curr_words:
                words.append(curr_words)
                labels.append(curr_labels)
                curr_words, curr_labels = [], []
    if curr_words:
        words.append(curr_words)
        labels.append(curr_labels)
    return words, labels


def process_sentences(words, orig_labels, tokenizer, max_seq_len, seq_start=0):
    # Tokenize words, split sentences to max_seq_len, and keep length
    # of each source word in tokens
    tokens, labels, lengths = tokenize_and_split_sentences(
        words, orig_labels, tokenizer, max_seq_len)

    # Extend each sentence to include context sentences
    combined_tokens, combined_labels, sentence_numbers, sentence_starts = combine_sentences(
        tokens, labels, max_seq_len-1, tokenizer, seq_start)

    return Sentences(
        words, tokens, labels, lengths, combined_tokens, combined_labels, sentence_numbers, sentence_starts)


def write_result(fname, original, token_lengths, tokens, labels, predictions, tokenizer, mode='train'):
    lines=[]
    with open(fname,'w+') as f:
        toks = deque([val for sublist in tokens for val in sublist])
        labs = deque([val for sublist in labels for val in sublist])
        pred = deque([val for sublist in predictions for val in sublist])
        lengths = deque(token_lengths)
        sentences = []
        for sentence in original:
            sent = []
            for word in sentence:
                tok = toks.popleft()
                # TODO avoid hardcoded "[UNK]" string
                if not (word.startswith(tok) or word.startswith(tok[1:]) or tok == tokenizer.unk_token or word.lower().startswith(tok)):
                    print('tokenization mismatch: "{}" vs "{}"'.format(
                        word, tok), file=sys.stderr)
                label = labs.popleft()
                predicted = pred.popleft()
                sent.append(predicted)
                for i in range(int(lengths.popleft())-1):
                    toks.popleft()
                    labs.popleft()
                    pred.popleft()                           
                if mode != 'predict':
                    line = "{}\t{}\t{}\n".format(word, label, predicted)
                else:
                    # In predict mode, labels are just placeholder dummies
                    line = "{}\t{}\n".format(word, predicted)
                f.write(line)
                lines.append(line)
            f.write("\n")
            sentences.append(sent)
    f.close()
    return lines, sentences



def combine_sentences(lines, tags, max_seq, tokenizer, start=0):
    lines_in_sample = []
    linestarts_in_sample = []
    new_lines = []
    new_tags = []
    position = start

    for i, line in enumerate(lines):
        line_starts = []
        line_numbers = []
        if start + len(line) < max_seq:                    
            new_line = [0]*start
            new_tag = [0]*start
            new_line.extend(line)
            new_tag.extend(tags[i])
            line_starts.append(start)
            line_numbers.append(i)
        else:
            position = max_seq - len(line) -1 
            new_line = [0]*position
            new_tag = [0]*position
            new_line.extend(line)
            new_tag.extend(tags[i])
            line_starts.append(position)
            line_numbers.append(i)
        j = 1
        next_idx = (i+j)%len(lines)
        ready = False
        while not ready:
            if len(lines[next_idx]) + len(new_line) < max_seq - 1: 
                new_line.append(tokenizer.sep_token)
                new_tag.append('O')
                position = len(new_line)
                new_line.extend(lines[next_idx])
                new_tag.extend(tags[next_idx])
                line_starts.append(position)
                line_numbers.append(next_idx)
                j += 1
                next_idx = (i+j)%len(lines)
            else:
                new_line.append(tokenizer.sep_token)
                new_tag.append('O')
                position = len(new_line)
                new_line.extend(lines[next_idx][0:(max_seq-position)])
                new_tag.extend(tags[next_idx][0:(max_seq-position)])
                ready = True                


        j=1
        ready = False
        while not ready:
            counter = line_starts[0]
            prev_line = lines[i-j][:]
            prev_tags = tags[i-j][:]
            prev_line.append(tokenizer.sep_token)
            prev_tags.append('O')

            if len(prev_line)<= counter:
                new_line[(counter-len(prev_line)):counter]=prev_line
                new_tag[(counter-len(prev_line)):counter]=prev_tags
                line_starts.insert(0,counter-len(prev_line))
                line_numbers.insert(0,i-j)  #negative numbers are indices to end of lines array
                j+=1
            else:
                if counter > 2:
                    new_line[0:counter] = prev_line[-counter:]
                    new_tag[0:counter] = prev_tags[-counter:]
                    ready = True
                else:
                    new_line[0:counter] = [tokenizer.pad_token]*counter
                    new_tag[0:counter] = ['O']*counter
                    ready = True
        new_lines.append(new_line)
        new_tags.append(new_tag)
        lines_in_sample.append(line_numbers)
        linestarts_in_sample.append(line_starts)
    return new_lines, new_tags, lines_in_sample, linestarts_in_sample

def get_predictions(predicted, lines, line_numbers):
    first_pred = []
    final_pred = []
    predictions = [[] for _ in range(len(lines))]
    for i, sample in enumerate(predicted):
        idx = 1
        for j, line_number in enumerate(line_numbers[i]):
            predictions[line_number].append(sample[idx:idx+len(lines[line_number])])
            if j == 0:
                first_pred.append(sample[idx:idx+len(lines[line_number])])
            idx+=len(lines[line_number])+1
    for i, prediction in enumerate(predictions):
        pred = []
        arr = np.stack(prediction, axis=0)
        for j in arr.T:
            u,c = np.unique(j, return_counts=True)
            pred.append(u[np.argmax(c)])
        final_pred.append(pred)
    return final_pred, first_pred

# def get_predictions2(probs, lines, line_numbers):
#     first_pred = []
#     final_pred = []
#     predictions = []
#     p_first = []
#     for i, line in enumerate(lines):
#         predictions.append(np.zeros((len(line),probs.shape[-1])))  #create empty array for each line
    
#     for i, sample in enumerate(probs):
#         idx = 1
#         for j, line_number in enumerate(line_numbers[i]):
#             if j == 0:
#                 p_first.append(sample[idx:idx+len(lines[line_number]),:])
#             predictions[line_number] += sample[idx:idx+len(lines[line_number]),:]
#             idx+=len(lines[line_number])+1
    
#     for k, line in enumerate(predictions): 
#         final_pred.append(np.argmax(line, axis=-1))
#         first_pred.append(np.argmax(p_first[k],axis=-1))
            
#     return final_pred, first_pred

# def process_docs(docs, doc_tags, line_ids, tokenizer, seq_len):
    
#     f_words = []
#     f_tokens = []
#     f_labels = []
#     f_lengths = []
#     f_combined_tokens = []
#     f_combined_labels = []
#     f_sentence_numbers = []
#     f_sentence_starts = []
#     start_sentence_number = 0
#     for i, doc in enumerate(docs):
#             tokens, labels, lengths = tokenize_and_split_sentences(
#                 doc, doc_tags[i], tokenizer, seq_len)
#             combined_tokens, combined_labels, sentence_numbers = combine_sentences(tokens, 
#                                                                                    labels,
#                                                                                    lengths,
#                                                                                    seq_len)
#             for numbers in sentence_numbers:
#                 f_sentence_numbers.append([num+start_sentence_number for num in numbers])
#             start_sentence_number += len(tokens)
#             f_words.extend(doc)
#             f_tokens.extend(tokens)
#             f_labels.extend(labels)
#             f_lengths.extend(lengths)
#             f_combined_tokens.extend(combined_tokens)
#             f_combined_labels.extend(combined_labels)
#             f_sentence_starts.extend([0]*len(tokens))
            
#     return Sentences(f_words, f_tokens, f_labels, f_lengths, f_combined_tokens, f_combined_labels, f_sentence_numbers, f_sentence_starts)         
                
# def split_to_documents(sentences, tags):
#     documents = []
#     documents_tags = []
#     doc_idx = 0
#     document = []
#     d_tags = []
#     line_ids =[[]]
#     for i, sentence in enumerate(sentences):
#         if sentence[0].startswith("-DOCSTART-") and i!=0:
#             documents.append(document)
#             documents_tags.append(d_tags)
#             document = []
#             d_tags = []
#             line_ids.append([])
#             doc_idx+=1
#         document.append(sentence)
#         d_tags.append(tags[i])
#         line_ids[doc_idx].append(i)
#     if documents:
#             documents.append(document)
#             documents_tags.append(d_tags)
#     return documents, documents_tags, line_ids
            
# def process_no_context(train_words, train_tags, tokenizer, seq_len):
#     train_tokens, train_labels, train_lengths = tokenize_and_split_sentences(train_words, train_tags, tokenizer, seq_len)
#     sentence_numbers = []
#     sentence_starts = []
#     for i, line in enumerate(train_tokens):
#         #line.append('[SEP]')
#         #train_labels[i].append('[SEP]')
#         sentence_numbers.append([i])
#         sentence_starts.append([0])
#     return Sentences(train_words, train_tokens, train_labels, train_lengths, train_tokens, train_labels, sentence_numbers, sentence_starts)

# def shuffle_inputs(documents, labels, line_ids):
#     docs_s = []
#     tags_s = []
#     ids_s = []  
#     for doc,tags,ids in zip(documents,labels,line_ids):
#         idxarr = np.arange(len(ids))
#         np.random.shuffle(idxarr)
#         docs_s.append([doc[i] for i in idxarr])
#         tags_s.append([tags[i] for i in idxarr])
#         ids_s.append([ids[i] for i in idxarr])
#     return docs_s, tags_s, ids_s

# def flatten_shuffled(documents, labels):
#     words_out = [sentence for doc in documents for sentence in doc]
#     tags_out = [tags for label in labels for tags in label]
#     return words_out, tags_out

def lengths_in_subwords(input_sequence, tokenizer):
    lengths = []
    length = 0 #Starting token does not have Ġ in the beginning 
    if isinstance(tokenizer, (RobertaTokenizer, RobertaTokenizerFast)):
        for token in input_sequence:
            if token.startswith('Ġ'):
                lengths.append(length)
                length = 0
            length+=1
        if length>0:
            lengths.append(length)
        return lengths
    elif isinstance(tokenizer, BertTokenizer):
        for token in input_sequence:
            if not token.startswith('##') and length>0:
                lengths.append(length)
                length = 0
            length+=1
        if length>0:
            lengths.append(length)
        return lengths