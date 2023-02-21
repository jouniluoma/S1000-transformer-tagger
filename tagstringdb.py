import os
import sys
import re

import numpy as np
import spacy

from collections import deque
import functools
from multiprocessing import Pool
from multiprocessing import cpu_count


from common_hf import encode
from common_hf import argument_parser
from common_hf import process_sentences, get_predictions

from transformers import Trainer, TrainingArguments
from transformers import RobertaTokenizer, RobertaForTokenClassification, RobertaConfig
import scipy
import torch

from stringdb import stream_documents, StringDoc
import datetime


class NERDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# Alnum sequences preserved as single tokens, rest are
# single-character tokens.
TOKENIZATION_RE = re.compile(r'([^\W_]+|.)')


def sentence_split(text):
    if sentence_split.nlp is None:
        # Cache spacy model                                                     
        nlp = spacy.load('en_core_sci_sm', disable=['tagger','parser','ner','lemmatizer','textcat','pos'])
        nlp.add_pipe('sentencizer')
        sentence_split.nlp = nlp
    sentence_texts = []
    for para in text.split('\t'):
        sentence_texts.extend([s.text for s in sentence_split.nlp(para).sents])
    return sentence_texts
sentence_split.nlp = None

def tokenize(text):
    return [t for t in TOKENIZATION_RE.split(text) if t and not t.isspace()]


def split_and_tokenize(text):
    sentences = sentence_split(text)
    return [tokenize(s) for s in sentences]


def dummy_labels(tokenized_sentences):
    sentence_labels = []
    for tokens in tokenized_sentences:
        sentence_labels.append(['O'] * len(tokens))
    return sentence_labels


def get_word_labels(orig_words, token_lengths, tokens, predictions):
    """Map wordpiece token labels to word labels."""
    toks = deque([val for sublist in tokens for val in sublist])
    pred = deque([val for sublist in predictions for val in sublist])
    lengths = deque(token_lengths)
    word_labels = []
    for sent_words in orig_words:
        sent_labels = []
        for word in sent_words:
            sent_labels.append(pred.popleft())
            for i in range(int(lengths.popleft())-1):
                pred.popleft()
        word_labels.append(sent_labels)
    return word_labels


def iob2_span_ends(curr_type, tag):
    if curr_type is None:
        return False
    elif tag == 'I-{}'.format(curr_type):
        return False
    elif tag == 'O' or tag[0] == 'B':
        return True
    else:
        assert curr_type != tag[2:], 'internal error'
        return True    # non-IOB2 or tag sequence error


def iob2_span_starts(curr_type, tag):
    if tag == 'O':
        return False
    elif tag[0] == 'B':
        return True
    elif curr_type is None:
        return True    # non-IOB2 or tag sequence error
    else:
        assert tag == 'I-{}'.format(curr_type), 'internal error'
        return False


def tags_to_spans(text, tokens, tags):
    spans = []
    offset, curr_type, start = 0, None, None
    assert len(tokens) == len(tags)
    for token, tag in zip(tokens, tags):
        if iob2_span_ends(curr_type, tag):
            spans.append((start, offset, curr_type, text[start:offset]))
            curr_type, start = None, None
        while offset < len(text) and text[offset].isspace():
            offset += 1
        if text[offset:offset+len(token)] != token:
            raise ValueError('text mismatch')
        if iob2_span_starts(curr_type, tag):
            curr_type, start = tag[2:], offset
        offset += len(token)
    if curr_type is not None:
        spans.append((start, offset, curr_type, text[start:offset]))
    return spans

def write_sentences(outfile, sentences, labels):
    for sentence, tagseq in zip(sentences,labels):
        for word, tag in zip(sentence, tagseq):
            outfile.write('{}\t{}\n'.format(word, tag))
        outfile.write('\n')

def writespans(outfile, doc_id, spans):
    for s in spans:
        # TODO: If input data contains '\t' characters inside tagged spans, it will break the output tsv format.
        outfile.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(doc_id,1,1,s[0],s[1]-1,s[3],s[2],'SERIAL'))

def create_samples(tokenizer, seq_len, document):
    words = split_and_tokenize(document.text)
    labels = dummy_labels(words)
    data = process_sentences(words, labels, tokenizer, seq_len) #One doc at time --> documentwise
    tids, sids = encode(data.combined_tokens, tokenizer, seq_len)
    return StringDoc(document.doc_id, document.other_ids,document.authors, document.forum, document.year, document.text, tids, sids, data)    

def main(argv):

    argparser = argument_parser()
    args = argparser.parse_args(argv[1:])

    infn = args.input_data

    outfn = './output/{}-spans.tsv'.format(args.output_tsv)
    out_tsv = './output/{}-sentences.tsv'.format(args.output_tsv)


    config = RobertaConfig.from_pretrained(args.ner_model_dir)
    c_dict = config.to_dict()

    tokenizer = RobertaTokenizer.from_pretrained(args.ner_model_dir,config=config,do_lower_case = False)
    model = RobertaForTokenClassification.from_pretrained(args.ner_model_dir,config=config)


    training_args = TrainingArguments(
        output_dir="output/",          # output directory
        per_device_train_batch_size=32,  # batch size per device during training
        per_device_eval_batch_size=args.batch_size,   # batch size for evaluation
        evaluation_strategy='no',
        logging_strategy='no',
        save_strategy='no'
        )

    trainer=Trainer(model=model, args=training_args)
    trainer.model=model.cuda()

    seq_len = 512 
    inv_tag_map = c_dict["id2label"]


    print("Input file: ", infn )
    print("Preprocessing and inference starts: ", datetime.datetime.now())


    os.makedirs(os.path.dirname(outfn), exist_ok=True)
    with open(outfn, 'w+') as out_spans, open(out_tsv,'w+') as out_sentences:
        
        print("CPU count ", cpu_count())
        partial_create_documents = functools.partial(create_samples,tokenizer,seq_len)
        with Pool(64) as p:
            input_docs = p.map(partial_create_documents,stream_documents(infn))
            print("input docs len", len(input_docs), flush=True)
                
            input_sentences=[]
            data_list = []
            input_sentence_counts = []
            num_input_sentences = 0
            tok_start = 0

            for count,document in enumerate(input_docs):

                num_input_sentences+=len(document.tids)
                input_sentence_counts.append(len(document.tids))
                data_list.append(document.data)
                input_sentences.append((document.doc_id, num_input_sentences, document.text))  #Sentences per doc for writing spans

                if num_input_sentences > args.sentences_on_batch:
                    print("num input sentences ", num_input_sentences)
                    print("Tok start ",tok_start)
                    print("count ", count)
                    toks = np.array([sample for samples in input_docs[tok_start:count+1] for sample in samples.tids])
                    seqs = np.array([sample for samples in input_docs[tok_start:count+1] for sample in samples.sids])
                    print("toks shape ", toks.shape)
                    print("seqs shape ", seqs.shape)
                    tok_start = count+1
                    print("Inference starts: ", datetime.datetime.now(),flush=True)
                    print(num_input_sentences, datetime.datetime.now(),flush=True)

                    predict_encodings = {"input_ids" : torch.from_numpy(toks),
                    "token_type_ids" : torch.from_numpy(np.zeros(toks.shape, dtype=int)),
                    "attention_mask" : torch.from_numpy(seqs)}


   
                    predict_dataset = NERDataset(predict_encodings, np.zeros(toks.shape,dtype=int))


                    logits = trainer.predict(predict_dataset).predictions
                    probs = scipy.special.softmax(logits, axis=-1)
                    preds = np.argmax(probs, axis=-1)

                    start = 0
                    print("Postprocess starts: ", datetime.datetime.now(),flush=True)
                    for data, indices in zip(data_list, input_sentences):
                        pred = preds[start:indices[1]]
                        pred_cmv, _ = get_predictions(pred, data.tokens, data.sentence_numbers)
                        
                        token_labels=[]
                        for pp in pred_cmv:
                            token_labels.append([inv_tag_map[t] for t in pp])

                        start=indices[1]
                        word_labels = get_word_labels(
                            data.words, data.lengths, data.tokens, token_labels)

                        # Flatten and map to typed spans with offsets
                        out_sentences.write(f"--DOCSTART--\tO\t{indices[0]}\n\n")
                        write_sentences(out_sentences, data.words, word_labels)

                        word_sequence = [w for s in data.words for w in s]
                        tag_sequence = [t for s in word_labels for t in s]
                        spans = tags_to_spans(indices[2], word_sequence, tag_sequence)                        
                        writespans(out_spans, indices[0], spans)

                    input_sentences=[]
                    data_list =[]
                    num_input_sentences=0
                    toks = np.array([], dtype=np.int64).reshape(0,seq_len)
                    seqs = np.array([], dtype=np.int64).reshape(0,seq_len)
                    out_spans.flush()
                    print("preprocess starts: ", datetime.datetime.now(),flush=True)

            if input_sentences:
                print("num input sentences ", num_input_sentences)
                print("Tok start ",tok_start)
                print("count ", count) 
                toks = np.array([sample for samples in input_docs[tok_start:] for sample in samples.tids])
                seqs = np.array([sample for samples in input_docs[tok_start:] for sample in samples.sids])
                print("Inference starts: ", datetime.datetime.now(),flush=True)
                print(num_input_sentences, datetime.datetime.now(),flush=True)
                print("toks shape ", toks.shape)
                print("seqs shape ", seqs.shape)
                predict_encodings = {"input_ids" : torch.from_numpy(toks),
                "token_type_ids" : torch.from_numpy(np.zeros(toks.shape, dtype=int)),
                "attention_mask" : torch.from_numpy(seqs)}


   
                predict_dataset = NERDataset(predict_encodings, np.zeros(toks.shape,dtype=int))

                logits = trainer.predict(predict_dataset).predictions

                probs = scipy.special.softmax(logits, axis=-1)
                preds = np.argmax(probs, axis=-1)

                start = 0
                for data, indices in zip(data_list, input_sentences):
                    pred = preds[start:indices[1]]
                    pred_cmv, _ = get_predictions(pred, data.tokens, data.sentence_numbers)
                    
                    token_labels=[]
                    for pp in pred_cmv:
                        token_labels.append([inv_tag_map[t] for t in pp])
                    start=indices[1]
                    word_labels = get_word_labels(
                        data.words, data.lengths, data.tokens, token_labels)
                    with open(out_tsv,'a+') as outputfile:
                        outputfile.write(f"--DOCSTART--\tO\t{indices[0]}\n\n")
                        write_sentences(outputfile, data.words, word_labels)

                    # Flatten and map to typed spans with offsets
                    word_sequence = [w for s in data.words for w in s]
                    tag_sequence = [t for s in word_labels for t in s]
                    spans = tags_to_spans(indices[2], word_sequence, tag_sequence)
                    writespans(of, indices[0], spans)
    print("inference ends: ", datetime.datetime.now(),flush=True)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
