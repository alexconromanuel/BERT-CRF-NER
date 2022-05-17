# Copyright (c) 2022, Institut Teknologi Del. All rights reserved.

import argparse
import os
import sys
import time

import ner

from tqdm import tqdm

import torch
from torch.utils import data

from transformers import AdamW
from transformers import AutoTokenizer
from transformers import BertModel

import warnings
warnings.filterwarnings("ignore")

def run(dataset):
    cuda_yes = torch.cuda.is_available()
    device = torch.device("cuda") if cuda_yes else torch.device("cpu")
    print('Device:', device)
    
    list_token = ner.read_txt_file(dataset)

    conllProcessor = ner.CoNLLDataProcessor(list_token)
    label_list = conllProcessor.get_labels()
    label_map = conllProcessor.get_label_map()
    train_examples = conllProcessor.get_train_examples()
    test_examples = conllProcessor.get_test_examples()

    tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")

    train_dataset = ner.NerDataset(train_examples, tokenizer, label_map, 613)
    test_dataset = ner.NerDataset(test_examples, tokenizer, label_map, 613)

    batch_size = 8
    train_dataloader = data.DataLoader(dataset=train_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=4,
                                    collate_fn=ner.NerDataset.pad)


    test_dataloader = data.DataLoader(dataset=test_dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=4,
                                    collate_fn=ner.NerDataset.pad)

    bert_model = BertModel.from_pretrained("indobenchmark/indobert-base-p1")

    start_label_id = conllProcessor.get_start_label_id()
    stop_label_id = conllProcessor.get_stop_label_id()

    model = ner.BERT_CRF_NER(bert_model, start_label_id, stop_label_id, 
                         len(label_list), 512 , batch_size, device)

    model.to(device)

    learning_rate0 = 5e-5

    optimizer = AdamW(model.parameters(), lr=learning_rate0, correct_bias=False)

    total_train_epochs = 1
    gradient_accumulation_steps = 1
    total_train_steps = int(len(train_examples) / batch_size / gradient_accumulation_steps * total_train_epochs)
    global_step_th = 0
    warmup_proportion = 0.1

    for epoch in range(total_train_epochs):
        tr_loss = 0
        train_start = time.time()
        model.train()
        optimizer.zero_grad()
        # for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        for step, batch in  tqdm(enumerate(train_dataloader)):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, predict_mask, label_ids = batch

            neg_log_likelihood = model.neg_log_likelihood(input_ids, segment_ids, input_mask, label_ids)

            if gradient_accumulation_steps > 1:
                neg_log_likelihood = neg_log_likelihood / gradient_accumulation_steps

            neg_log_likelihood.backward()

            tr_loss += neg_log_likelihood.item()

            if (step + 1) % gradient_accumulation_steps == 0:
                # modify learning rate with special warm up BERT uses
                lr_this_step = learning_rate0 * ner.warmup_linear(global_step_th/total_train_steps, warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step_th += 1

        print('--------------------------------------------------------------')
        print("Epoch:{} completed, Total training's Loss: {}, Spend: {}m".format(epoch, tr_loss, (time.time() - train_start)/60.0))
        print(ner.evaluate(model, test_dataloader, label_map))


_examples = '''examples:
  # Train and Evaluate IndoBERT-CRF
  python %(prog)s --dataset=~/data/singgalang.txt
'''

def main():
    parser = argparse.ArgumentParser(
        description='IndoBERT-CRF.',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--dataset', help='Dataset file', required=True)

    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print('Error: dataset does not exist.')
        sys.exit(1)
    
    run(**vars(args))

if __name__ == "__main__":
    main()