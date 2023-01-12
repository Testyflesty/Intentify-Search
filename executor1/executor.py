import multiprocessing
import os
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from jina import DocumentArray, Executor, requests
from text_dataset import TextDataset
from model import Model

from transformers import RobertaModel, RobertaTokenizer

class MyExecutor(Executor):
    def __init__(
        self,
        metas: Optional[Dict] = None,
        requests: Optional[Dict] = None,
        runtime_args: Optional[Dict] = None,
        workspace: Optional[str] = None,
        dynamic_batching: Optional[Dict] = None,
        **kwargs
    ):
        super().__init__(
            metas, requests, runtime_args, workspace, dynamic_batching, **kwargs
        )
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")
        self.model = RobertaModel.from_pretrained("microsoft/graphcodebert-base")    
        self.model = Model(self.model)
        checkpoint_prefix = "checkpoint-best-mrr/model.bin"
        output_dir = os.path.join("saved_models/python", "{}".format(checkpoint_prefix))
        self.model.load_state_dict(torch.load(output_dir), strict=False)

    @requests
    def foo(self, docs: DocumentArray, **kwargs):
        docs[0].text = "hello, world!"
        docs[1].text = "goodbye, world!"

    @requests(on="/crunch-numbers")
    def bar(self, docs: DocumentArray, **kwargs):
        for doc in docs:
            doc.tensor = torch.tensor(np.random.random([10, 2]))

    def predict_test(self):
        pool = multiprocessing.Pool(8)
        code_dataset = TextDataset(self.tokenizer, None, "test.jsonl", pool)
        code_sampler = SequentialSampler(code_dataset)
        code_dataloader = DataLoader(code_dataset, sampler=code_sampler, batch_size=64,num_workers=4)    
        code_vecs = []

        for batch in code_dataloader:
            code_inputs = batch[0]
            attn_mask = batch[1]
            position_idx = batch[2]
            print(code_inputs)
            print(attn_mask)
            print(position_idx)
            with torch.no_grad():
                code_vec= self.model(code_inputs=code_inputs, attn_mask=attn_mask,position_idx=position_idx)
                code_vecs.append(code_vec.cpu().numpy())  
        code_vecs=np.concatenate(code_vecs,0)
        return code_vecs