#!/usr/bin/env python3

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, logging
from models.lora import inject_lora
from tqdm import tqdm

logging.set_verbosity_error()

if __name__ == '__main__':
    RANK = 32
    ALPHA = 128

    DROPOUT = 0.1
    LEARNING_RATE = 5e-5
    BATCH_SIZE = 16
    EPOCHS = 5
    SEQ_LENGTH = 512
    GRAD_CLIP = 1.0
    TARGET = ['c_attn', 'c_proj']
    DATA_FILE = 'datasets/yaro_letter.txt'

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'\nUsing device: {device}')

    print('\nLoading GPT2 model and Tokenizer...')
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.model_max_length = int(1e9)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.requires_grad_(False)

    inject_lora(model, RANK, ALPHA, TARGET, DROPOUT)
    model.to(device)
    model.train()

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f'File not found: {DATA_FILE}')

    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        text_data = f.read()

    tokens = tokenizer(
        text_data,
        return_tensors='pt'
    )['input_ids'][0]

    chunks_size = len(tokens) // SEQ_LENGTH

    data = []
    for idx in range(chunks_size):
        start = idx * SEQ_LENGTH
        end = start + SEQ_LENGTH
        chunk = tokens[start:end]
        data.append(chunk)

    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f'Total parameters: {total_params}')
    print(f'Total trainable parameters: {train_params}')
    if total_params > 0:
        print(f'Trainable %: {(train_params / total_params) * 100:.2f}%')

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE
    )
    scaler = torch.amp.GradScaler() if device.type == 'cuda' else None

    V_SIZE = model.config.vocab_size
    print('\nFinetuning GPT2 model Started...')

    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}', leave=True)

        for batch in pbar:
            INPUT = batch.to(device)
            optimizer.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(INPUT, labels=INPUT)
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(INPUT, labels=INPUT)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], GRAD_CLIP)
                optimizer.step()

            current_loss = loss.item()
            epoch_loss += current_loss
            pbar.set_postfix({'loss': f'{current_loss:.4f}'})

        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch {epoch}/{EPOCHS} | Avg Loss: {avg_loss:.4f}')

    print('Finetuning GPT2 model Completed...')

    save_path = './outputs/gpt2_lora_weights.pt'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    lora_state_dict = {k: v for k, v in model.state_dict().items() if 'lora' in k}
    torch.save(lora_state_dict, save_path)
    print(f"Model weights saved to {save_path}")