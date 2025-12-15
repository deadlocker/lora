#!/usr/bin/env python3

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, logging
from models.lora import inject_lora

logging.set_verbosity_error()

if __name__ == '__main__':
    RANK = 32
    ALPHA = 128

    DROPOUT = 0.1
    TARGET = ['c_attn', 'c_proj']
    WEIGHTS_PATH = './outputs/gpt2_lora_weights.pt'

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f'\nUsing device: {device}')

    print(f'\nLoading model..')
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    inject_lora(model, RANK, ALPHA, TARGET, DROPOUT)

    print(f'\nLoading weights from {WEIGHTS_PATH}...')
    try:
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device), strict=False)
    except FileNotFoundError:
        print(f'\nWeights file {WEIGHTS_PATH} not found.')
        exit()

    model.eval()
    model.to(device)

    prompt = 'My dearest Yaro, I am writing to tell you about the unexpected events that have occurred recently.'
    print(f'Prompt: {prompt}')

    tokenizer.pad_token = tokenizer.eos_token
    input_ids = tokenizer.encode(
        prompt,
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=150,
            num_return_sequences=1,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f'\nGenerated text: {generated_text}')