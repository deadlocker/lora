# GPT-2 Fine Tune using LORA

A lightweight PyTorch implementation of Low-Rank Adaptation (LoRA) for fine-tuning GPT-2 on custom text datasets

Dataset Soure: https://theanarchistlibrary.org/library/sophia-nachalo-and-yarostan-vochek-letters-of-insurgents

# Installation and Setup
1. pip install -r requirements.txt 

# Train model
1. python train.py

# Outputs
Model weights saved to /outputs

# Generated Sample Eval data
1. python eval.py

Using device: cuda

Loading model..

Loading weights from ./outputs/gpt2_lora_weights.pt...
Prompt: My dearest Yaro, I am writing to tell you about the unexpected events that have occurred recently.

Generated text: My dearest Yaro, I am writing to tell you about the unexpected events that have occurred recently. 
During my visit we visited a large school building in downtown Paris which is under siege by armed students who are 
attempting an attack on it from behind its walls and barricades. This time they were able only because of our presence. 
Our arrival was accompanied with signs saying: “The police’s headquarters here!” We then saw two men wearing black 
uniforms carrying guns; one carried a pistol while the other held his rifle pointed at us. The gun had been confiscated
during their first encounter. It did not belong either inside or outside this classroom but within each of them. 
They also told me there was no way they could possibly be connected