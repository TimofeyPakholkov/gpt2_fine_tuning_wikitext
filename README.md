Technical task:

You have a pre-trained GPT2 model from HF https://huggingface.co/openai-community/gpt2. You need to perform
fine-tuning on WikiText dataset from HF https://huggingface.co/datasets/wikitext (at least 2 epochs), measure quality

with BLEU metric, collect and build charts describing dynamics of training. Fine-tuning must be performed with Low-
Rank Adapters (LoRA) method https://arxiv.org/abs/2106.09685. The LoRA is a Parameter-Efficient-Fine-Tuning

(PEFT) approach that freezes pre-trained model and applies additional trainable parameters (weights) that are factorized

with small decomposition rank R. Thus, the number of LoRA trainable parameters is much smaller than the full fine-
tuning, so the fine-tuning requires much smaller amount of time.

///////////////////////////

How to run:

1) Create separate folder and extract files in there
2) Open terminal in created folder
3) Install all required libraries: 'pip install -r requirements.txt'
4) Run either 'python3 train_full_ft.py' (full fine-tunning implementation) or 'python3 train_lora.py' (LoRA-based implementation)

Results and comparison of two approaches:

LoRA allows to set considerably bigger batch size(8(LoRA) to 2(full FT)) on 8 GB GPU memory available.

![mem](https://github.com/TimofeyPakholkov/gpt2_fine_tuning_wikitext/assets/63054134/b1301331-22e7-4f52-aa8d-7fc620abfea9)

To runtime comparing equal batch sizes were set.

![time](https://github.com/TimofeyPakholkov/gpt2_fine_tuning_wikitext/assets/63054134/22003b4b-002d-48a9-a757-628ef21565d8)

Loss dynamic charts for both approaches:

![loss](https://github.com/TimofeyPakholkov/gpt2_fine_tuning_wikitext/assets/63054134/cad047c6-a79b-4782-b9e3-39366fd44769)
