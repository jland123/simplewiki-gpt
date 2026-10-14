# simplewiki-gpt

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jland123/simplewiki-gpt/blob/main/simplewiki_gpt.ipynb)
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/jland123/simplewiki-gpt/blob/main/simplewiki_gpt.ipynb)

A small GPT-style language model built from scratch in PyTorch and trained on Simple English Wikipedia.
Every layer is written by hand in a few lines of math, and the notebook explains each step in plain language.

## What is this?

A **language model** is a program that reads some text and guesses what comes next.
Make it guess, add its guess to the text, and ask again, and it writes.

This notebook builds one, trains it, and lets you give it the start of a sentence to finish.
It uses the same design as GPT (the models behind ChatGPT), but it is much smaller: about 38 million **parameters**, which are the adjustable numbers the model learns during training.

The notebook doesn't use PyTorch's ready-made layers (`nn.Linear`, `nn.LayerNorm`, `nn.MultiheadAttention`, ...). Each one is written out by hand, so you can see exactly what it does.

## Run it

You need a **GPU** (a graphics card). It does this kind of math far faster than a normal processor. Kaggle and Colab both lend you one for free.

**Kaggle (easiest):**
1. Click **Open in Kaggle** above.
2. In the right-hand panel, go to *Session options* → *Accelerator* and pick a GPU.
3. Click *Add Input*, search for **plain text wikipedia simpleenglish** (by *ffatty*), and add it.
4. Click *Run All*.

**Colab:** click **Open in Colab**, choose *Runtime* → *Change runtime type* → *T4 GPU*, then *Run all*.
The data cell downloads the dataset with `kagglehub`, which may ask you to sign in to Kaggle.

**Your own computer** (an Nvidia GPU is strongly recommended):

```bash
pip install -r requirements.txt
```

```bash
jupyter notebook simplewiki_gpt.ipynb
```

**Quick test run:** in the first code cell, set `MAX_LINES = 20_000` and `EPOCHS = 1`.

## What the notebook does

1. **Settings.** Every number you might want to change is in one cell.
2. **Load the text.** It reads 250,000 paragraphs of Simple English Wikipedia.
3. **Turn the text into numbers.** It splits the text into *tokens* (common chunks such as words or pieces of words) using GPT-2's tokenizer. Then it keeps only tokens that appear at least 50 times.
4. **Build the model.** Embeddings, then 4 transformer blocks, then an output layer, all written by hand.
5. **Train.** It shows the model examples, measures how wrong its guesses are, and nudges the parameters to be a little less wrong, over and over. The weights are saved to `weights.pt` after every epoch (one full pass through the training data).
6. **Check how well it learned.** It compares how wrong the model is on training text and on held-back text it never trained on.
7. **Let it write.** You give it the start of a sentence and it continues.

## The model at a glance

| Part | Default setting |
|---|---|
| Vocabulary | tokens seen at least 50 times in the data (≈24,500) |
| Numbers per token (`DIMS`) | 512 |
| Transformer blocks (`N_LAYERS`) | 4 |
| Attention heads per block (`N_HEADS`) | 4 |
| Longest text it reads at once | 170 tokens |
| Parameters | 37,828,558 |

Each transformer block does two steps, and each step adds its result onto what was already there:
- **Attention:** every token gathers information from the tokens before it, never from the ones after.
- **Feed-forward:** a small two-layer network that works on each token by itself (512 → 2048 → 512 numbers).

## Results so far

The **loss** is a score for how wrong the model's guesses are. Lower is better.
Guessing completely at random would score about 10.1 with this vocabulary.

| Run | Training loss | Validation loss |
|---|---|---|
| 2 blocks, 128 numbers per token | 4.14 | 4.35 |
| 4 blocks, 512 numbers per token (default) | 4.93 (average over the 1st epoch) | not measured yet |

"Validation" means text the model never trained on.
The full-size run stopped early in its second epoch because the GPU ran out of memory.
This cleaned-up version uses less GPU memory: it keeps the dataset off the GPU and stops layers from holding onto old results.
It hasn't been re-run at full size yet.

## Credits

- **Dataset:** [Plain text Wikipedia (SimpleEnglish)](https://www.kaggle.com/datasets/ffatty/plain-text-wikipedia-simpleenglish) by ffatty on Kaggle. The text comes from [Simple English Wikipedia](https://simple.wikipedia.org).
- **Tokenizer:** GPT-2's, through OpenAI's [`tiktoken`](https://github.com/openai/tiktoken) library.
