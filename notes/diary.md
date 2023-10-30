# Diary

## 2023-08-30
When we try default setup, a peculiar bug happens: The shuffle for val loader causes significant change in val accuracy! This obviously should not happen and it now tracked down to the speculation that val loader shuffle actually impact random generator state which in turn causes different shuffled data for train. The difference is too big: val accuracy with shuffle reaches ~1 within 3k steps if val shuffle is on while it doesn't do so otherwise well beyond 10k steps. Just random number generation should not be causing more than 3X speedup in generalization.

So, as next step, we just scan different random seeds and see if such 3X speedup is causes by some of the seeds. It turns out 18 out of 2073 seeds causes 3X speedup. That's about 0.86% of all seeds. More curiously, 93% of the seeds shows normal behaviour while other 7% causes noticeable speedup in generalization.

The next question was what is the role of random state of model vs order of data. To analyse that, data loader now has different seed. if we don't change model seed but only data loader seed, we find about 0.43% of seeds causing 3X speedup. This may be noisy estimate because we still see similar 94% of data loader seed showing regular behaviour while 6% causing noticeable speedup.

Next question is if data loader seed is good enough? It turns out it is not. For a given model seed, there is good data loader seed but then that same data loader seed might now show same speed up for other model seed. However, good data loader seed definitely improves chances! For data loader seed 8, we find that number of good model seeds goes up by 25%. That is, if you pick good data loader seed, the model seed has 25% more chance of cooperating!

All this requires re-thinking some of the hypothesis. Specifically, that model generalizes first then weight decay (wd) redistributes the weights to cause generalization. The difference between two events causes grokking phenomenon. If weight decay is increased then grokking happens faster.

Curiously, when wd=0, grokking still happens but takes much longer (~100k as opposed to ~12k steps). This perhaps indicaes that SGD has some self-regularization.

Can we find seed that causes val=1 as soon as 3k steps with wd=0? It turns out we can if we use magic data loader seed but those seeds are extremely rare, about 13X rarer.

Why should data loader and model seeds should yield such a fast generalization?

Two differences between fast and slow generalization are:

1. For slow generalization, weight norm doesn't change much until some time and then it starts dropping and then generalization is achieved. For fast generalization, generalization is achieved before norm starts going down.

2. For fast generalization, val loss remains more or less same and then there is rapid drop while for slow generalization, validation loss keeps increasing, picks at 3k steps and then starts dropping, going to near 0 at 10k steps.

The #2 above might suggest that slow generalization indeed follows memorization (overfitting) followed by prunning cycle. However, fast generalization doesn't follow that cycle. Another thing is that training loss goes to zero in <40 steps so all memorization happens in rather small window but slow mode still faces val loss keep going up until 3k steps. It's like model has already achieved its goal but then it is heading exact opposite direction of generalization despite of weight decay keep working at it! As loss is 0, LR shouldn't be causing model to memorize even more strongly. Curiously, weight norm keeps going up even after train loss is 0. So what's going on?

One speculaton is that weight decay tries redistribute weights but then sgd immidiately rushes to fill the void. So, weights gets redistributed but memorization stays constant. This keeps going on until weight decay has no more space to restribute and must fight with sgd to force is to generalize.

## 2023-08-31

For a given data loaer seed, there is 0.4% chance that model seed will be good.
For a given model seed, there is 0.6% chance that data loader seed will be good.

## 2023-09-01T05:25:54Z

After various trials, I found that embedding layer, the last linear layer and the 2nd DecoderLayer initialization is what caused faster generalization. However this accounts for val acc of around 0.5 to 0.7 even with data loader seed set to 8. This means there is something else random somewhere but it's hard to tell why.

I have decided to abandon going after figuring out why some seeds work best. My current speculation is that the current model is too small, it's context length
even smaller and data is super small. This means there is only small number of coincidences needed for various randomness in the system to suddenly perform really good. Currently, it seems upto 0.8% of seeds are those super seeds. This would translates to about 7-8 coin tosses where all tosses comes heads. Also, this is very data dependent. So, for certain order of data, there is certain permutation that causes super convergence. However, if data grows, model grows then the number of coins grows and super seeds will be very hard to come by. So, these super seeds are likely the artifacts of small model/small data regimes and they almost never appear otherwise.

As super seeds are data dependent, I wonder if super seeds simply sets up things so that weight decay's redistribution is more efficient and in the right direction. That is, the struggle phase between step 40 to 3k is significantly reduced.

I have also came to decision that this entire synthetic data is bad for testing because (1) it it really just 3 token input, other 2 are redundent, (2) it's not autoregressive. So, I have decided to leave this regim and go back to TinyStories.

A lot of refactoring needs to be done (again!). My precious days seems to be such a waste chasing after this randomness.

## 2023-09-17T03:07:36Z
After 335hr = 14.6 days jobs get terminated. This is 70k steps. So, that's 4.8k steps/day. For WikiText103, val reaches min=22.6 at 4k steps, so within a day. After that val loss rises but train loss keeps getting lower.
https://wandb.ai/microsoft-research-incubation/nanogpt-wikitext103/runs/7308742060.22371-5a834e3a-d277-4556-98f6-f590b7da1848/workspace?workspace=user-sytelus

#### GPT2 results
https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

https://github.com/NVIDIA/Megatron-LM/tree/v1.1

Wikitext103:
37.5 ppl 117M params, 26.37 ppl 345M
117M: 12 layers, d_model=768, heads=16
345M: 24 layers, d_model=1024, heads=16
vocab size: 50257
ctx: 1024
batch: 512

(Trained using webtext)
LR: 1.5x10e-4 / cosine (decay until only 320k steps)
weight decay: 1e-2
grad_clip: 1.0
max_steps=500k
warmup: 5k

1.16B model will take ~4.8k A100/40GB GPU hours.This is 25 days on 8xA100/40GB.
0.124B model should take 2.7 days on 8xA100/40GB


Iteration time for A100 for 1B param model is ~1 sec, leading to ~150TF at context length=1024 and batch=16. This model has d_model=1920, n_heads=15, layers=24.

Rough formulat to calculate training TFLOPs is factor * params * batch * ctx_len. For GPT2, factor seems to be 9 but it should be 6. For 1B GPT2, its 9*1e6*16*1024=147 TFLOPs (actual 149). Forward pass factor is 6 instead of 9.

V100 TFLOPs are almost half so batch size must be reduced to 1/2 (i.e. 8 for GPT2).

Data parallelism can reduce individual flops utilization by upto 30% for 1024 GPUs.

##### GPT-117M estimates
Ref: https://tomekkorbak.com/2022/10/10/compute-optimal-gpt2/#:~:text=Based%20on%20my%20recent%20experiments,small%20on%203.3B%20tokens.

117M params, Chinchilla optimal tokens is 3.3B, flops is 2.4E+18, cost $80
Mistral paper used lr=6e-4, batch=512.

Predicted loss=~3.3 (PPL=24 to 27), however actual train/val loss in GPT2 paper seems to be ~2.8 (PPL=16.1 to 16.3) because may be they trained on 9B+ tokens (more than 3X of Chinchilla). WikiText PPL is 37.5.



#### TransformerXL
Original code: https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/Transformer-XL

Base Model (151M):
d_model=512, n_head=8, d_head=64, d_inner=2048, seq_len=192
dropout=0.1, lr=1e-2, min_lr=1e-3, max_steps=40k, warmup=1k, batch=256
ppl=23.24
    in 75 minutes with 8xA100/40GB with FP16, batch=32/GPU
    in 121 minutes with 8xV100/16GB with FP16, batch=32/GPU
    in 766 minutes with 1xV100/16GB with FP16, batch=32/GPU
    in 80 minutes with 16xV100/32GB with FP16, batch=16/GPU
A100 throughput: 60k tok/s
V100 throughput: 32.8k tok/s

Large model (247M):
d_model=1024, n_head=16, d_head=64, d_inner=4096, seq_len=384
dropout=0.3, lr=1e-2, min_lr=1e-4, max_steps=100k, warmup=16k, batch=128
ppl=18.18 in 430 minutes with 8xA100/40GB with FP16, batch=16/GPU
    in 394 minutes with 16xV100/32GB with FP16, batch=8/GPU
    in 984 minutes with 8xV100/16GB with FP16, batch=4/GPU
A100 throughput: 21.5k tok/s
V100 throughput: 14.8k tok/s

Steps   Base_ppl    Large_ppl
10k     32.3        32.6
20k     26.1        24.1
30k     23.7        21.5
40k     22.9        20.2


StdDev in PPL is very low (<0.2).

#### NanoGPT Reference Run on WT103
https://wandb.ai/microsoft-research-incubation/nanogpt-wikitext103/runs/7308742060.22371-5a834e3a-d277-4556-98f6-f590b7da1848/logs?workspace=user-sytelus

Model (353M):
n_layer=24, n_head=16, n_embd=1024, block_size=1024, bias=False, vocab_size=50304, dropout=0

8xV100, global batch=512, tok/iter=524.288k, train toks=119M

~3 steps/min @ 512 global batch on 8xV100
Best val ppl=22.6 @ 4k steps, 22hr run

#### NanoGPT

OpenWebText: 9B train tokens (54GB text in 8M files)
Initial times 124M model:
    256 A100/40GB / 533 hr with NVidia Megatron June 2021 code?
    Karpathy
        500K iterations  / 8 GPUs / 8*5 microsteps / batch=12
        30M samples/GPU, 240M samples total / 3.7 epochs/GPU (8M docs in dataset)
        30.7B tokens for training
        theoratical time on 8xA100 with flash attn = 1.8 days
        250ms/step without compile, 135ms/step with compile


    65B total training tokens (7 epochs)

### Summary of various reference runs

Fastest smallest GPT2 takes 256hr or ~10 days for a single A100 GPU if we use full OpenWebText.

However, one can use ~1/3rd of OpenWebText (3.3B tokens) per Chinchilla to reduce single A100 training time to 94hr or 4 days.

In our reference run, we used 117M tokens from WT103 and it took 22hr for 8xV100. At batch=512/ctx=1024, 4k steps translates to 2.1B tokens or 17 epochs. This is close to 3.3B tokens suggested by Chinchilla by 30% less, may be, because it's same tokens over and over so much easier to learn. Note that training time for 4k steps should have been 60 A100 GPU hours or 15hr for 8xv100 but for 124M model. For 315M, 22hr is 1.4X longer but model is 2.5X larger.

However, this shows that one must spent 176-256 A100 hours to get lowest ppl on 100M+ token datasets for 124M models.

Directly training on WT103 gets ppl of 22.6 while only OpenWebText without contamination gets us 37.5. Former takes 30% less time but later has no direct knowledge of test set! This indicates, there is likely not a lot of advantage in training only with WT103.

TransformerXL numbers for base 151M model doesn't make sense for 1xV100 but for 247M model it took 16.4hr for 8xV100/16GB which is close to 20.9hr equivalent for 315M model. However, PPL you get is much lower 18.18 instead of ~22. This may be because of hyper param differences. However, batch size number doesn't make sense at 4/GPU which is 8*4=32 global instead of 256.
