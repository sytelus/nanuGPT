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