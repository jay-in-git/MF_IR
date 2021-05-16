# IR Learning-to-rank recommendation models

### MF with BCE model

- Public MAP score: 0.05353

- Loss function:
  Each (user, item, attr) pair indicates the relation of the user(0 ~ num_user - 1) to item(0 ~ num_item - 1) is attr (1 or 0).
  Let ![](http://latex.codecogs.com/svg.latex?{\text{prediction}=\text{userEmbedding[user]}\cdot\text{itemEmbedding[item]}}) for each pair.

  So for one pair, the loss function ![](http://latex.codecogs.com/svg.latex?L) will be: ![](http://latex.codecogs.com/svg.latex?-(\text{attr}\times\log\sigma(\text{prediction})+(1-\text{attr})\times\log(1-\sigma(\text{prediction})))))

  ![](http://latex.codecogs.com/svg.latex?{\text{totalLoss}=\displaystyle\sum_{\text{(user,item,attr)}\in\text{Data}}\dfrac{L}{\text{DataSize}}})

- Parameters: 
  Epoch = 10, Batch size = 4096, Hidden dim = 512, Optimizer = AdamW and SGD (switch at the 5th epoch).

- Negative sample method:
  Pasitive ratio : Negative ratio = 1 : 5.
  Maintained a 2D array named **total_negative** that stored all the missing values for the users in the format (user, item, 0). That is, total_negative[user] was a array that stored elements like (user, item1, 0), (user, item2, 0), ..., (user, item![](http://latex.codecogs.com/svg.latex?_\text{k}), 0).
  Each time a new epoch started, just used random.sample to sample 5 times more than positive data of each user from total_negative, and then append them to the dataset.


There's no need to check if the samples for validation and the samples for training is disjoint because the the performance will be littlely affect.

---

### MF with BPR model

- Public MAP score: 0.05847

- Loss function:
  Each (user, posItem, negItem) pair provides one positive item and one negative item for the user to calculate the loss.

  Let ![](https://latex.codecogs.com/svg.latex?\left\{\begin{aligned}\text{posPrediction}=\text{userEmbedding[user]}\cdot\text{itemEmbedding[posItem]}\\\\\text{negPrediction}=\text{userEmbedding[user]}\cdot\text{itemEmbedding[negItem]}\end{aligned}\right.) for each pair.

  So for one pair, the loss function ![](http://latex.codecogs.com/svg.latex?L) will be: ![](http://latex.codecogs.com/svg.latex?-\ln\sigma(\text{posPrediction}-\text{negPrediction}))

  ![](http://latex.codecogs.com/svg.latex?{\text{totalLoss}=\displaystyle\sum_{\text{(user,posItem,negItem)}\in\text{Data}}\dfrac{L}{\text{DataSize}}})

- Parameters:
  Epoch = 30, Batch size = 4096, Hidden dim = 512, Optimizer = AdamW and SGD (switch at the 10th epoch).

- Negative sample method:
  Pasitive ratio : Negative ratio = 1 : 5.
  Similar to BCE sampling method, I used **total_negative** to sample the negative data.
  However, BPR required that each pair to be in the format (user, pos_item, neg_item), so I couldn't get the samples for one user by sampling once. Instead, I had to sample 5 negative pairs for each positive pair, and append them to the data within the correct format.

  
  In BPG, I didn't even check whether the samples are disjoint in training data because it's already powerful enough, and it did outperform BCE.


---

### Result comparison

---

### MAP value under different hidden factor

The points are (16, 0.04791), (32, 0.04765), (64, 0.04975), and (128, 0.05157). Besides, I tested the MAP value for d=512 and got (512, 0.05498).

The experiment is under the condition decribed following:

- optimizer: SGD
- Random seed: 42069
- Loss function: BPR
- Learning rate: 0.005
- negative sample ratio: 5
- 50 epochs in total

![](https://imgur.com/mKHufYc.png)

With the hidden dimension increases, the model becomes more powerful. However, there're overfits when I trained the model with higher hidden dimension, which are not shown in the curve. The curve can't show the overfit results because the program only saves the model with the least BPR loss on validation data, so that the overfitting models will not be used to generate prediction.

The decreasing in MAP value between d = 16 and d = 32 may due to that 16 and 32 are not significantly different. Thus,  when d = 32, the MAP value may be affected by random seed or several reasons, leading to lower value.


