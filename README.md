# IR Learning-to-rank recommendation models

### MF with BCE model

- Public MAP score: 0.05353

- Loss function:
  Each (user, item, attr) pair indicates the relation of the user(0 ~ num_user - 1) to item(0 ~ num_item - 1) is attr (1 or 0).
  Let ![1](http://latex.codecogs.com/svg.latex?{\text{prediction}=\text{userEmbedding[user]}\cdot\text{itemEmbedding[item]}}) for each pair.

  So for one pair, the loss function ![2](http://latex.codecogs.com/svg.latex?L) will be: ![3](http://latex.codecogs.com/svg.latex?-(\text{attr}\times\log\sigma(\text{prediction})+(1-\text{attr})\times\log(1-\sigma(\text{prediction})))))

  ![4](http://latex.codecogs.com/svg.latex?{\text{total loss}=\displaystyle \sum_{\text{(user, item, attr)}\in\text{Data}} \dfrac{L}{\text{Data size}}})

- Parameters: 
  Epoch = 10, Batch size = 4096, Hidden dim = 512, Optimizer = AdamW and SGD (switch at the 5th epoch).

- Negative sample method:
  Pasitive ratio : Negative ratio = 1 : 5.
  Maintained a 2D array named **total_negative** that stored all the missing values for the users in the format (user, item, 0). That is, total_negative[user] was a array that stored elements like (user, item1, 0), (user, item2, 0), ..., (user, item![5](http://latex.codecogs.com/svg.latex?_\text{k}), 0).
  Each time a new epoch started, just used random.sample to sample 5 times more than positive data of each user from total_negative, and then append them to the dataset.


There's no need to check if the samples for validation and the samples for training is disjoint because the the performance will be littlely affect.

---

### MF with BPR model

- Public MAP score: 0.05847

- Loss function:
  Each (user, pos_item, neg_item) pair provides one positive item and one negative item for the user to calculate the loss.

  Let ![6](http://latex.codecogs.com/svg.latex?\left\{\begin{array}{c}\text{pos_prediction}=\text{userEmbedding[user]} \cdot \text{itemEmbedding[pos_item]}\\\text{neg_prediction}=\text{userEmbedding[user]} \cdot \text{itemEmbedding[neg_item]}\end{array}\right.) for each pair.

  So for one pair, the loss function ![7](http://latex.codecogs.com/svg.latex?L) will be: ![8](http://latex.codecogs.com/svg.latex?-\ln\sigma(\text{pos_prediction} - \text{neg_prediction}))

  ![9](http://latex.codecogs.com/svg.latex?{\text{total loss} = \displaystyle \sum_{\text{(user, pos_item, neg_item)}\in\text{Data}}\dfrac{L}{\text{Data size}}})

- Parameters:
  Epoch = 30, Batch size = 4096, Hidden dim = 512, Optimizer = AdamW and SGD (switch at the 10th epoch).

- Negative sample method:
  Pasitive ratio : Negative ratio = 1 : 5.
  Similar to BCE sampling method, I used **total_negative** to sample the negative data.
  However, BPR required that each pair to be in the format (user, pos_item, neg_item), so I couldn't get the samples for one user by sampling once. Instead, I had to sample 5 negative pairs for each positive pair, and append them to the data within the correct format.

  
  In BPG, I didn't even check whether the samples are disjoint in training data because it's already powerful enough, and it did outperform BCE.


---

### Result comparison




