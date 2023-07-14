# Parameters of the XGBoost model

## Booster parameters

### booster

* The type of model to tune at each iteration
* Default: ```None```
* Options: ```"gbtree"```, ```"dart"``` or ```"gblinear"```
* ```"gbtree"``` and ```"dart"``` use tree-based models
* ```"gblinear"``` uses a linear model
* ```"gbtree"``` and ```"dart"``` always outperform ```"gblinear"```

### verbosity

* Prints messages to the console for debugging purposes.
* Default: ```0```
* Options: ```0``` (silent), ```1``` (warning), ```2``` (info), ```3``` (debug)

### nthread 

* Number of parallel threads used to run XGBoost
* Default: maximum number of threads available if not set
* Used for parallel processing

### eta

* Similar to learning rate in GBM
* Default : ```0.3```
* Range : ```[0,1]```
* AKA: ```learning_rate```
* It's the step size shrinkage used in each iteration to prevent overfitting
* After each iteration, we get the weights of the new features, and eta shrinks the feature weights to keep the boosting process under control

### gamma 

* Specifies the minimum loss reduction required to make a split (nodes are split only when the split gives a reduction in the loss function)
* Default : ```0```
* Range: [0,∞]
* AKA: ```min_split_loss```
* The bigger gamma is, the more conservative the model will be

### max_depth 

* Maximum depth of a tree
* Default : ```6```
* Range : ```[0,∞]```
* Lower depths decrease complexity and the chance of overfitting
* ```0``` is only valid when ```tree_method="hist"``` and it indicates no limit on depth
* Deep trees consume lots of memory, so be careful
* Should be tuned with cross validation

### min_child_weight

* Minimum sum of weights of all observations required in a child
* Default : ```1```
* Range : ```[0,∞]```
* Used to control over-fitting
* Higher values = higher chance of under-fitting, and vice-versa
* Higher values stop a model from "learning" relationships which might only be true for a particular sample selected for a tree
* Tuned using cross validation

### max_delta_step 

* The maximum delta that each tree's weight estimation is allowed to be
* Default : ```0```
* Range : ```[0,∞]```
* If its ```0```, there is no constraint on each tree's weight estimation
* Higher values = higher chance of under-fitting
* Should only consider it in logistic regression when classes are extremely imbalanced

### subsample 

* The fraction of observations to be randomly samples for each tree
* Default : ```1```
* Range : ```(0,1]```
* If you set it to 0.5, that means XGBoost will randomly sample half the training data before growing trees
    * This helps prevent overfitting
* Occurs once per boosting iteration
* Lower values = higher chance of under-fitting

### colsample_bytree, colsample_bylevel, colsample_bynode 

* Family of parameters for subsampling of columns
* Default : ```1```
* Range : ```(0, 1]```
* ```colsample_bytree```: The subsample ratio of columns when constructing each tree (subsampling occurs once for every tree constructed)
* ```colsample_bylevel```: The subsample ratio of columns for each level. Subsampling occurs once for each new depth level reached in a tree. Columns are subsampled from the set of columns chosen for the current tree.
* ```colsample_bynode```: The subsample ratio of columns for each node (split). Subsampling occurs once every time a new split is evaluated. Columns are subsampled from the set of columns chosen for the current level.
* ```colsample_by*```: Allows you to use a combination of ```colsample_bytree```, ```colsample_bylevel```, or ```colsample_bynode```. For example: ```{colsample_bytree: 0.75, colsample_bylevel: 0.75}``` with 16 features leaves 9 features to choose from at each split 16 * ((3/4)*(3/4)) = 9

### lambda

* L2 regularization on weights (similar to ridge regression)
* Default : ```1```
* Range : ```[1,∞]```
* AKA: ```reg_lambda```
* Handles the regularization part of XGBoost
* Higher values = higher chance of under-fitting

### alpha 

* L1 regularization term on weights (similar to Lasso regression)
* Default : ```0```
* Range : ```[1,∞]```
* AKA: ```reg_alpha```
* Can be used for high-dimensionality situations so the model can run faster
* Higher values = higher chance of under-fitting

### tree_method

* The tree construction algorithm used in XGBoost
* Default : ```"auto"```
* Options : ```"auto", "exact", "approx", "hist", "gpu_hist"```
* ```"auto"```: Use heuristic to choose the fastest method
* ```"exact"```: Exact greedy algorithm
* ```"approx"```: Approximate greedy algorithm using gradient histogram and quantile sketch, will be chosen for very large datasets
* ```"hist"```: Fast histogram optimized approximate greedy algorithm
* ```"gpu_hist"```: GPU implementation of ```"hist"```

### scale_pos_weight 

* Controls the balance of positive and negative weights
* Default : ```1```
* Range : ``````
* Useful for imbalanced classes
* A good rule of thumb: ```sum(negative_instances) / sum(positive_instances)```

### max_leaves 

* Maximum number of nodes to be added
* Default : ```0```
* Only used when ```grow_policy="lossguide"```

## Learning task parameters

### objective 

* Defines the loss function to be minimized
* Default : ```reg:squarederror```
* Options
    * ```reg:squarederror``` : regression with squared loss
    * ```reg:squaredlogerror```: regression with squared log loss 1/2[log(pred+1)−log(label+1)]2
        * All input labels must be greater than -1
    * ```reg:logistic``` : logistic regression
    * ```binary:logistic``` : logistic regression for binary classification, output probability
    * ```binary:logitraw```: logistic regression for binary classification, output score before logistic transformation
    * ```binary:hinge``` : hinge loss for binary classification
        * Makes predictions of 0 or 1, rather than probabilities
    * ```multi:softmax``` : Sets XGBoost to do multiclass classification using the softmax objective
        * Need to set ```num_class(number_of_classes)```
    * ```multi:softprob``` : same as ```softmax```, but outputs a vector of ndata nclass, which can be further reshaped to ndata nclass matrix
        * The result contains the predicted probability of each data point belonging to each class

### eval_metric 

The default values are rmse for regression, error for classification and mean average precision for ranking.
We can add multiple evaluation metrics.
Python users must pass the metrics as list of parameters pairs instead of map.
The most common values are given below -

rmse : root mean square error
mae : mean absolute error
logloss : negative log-likelihood
error : Binary classification error rate (0.5 threshold). It is calculated as #(wrong cases)/#(all cases). For the predictions, the evaluation will regard the instances with prediction value larger than 0.5 as positive instances, and the others as negative instances.
merror : Multiclass classification error rate. It is calculated as #(wrong cases)/#(all cases).
mlogloss : Multiclass logloss
auc: Area under the curve
aucpr : Area under the PR curve

* metric to be used for validation data
* Default : ```"rmse"``` for regression, ```"error"``` for classification and mean average precision for ranking
* Options
    * 

### name 

* 
* Default : ``````
* Range : ``````
* AKA: ``````

### name 

* 
* Default : ``````
* Range : ``````
* AKA: ``````

### name 

* 
* Default : ``````
* Range : ``````
* AKA: ``````
