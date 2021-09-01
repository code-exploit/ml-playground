model_imports = {
    "Logistic Regression": "from sklearn.linear import LogisticRegression",
    "Decision Tree": "from sklearn.tree import DecisionTreeClassifier",
    "Random Forest": "from sklearn.ensemble import RandomForestClassifier",
    "Gradient Boosting": "from sklearn.ensemble import GradientBoostingClassifier",
    "Neural Network": "from sklearn.neural_network import MLPClassifier",
    "K Nearest Neighbors": "from sklearn.neighbors import KNeighborsClassifier",
    "Gaussian Naive Bayes": "from sklearn.naive_bayes import GaussianNB",
    "SVC": "from sklearn.svm import SVC",
}


model_urls = {
    "Logistic Regression": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html",
    "Decision Tree": "https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html",
    "Random Forest": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html",
    "Gradient Boosting": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html",
    "Neural Network": "https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html",
    "K Nearest Neighbors": "https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html",
    "Gaussian Naive Bayes": "https://scikit-learn.org/stable/modules/naive_bayes.html",
    "SVC": "https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html",
}


model_infos = {
    "Logistic Regression": """
        - Logistic Regression is a core supervised learning technique for solving **classification problems**
        - A logistic regression is only suited to **linearly separable** problems
        - The key representation in logistic regression are the coefficients, just like linear regression
        - The coefficients in logistic regression are estimated using a process called maximum-likelihood estimation
        - It's computationally fast and interpretable by design
        - It can handle non-linear datasets with appropriate feature engineering
        - It is widely adopted in real-life machine learning production settings for several reasons like Ease of use, Interpretability, Scalability, Real-time predictions.
    """,
    "Decision Tree": """
        - Decision Tree models are created using 2 steps: Induction and Pruning
        - Induction is where we actually build the tree i.e set all of the hierarchical decision boundaries based on our data
        - Pruning is the process of removing the unnecessary structure from a decision tree, effectively reducing the complexity to combat overfitting with the added bonus of making it even easier to interpret
        - Decision trees are simple to understand and intrepret
        - They are prone to overfitting when they are deep (high variance)

        😃 ADVANTAGES -
        - Easy to understand and interpret
        - Require very little data preparation
        - The cost of using the tree for inference is logarithmic in the number of data points used to train the tree

        ☹️ DISADVANTAGES -
        - Overfitting is quite common with decision trees simply due to the nature of their training
        - For similar reasons as the case of overfitting, decision trees are also vulnerable to becoming biased to the classes that have a majority in the dataset
    """,
    "Random Forest": """
        😃 ADVANTAGES -
        - Random Forest has a high accuracy than other algorithms
        - They are robust to outliers
        - They are computationally intensive and runs efficiently on large datasets
        - It has an effective method for estimating missing data and maintains accuracy when a large proportion of the data are missing

        ☹️ DISADVANTAGES -
        - Random forests may result in overfitting for some datasets with noisy regression tasks
        - They are not easily interpretable
        - For data with categorical variables having a different number of levels, random forests are found to be biased in favor of those attributes with more levels
    """,
    "Gradient Boosting": """
        - Boosting is a sequential technique which works on the principle of ensemble. It combines a set of weak learners and delivers improved prediction accuracy
        - Gradient boosting combines decision trees in an additive fashion from the start
        - Gradient boosting builds one tree at a time sequentially
        - Carefully tuned gradient boosting can result in better performance than random forests

        😃 ADVANTAGES -
        - Often provides predictive accuracy that cannot be beat
        - Lots of flexibility - can optimize on different loss functions and provides several hyperparameter tuning options that make the function fit very flexible
        - No data pre-processing required - often works great with categorical and numerical values as is
        - Handles missing data - imputation not required

        ☹️ DISADVANTAGES -
        - GBMs will continue improving to minimize all errors. This can overemphasize outliers and cause overfitting
        - Computationally expensive - GBMs often require many trees (>1000) which can be time and memory exhaustive
        - The high flexibility results in many parameters that interact and influence heavily the behavior of the approach (number of iterations, tree depth, regularization parameters, etc.). This requires a large grid search during tuning
    """,
    "Neural Network": """
        - Neural network models are trained using stochastic gradient descent and model weights are updated using the backpropagation algorithm
        - Neural Networks have great representational power but overfit on small datasets if not properly regularized
        - They have many parameters that require tweaking
        - They are computationally intensive on large datasets
        - To improve neural network, the steps are very useful -
            1. Increase hidden Layers
            2. Change Activation function
            3. Change Activation function in Output layer
            4. Increase number of neurons
            5. Weight initialization
            6. More data
            7. Normalizing/Scaling data
    """,
    "K Nearest Neighbors": """
        - KNNs are intuitive and simple. They can also handle different metrics
        - KNNs don't build a model per se. They simply tag a new data based on the historical data
        - They become very slow to predict any test data sample as the dataset size grows because of they do not engage training dataset to train the model seperately and that's why called **'The Lazy Learner'**

        😃 ADVANTAGES -
        - No training time is required
        - Its simple and easy to implement
        - New data points can be added to the train data set at any time since model training is not required

        ☹️ DISADVANTAGES -
        - Require feature scaling
        - Does not work well when the dimensions are high
        - Sensitive to outliers
        - Prediction is computationally expensive as we need to compute the distance between the point under consideration and all other points
    """,
    "Gaussian Naive Bayes": """
        - Naive Bayes is a simple and powerful technique that you should be testing and using on your classification problems
        - Naive Bayes learners and classifiers can be extremely fast compared to more sophisticated methods
        - Naive Bayes methods are a set of supervised learning algorithms based on applying Bayes’ theorem with the “naive” assumption of conditional independence between every pair of features given the value of the class variable
        - GaussianNB implements the Gaussian Naive Bayes algorithm for classification. The likelihood of the features is assumed to be Gaussian
        - It works well with high-dimensional data such as text classification problems quite well in many real-world situations, famously document classification and spam filtering
        - They require a small amount of training data to estimate the necessary parameters
        - The assumption that all the features are independent is not always respected in real-life applications

        😃 ADVANTAGES -
        - This algorithm works quickly and can save a lot of time
        - Naive Bayes is suitable for solving multi-class prediction problems
        - If its assumption of the independence of features holds true, it can perform better than other models and requires much less training data
        - Naive Bayes is better suited for categorical input variables than numerical variables

        ☹️ DISADVANTAGES -
        - Naive Bayes assumes that all predictors (or features) are independent, rarely happening in real life
        - This algorithm faces the ‘zero-frequency problem’ where it assigns zero probability to a categorical variable whose category in the test data set wasn’t available in the training dataset
        - Its estimations can be wrong in some cases, so you shouldn’t take its probability outputs very seriously
    """,
    "SVC": """
        - Support vectors are data points that are closer to the hyperplane and influence the position and orientation of the hyperplane
        - SVMs or SVCs are effective when the number of features is larger than the number of samples
        - SVC, NuSVC and LinearSVC are classes capable of performing binary and multi-class classification on a dataset
        - They provide different type of kernel functions
        - They require careful normalization

        Goal - 
        - The SVM should maximize the distance between the two decision boundaries. Mathematically, this means we want to maximize the distance between the hyperplane defined by wTx+b=−1 and the hyperplane defined by wTx+b=1
        - The SVM should also correctly classify all x(i), which means y(i)(wTx(i)+b)≥1,∀i∈{1,…,N}

        😃 ADVANTAGES -
        - Effective in high dimensional spaces
        - Still effective in cases where number of dimensions is greater than the number of samples
        - Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient

        ☹️ DISADVANTAGES -
        - If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial
        - SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation
    """,
}