import numpy as np
import streamlit as st


from models.NaiveBayes import nb_param_selector
from models.NeuralNetwork import neural_network_param_selector
from models.RandomForest import random_forest_param_selector
from models.DecisionTree import decision_tree_param_selector
from models.LogisticRegression import logistic_regression_param_selector
from models.KNearestNeighbors import knn_param_selector
from models.SVC import svc_param_selector
from models.GradientBoosting import gradient_boosting_param_selector

from models.utils import model_imports
from utils.functions import img_to_bytes


'''Function to design the header of the webpage'''
def introduction():
    st.title("**Welcome to ML playground ⚗️👨🏻‍💻**")
    st.subheader(
        """
        Hey! Do you wanna tinker 🧪 with machine learning models directly from your browser?? 🌎
        Follow the steps -
        """
    )

    st.markdown(
        """
    - 📂 Choose a dataset
    - ⚙️ Pick a model and set its hyper-parameters
    - 📈 Train it and check its performance metrics and decision boundary on train and test data
    - 🔬 Inspect the changes occured by selecting other possible models and hyper-parameters using other settings
    - 🥇 Compare the accuracy and shape of the decision boundary introduced by every model with all possible settings
    - 🕵🏻 Make a decision on the perfect model🏆 along with respective hyper-parameters
    -----
    """
    )


'''Function to choose the dataset with setting up the train and test noise effectively'''
def dataset_selector():
    dataset_container = st.sidebar.beta_expander("Configure a dataset", True)
    with dataset_container:
        dataset = st.selectbox("Choose a dataset", ("moons", "circles", "blobs"))
        n_samples = st.number_input(
            "Number of samples",
            min_value=50,
            max_value=1000,
            step=10,
            value=300,
        )

        train_noise = st.slider(
            "Set the noise (for train data)",
            min_value=0.01,
            max_value=0.2,
            step=0.005,
            value=0.06,
        )
        test_noise = st.slider(
            "Set the noise (for test data)",
            min_value=0.01,
            max_value=1.0,
            step=0.005,
            value=train_noise,
        )

        if dataset == "blobs":
            n_classes = st.number_input("centers", 2, 5, 2, 1)
        else:
            n_classes = None

    return dataset, n_samples, train_noise, test_noise, n_classes


'''Function to pick a model for training and predicting the model performance'''
def model_selector():
    model_training_container = st.sidebar.beta_expander("Train a model", True)
    with model_training_container:
        model_type = st.selectbox(
            "Choose a model",
            (
                "Logistic Regression",
                "Decision Tree",
                "Random Forest",
                "Gradient Boosting",
                "Neural Network",
                "K Nearest Neighbors",
                "Gaussian Naive Bayes",
                "SVC",
            ),
        )

        if model_type == "Logistic Regression":
            model = logistic_regression_param_selector()

        elif model_type == "Decision Tree":
            model = decision_tree_param_selector()

        elif model_type == "Random Forest":
            model = random_forest_param_selector()

        elif model_type == "Neural Network":
            model = neural_network_param_selector()

        elif model_type == "K Nearest Neighbors":
            model = knn_param_selector()

        elif model_type == "Gaussian Naive Bayes":
            model = nb_param_selector()

        elif model_type == "SVC":
            model = svc_param_selector()

        elif model_type == "Gradient Boosting":
            model = gradient_boosting_param_selector()

    return model_type, model


'''Function to generate different code snippet correspond to different machine learning models which can be reused in other windows'''
def generate_snippet(
    model, model_type, n_samples, train_noise, test_noise, dataset, degree
):
    train_noise = np.round(train_noise, 3)
    test_noise = np.round(test_noise, 3)

    model_text_rep = repr(model)
    model_import = model_imports[model_type]

    if degree > 1:
        feature_engineering = f"""
    >>> for d in range(2, {degree+1}):
    >>>     x_train = np.concatenate((x_train, x_train[:, 0] ** d, x_train[:, 1] ** d))
    >>>     x_test= np.concatenate((x_test, x_test[:, 0] ** d, x_test[:, 1] ** d))
    """

    if dataset == "moons":
        dataset_import = "from sklearn.datasets import make_moons"
        train_data_def = (
            f"x_train, y_train = make_moons(n_samples={n_samples}, noise={train_noise})"
        )
        test_data_def = f"x_test, y_test = make_moons(n_samples={n_samples // 2}, noise={test_noise})"

    elif dataset == "circles":
        dataset_import = "from sklearn.datasets import make_circles"
        train_data_def = f"x_train, y_train = make_circles(n_samples={n_samples}, noise={train_noise})"
        test_data_def = f"x_test, y_test = make_circles(n_samples={n_samples // 2}, noise={test_noise})"

    elif dataset == "blobs":
        dataset_import = "from sklearn.datasets import make_blobs"
        train_data_def = f"x_train, y_train = make_blobs(n_samples={n_samples}, clusters=2, noise={train_noise* 47 + 0.57})"
        test_data_def = f"x_test, y_test = make_blobs(n_samples={n_samples // 2}, clusters=2, noise={test_noise* 47 + 0.57})"

    snippet = f"""
    >>> {dataset_import}
    >>> {model_import}
    >>> from sklearn.metrics import accuracy_score, f1_score
    >>> {train_data_def}
    >>> {test_data_def}
    {feature_engineering if degree > 1 else ''}    
    >>> model = {model_text_rep}
    >>> model.fit(x_train, y_train)
    
    >>> y_train_pred = model.predict(x_train)
    >>> y_test_pred = model.predict(x_test)
    >>> train_accuracy = accuracy_score(y_train, y_train_pred)
    >>> test_accuracy = accuracy_score(y_test, y_test_pred)
    """
    return snippet


'''Function to set up the highest degree of the polynomial for a particular model as part of Feature Engineering'''
def polynomial_degree_selector():
    return st.sidebar.number_input("Highest polynomial degree", 1, 10, 1, 1)


'''Function to design the footer of the webpage'''
def footer():
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        [<img src='data:image/png;base64,{}' class='img-fluid' width=25 height=25>](https://github.com/ranjan-das-cloud/machine-learning-playground) <small> Version 0.1.0 &copy;Ranjan Das | July 2021</small>""".format(
            img_to_bytes("./images/github.png")
        ),
        unsafe_allow_html=True,
    )
