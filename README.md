[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/ranjan/playground/main/app.py)

# Machine Learning Playground ⚗️👨🏻‍💻

Machine Learning Playground is a **streamlit** application that allows you to tinker with machine learning models from your browser.

So if you're a data science practitioner you should definitely try it out 😉

_This app is made after inspired by the great Tensorflow [playground](https://playground.tensorflow.org/)._

## Demo

Click [here](https://mlplayground.herokuapp.com/)

## How does it work ?

1. 📂 Choose a dataset
2. ⚙️ Pick a model and set its hyper-parameters
3. 📈 Train it and check its performance metrics and decision boundary on train and test data
4. 🔬 Inspect the changes occured by selecting other possible models and hyper-parameters using other settings
5. 🥇 Compare the accuracy and shape of the decision boundary introduced by every model with all possible settings
6. 🕵🏻 Make a decision on the perfect model🏆 along with respective hyper-parameters

**Bonus point**: This app can show corresponding dataset on which the training and testing are to be performed along with it also provides the ability to perform feature engineering by adding polynomial features.

## What can you learn from playground?

If you're new to machine learning, playing with this app will probably (and hopefully 😄) get you familiar with fundamental concepts and help you build your first intuitions.

### 1. Decision boundaries will (partially) tell you how models behave

### 2. You'll get a sense of the speed of each model while training them

### 3. Feature engineering can help

### 4. Some models are more robust than others to noise

### 5. Try out different combinations of hyper-parameters


Go, and give it a try, and I hope you'll learn something from it!

## Run the app locally

Make sure you have pip installed with Python 3.

- install pipenv

```shell
pip install pipenv
```

- go inside the folder and install the dependencies

```shell
pipenv install
```

- run the app

```shell
streamlit run app.py
```

## Structure of the code

- `app.py` : The main script to start the app or the entry point of the server
- `utils/`
  - `ui.py`: UI functions to display the different components of the app
  - `functions.py`: for data processing, training the model and building the plotly graphs
- `models/`: where each model is defined and constructed along with its hyper-parameters

## Contributions are welcome!

Feel free to open a pull request or an issue if you're thinking of a feature you'd like to introduce in the app such as:

- [ ] Adding other non-linear datasets
- [ ] Adding more models
- [ ] Implementing sophisticated feature engineering (sinusoidal features for instance)
- [ ] Implementing a custom dataset reader with dimensionality reduction
- [ ] Adding feature importance plots

But if you've got other ideas, I will be happy to discuss them with you.
