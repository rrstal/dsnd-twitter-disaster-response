# Twitter Disaster Response Pipeline Project


<a name="motivation"></a>
## Motivation

This project is part of Data Science Nanodegree Program.
The dataset is provided by Udacity in collaboration with Figure Eight and consists of tweets from life disasters, grouped into 36 distinct categories. The goal is to develop a tool with machine learning model at its core to automatically assign categories to the tweets.

Project phases:

1. ETL Pipeline: data processing that aims at extracting and cleaning the data, then saving it into a database
2. Machine Learning Pipeline: aims at training a model for multi-label classification of the tweets
3. Web App: aims at providing a minimal GUI for the user to interact with the model and to visualize model's predictions.

## Getting Started

### Minimal requirements:
* python 3.x,
* numpy 1.12.1, 
* pandas 0.23.3, 
* scikit-learn 0.19.1,
* nltk 3.2.5,
* sqlalchemy 1.2.18,
* flask 0.12.4,
* plotly 2.0.15.

### Installation:
Clone the following repository:
```
git clone https://github.com/rrstal/dsnd-twitter-disaster-response.git
```

### Execution:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Software structure:
The input data files of tweet messages and categories are located in **data** folder in form of .csv files. The python script in the same directory is for loading, cleaning and saving the data into a database. Lastly an exemplary database file is also located in the directory. The script to train the classifier and the exemplary trained model are located in the folder **models**. The folder **app** contains scripts and .html files necessary to run the web app. Finally **media** is solely for the purposes of this documentation. Structure is as follows:

- **app/**
    - run.py
    - templates/
        - go.html
        - master.html
- **data/** 
    - disaster_categories.csv
    - disaster_messages.csv
    - DisasterResponse.db
    - process_data.py
- **graphics/** 
- **models/**
    - train_classifier.py
    - classifier.pkl 


## Screenshots

1. Upon launching the web app, user can see the following screen. Main page presents graphs conveying information about the training dataset.  

![Sample Input](media/twitter.01.PNG)

2. Once the user inserts a message and clicks on the button **Classify Message**, a list of possible categories unrolls. If the message was categorized to a certain category, this one will be highlighted in light green.

![Sample Output](media/twitter02.PNG)
