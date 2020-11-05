# Disaster Response Pipeline Project

### Table of Contents
1.	[Project Motivation](#project-motivation)
2.	[File Descriptions](#file-descriptions)
3.	[Results](#results)
4.	[Installation](#installation)
5.	[Licensing, Authors, and Acknowledgements](#licensing-authors-and-acknowledgements)

### Project Motivation

The motivation behind this project is to apply data engineering to analyze data proivded by Figure Eight. This analysis includes creating ETL pipeline to prepare the data and store it in a local database to be then used by the ML pipeline to train the classifier and save the results. After that, the model will be presented in an interactive web app. The user will access the page to enter a message that will be automatically classified by the model.


### File Descriptions
There are 3 folders here:
1. data:

process_data.py: This is where the ETL pipeline is created, and this is done by the following: 
- Load the data sets (messages and categories) 
- Merge them and remove duplicates 
- Store the data in a SQLite database

2. models:

train_classifier.py: This is where the ML pipeline is created, and it's done by the following: 
- Load data from SQLite database 
- Process it (tokenized, lemmatized, normalized, striped, and doesn't have stop words) 
- Split it into training and testing sets 
- Build the ML pipeline 
- Train the model using GridSearchCV 
- Evaluate the model and measure its accuracy score 
- Save the model as a pickle file

3. app:

run.py: a Flask web app that allows the user to use the model


### Results
The results can be found here  http://0.0.0.0:3001/, after running the application locally on your workstation.

The web app is as shown below, where you enter any message in the text box, then click: Classify Message button:

![App](../master/images/1.png)

In addition, general visuals on the data are displayed:
1. The first graph shows the genres of the messages with their count
![App](../master/images/V1.png)


2. The second graph shows the categories of the messages with their count
![App](../master/images/V3.png)


3. The graph below shows the count of messages that have specific words
![App](../master/images/V2.png)



Then the app will classify the message into one or more categories. This can be used by the support agencies to identify the right actions. For example, the message entered is: Please help. We need Food, water and medical aid.

The classifications determined by the model are: *Related, Request, Aid Related, Medical Help, Water, Food,and Direct Report.*

![result](../master/images/3.png)

The classification report and accuracy score (per category) of the model are shown in the screenshots below:
![classification](../master/images/4.png)

![accuracy](../master/images/5.png)





### Installation Instructions
1. Download project files:

Download the project files to your local machine, and make sure you keep the same folder structure.

2. Clean the data: 

Run the following command in the project's root directory to run ETL pipeline that cleans data and stores in database:
    
    python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

3. Build the model:

Run the following command in the project's root directory to run ML pipeline that trains classifier and saves it:
    
    python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

4. Start the web app:

Run the following command in the app's directory to run your web app:

    python run.py


5. Access the web app:

Go to http://0.0.0.0:3001/

and the git repository is: https://github.com/se-Laila/DRP.git


### Licensing, Authors, Acknowledgements
Credit goes to Figure Eight for the data and Udacity for the HTML pages. 
