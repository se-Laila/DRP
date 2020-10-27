# Disaster Response Pipeline Project

### Table of Contents
1.	[Project Motivation](#project-motivation)
2.	[File Descriptions](#file-descriptions)
3.	[Results](#results)
4.	[Installation](#installation)
5.	[Licensing, Authors, and Acknowledgements](#licensing-authors-and-acknowledgements)

### Project Motivation

The motivation behind this project is to apply datat engineering to analyze data proivded by Figure Eight. This analysis includes
creating an ETL pipeline to prepare the data and store it in a local database to be then used by the ML pipleine to train the classifier and save the results. After that, the model will be presented in an interactive web page. The user will access the page to enter a message that will be automatically categorized by the model.


### File Descriptions
There are 3 folders here:
1. app
    run.py
    templates folder has the HTML page
2. data: 
    process_data.py
3. models:
    train_classifier.py


### Results
The results can be found here  http://0.0.0.0:3001/, after running the application locally on your workstation



### Installation Instructions: 
1. Download project files:
Download the project files to your local machine, and make sure you keep the same folder structure.

2. Clean the data: 
    Run the following command in the project's root directory to run ETL pipeline that cleans data and stores in database:
        python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

    data/process_data.py >> is the ETL pipleine
    data/disaster_messages.csv data/disaster_categories.csv
    
3. Build the model:
    Run the following command in the project's root directory to run ML pipeline that trains classifier and saves it:
        python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

4. Start the web app:
    Run the following command in the app's directory to run your web app:
        python run.py

5. Access the web app:
    Go to http://0.0.0.0:3001/


### Licensing, Authors, Acknowledgements
Credit goes to Figure Eight for the data and Udacity for the HTML pages. 