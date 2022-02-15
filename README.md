# Disaster Response Pipeline Project
### Description:
This project is part of Data Science Nanodegree Program by Udacity. The goal is to build a NLP model that classify pre-labeled disaster messages, so that the messages
can be sent to the relevant organization. 

The project consists of 3 parts:
    <ol>
    <li>Data Processing - prepare an ETL pipeline to extract data from sources, transform to the proper format, and load to a database</li>
    <li>Machine Learning Pipeline - build a NLP pipeline to classify the text messages </li>
    <li>Web App - build a web app to view and classify in real time</li>
    </ol>

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Go to http://0.0.0.0:3001/
