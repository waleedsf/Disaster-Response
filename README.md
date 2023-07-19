# Disaster-Response-pipeline project
The goal of this project is to analyze disaster data and develop a model that can classify disaster messages. The dataset consists of real messages sent during various disaster events. To achieve this, a machine learning pipeline is built to categorize the messages and determine the appropriate disaster relief agency to send them to.

Additionally, a web application is created as part of the project. This application allows emergency workers to input new messages and receive classification results across multiple categories. This way, they can quickly identify the relevant agencies and take appropriate action based on the message content.

![image](https://github.com/waleedsf/Disaster-Response-pipeline/assets/86128909/0fb09145-25a1-4778-b0e5-5dd0e2267c3d)

Instructions:
Run the following commands in the project's root directory to set up your database and model:

- To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
- To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

- Run the following command in the app's directory to run your web app. python run.py
