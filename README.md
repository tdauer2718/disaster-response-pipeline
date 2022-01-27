### Table of Contents

- [Installation <a name="installation"></a>](#installation-)
- [Motivation <a name="motivation"></a>](#motivation-)
- [File descriptions <a name="files"></a>](#file-descriptions-)
- [Instructions for reproducing results <a name="instructions"></a>](#instructions-for-reproducing-results-)
- [Results <a name="results"></a>](#results-)
- [Licensing and acknowledgements <a name="acknowledgements"></a>](#licensing-and-acknowledgements-)

## Installation <a name="installation"></a>

See the packages and version numbers listed in 'requirements.txt' inside the 'app' directory. In particular, Scikit-Learn will need to be updated if you have a version older than 0.22.2.

## Motivation <a name="motivation"></a>

Dealing appropriately with messages during a disaster is a challenge; there are many messages coming in at once to a limited number of people with the expertise and connections to read and respond to them appropriately, and at any given time many of the messages could be unrelated to the disaster. A good machine learning model can help workers deal with the flood of messages more efficiently by automatically determining whether they are related to the disaster and classifying them based on the category of relief required. Building such a machine learning pipeline is my goal in this project.

## File descriptions <a name="files"></a>

I have divided the files into three directories: "data", "models", and "app".

In the "data" directory, I have included the raw message data in "messages.csv" and the data about which categories these messages fall into in "categories.csv". The Jupyter notebook "ETL Pipeline Preparation.ipynb" shows my work in preparing an ETL pipeline for this data, and "process_data.py" is the ETL pipeline script itself. Finally, the pipeline loads the data into a SQL database named "data.db".

In the "models" directory, I have included the Jupyter notebook "ML Pipeline Preparation.ipynb" that shows my work in preparing the machine learning model. The script "train_classifier.py" trains, evaluates, and saves the model with hyperparameters I determined in the Jupyter notebook.

The "app" directory is where the files for creating the web app are located. The files inside the "templates" directory are HTML files required for this, and "run.py" is a script for launching the web app. The file "requirements.txt" gives the required packages and versions for this project. I note that most of the code in the "app" directory comes from Udacity; I am quite new to web development and did little to modify the basic structure that they generously provided.

## Instructions for reproducing results <a name="instructions"></a>

What follows are instructions for reproducing my results and running the web app.

First, download this repo to your local machine and cd into the top-level directory ("disaster-response-pipeline"). Then run the ETL pipeline script using the command

`python data/process_data.py data/messages.csv data/categories.csv data/data.db`

This will save the data to data.db in the "data" directory. Then run the script to train, test, and save the model using the command

`python models/train_classifier.py data/data.db models/final_model.pkl`

This will save the model to final_model.pkl in the "models" directory. Next, start the web app by using the command

`python app/run.py`

Then open a browser and go to http://0.0.0.0:3001, where you can view my visualizations and see how the model classifies messages you type in.

## Results <a name="results"></a>

The results of evaluating the model on a test set are printed to the terminal when train_classifier.py is run. These results include the F1-score, precision, and recall for Class 1 for each category. I have also included the fraction of instances in the test set that are in Class 1 for each category.

In most of the classes, the model has at least some predictive power; it generally finds some of the messages in the category (nonzero recall, and for many classes more like 80% recall) and has precision over 50% for its positive predictions, meaning that the disaster response professionals who read positive results from this model don't get more than half false positives.

However, there's a lot left to be desired. Since in many categories, such as "security" and "medical help", the recall is quite low (less than 20%, and in many of the highly imbalanced categories it is 0), in a real disaster situation it isn't feasible to throw away messages that the model labels '0' for a given category. I think the system should be used as follows: For each message that comes in, it is put through the model, and for each category where the model labels it '1', the message is automatically sent to that team (such as the medical team or the search-and-rescue team). Some of these messages will not be relevant to the team to which they are sent, as the model's precision is less than 1 in most categories, but the fraction of irrelevant is generally not more than one-half. This should be helpful for addressing some situations quickly. However, since the recall is low for many categories, there also needs to be a team of humans that sifts through the remaining messages to determine which are relevant and which team to send it to; this will take longer than using the machine learning model, but the model simply misses too many important messages to use it alone.

A better situation would be one where the recall is above 90% for most categories, meaning that few relevant messages are missed; in that case one might not have to also rely on a separate team of humans to sift through the flood of mostly-irrelevant messages. In addition to high recall, I would want precision that is not too low. A lofty goal would be to have over 50% precision for each category in addition to high recall, though for classes that contain very few messages, this is perhaps unnecessary. For example, if only 1% of messages are in Class 1 in some category, then without the model a team of humans would have to read 99 irrelevant messages for every 1 relevant message. So if the model's precision were even 10%, this would mean the humans only have to read 9 irrelevant messages for every relevant one, reducing their workload by 10 times. In practice, the desired precision for a given category would likely be determined by how quickly that response team can read messages are directed to them by the model.

To achieve such high recall with good-enough precision, I would need to try a new kind of model. I did quite a bit of hyperparameter tuning on my random forest models and was not able to get reasonable recall for many of the categories. I could try using Word2Vec instead of CountVectorizer to get initial features for the model, and I could try more sophisticated models such as XGBoost or temporal CNNs.

## Licensing and acknowledgements <a name="acknowledgements"></a>

This code is freely available under a GPL-3.0 License. The raw data in 'messages.csv' and 'categories.csv' is provided by Figure Eight and Udacity and is subject to their rules for use.

I want to thank Udacity in particular for the help with the code that makes the web app run; I intend to customize the web app more later, but as it stands, almost all the code to create it is theirs. I also want to thank all the developers who created and maintain the open-source software that makes this work and so much like it possible.