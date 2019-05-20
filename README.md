# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

#### Important Files: 
1.  data/pre_process_data.py :  performs ETL pipeline and  tokenizes and cleans to text for modelling
2.  models/train_classifier.py :The ML pipeline that  models and trains the classifier and saves the model in a pickle file
named classifier.pkl
3. app/templates/*.html: HTML templates for the web app.
4. run.py: Starts the Python server for the web app and prepare visualizations.
5.  Need to be loaded:
		nltk.download('stopwords')
		nltk.download('punkt')
		nltk.download('wordnet')
		
4.  ETL pipeline, preprocess_data.py:  
	- load the datafiles
	- merges these two datafiles 
	- cleans the merged data
	- saves the cleaned data to sql databse
5. ML Pipeline, train_classifier.py:
	- loads the sql database
	- splits the data for training
	- builds a text processing and ML pipeline
	- trains and tunes the  model using GridSearch
	- Exports the trained model as pickle file
	