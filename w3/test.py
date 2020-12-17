from dataPreprocessing.read_data import read_and_preprocess_database as read_and_preprocess_database

s = read_and_preprocess_database("adult")
print(len(s))