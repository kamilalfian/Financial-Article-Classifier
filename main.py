# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
from gravityai import gravityai as grav
import pickle
import pandas as pd
model=pickle.load(open('financial_text_classifier.pkl','rb'))
tfidf_vectorizer=pickle.load(open('financial_text_vectorizer.pkl','rb'))
label_encoder=pickle.load(open('financial_text_encoder.pkl','rb'))
def process(inPath, outPath):
    input_df=pd.read_csv(inPath)
    features=tfidf_vectorizer.transform(input_df['body'])
    predictions=model.predict(features)
    input_df['category']=label_encoder.inverse_transform(predictions)
    output_df=input_df[['id','category']]
    output_df.to_csv(outPath,index=False)
grav.wait_for_requests(process)