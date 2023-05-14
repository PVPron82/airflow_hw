# <YOUR_IMPORTS>
from datetime import datetime
import os
import json
import dill
import pandas as pd

path = os.environ.get('PROJECT_PATH', '..')


def predict():
    def load_model():
        files_list = os.listdir(f'{path}/data/models/')
        model_file_name = files_list[0]
        with open(f'{path}/data/models/{model_file_name}', 'rb') as file:
            return dill.load(file)

    model = load_model()

    def predict_test_date(model):
        test_file_list = os.listdir(f'{path}/data/test/')
        df_predict = pd.DataFrame(columns=['car_id', 'pred'])
        pred_dict = {}
        for test_file_name in test_file_list:
            with open(f'{path}/data/test/{test_file_name}') as file:
                data = json.load(file)
            car_id = data['id']
            df = pd.DataFrame.from_dict([data])
            y = model.predict(df)[0]
            pred_dict[car_id] = y
        df_predict = pd.DataFrame(list(pred_dict.items()), columns=['car_id', 'pred'])
        return df_predict

    df_pred = predict_test_date(model)

    def predict_to_file(df_pred):
        pred_filename = f'{path}/data/predictions/pred_{datetime.now().strftime("%Y%m%d%H%M")}.csv'
        df_pred.to_csv(pred_filename)



    predict_to_file(df_pred)




if __name__ == '__main__':
    predict()
