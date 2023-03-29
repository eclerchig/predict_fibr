import flask
from flask import render_template
import pickle
import sklearn
from keras.models import Sequential
import xgboost
import numpy as np
import joblib
import tensorflow as tf
import pandas as pd

SCALERS = joblib.load('scaler.save')

variables_name = ['value_tsh',
                  'value_prl',
                  'value_lh',
                  'value_fsh',
                  'value_shbg',
                  'value_17hp',
                  'value_dheas',
                  'value_test2',
                  'value_test4',
                  'value_test7',
                  'value_triglycerides',
                  'value_total_cholesterol',
                  'value_hdl',
                  'value_ldl',
                  'value_vldl',
                  'value_glucose',
                  'value_test1_bioch',
                  'value_kd_ct',
                  'value_test2_bioch',
                  'value_test3_bioch',
                  'value_test4_bioch',
                  'value_test6_bioch',
                  'value_test8_bioch',
                  'value_test9_bioch',
                  'value_test10_bioch',
                  'value_test11_bioch',
                  'total_testosteron',
                  'estradiol_generated'
                  ]

app = flask.Flask(__name__, template_folder = 'templates')

@app.route('/', methods = ['POST', 'GET'])

@app.route('/index', methods = ['POST', 'GET'])
def main():
    if flask.request.method == 'GET':
        return render_template('main.html', output = False)
    if flask.request.method == 'POST':

        # ----------- код для XGBClassifier ---------
        #with open('xgb.pkl', 'rb') as f:
            #loaded_model = joblib.load(f)
        # -------------------------------------------

        # -------------- код для ANN ----------------
        loaded_model = tf.keras.models.load_model("ANN.keras")
        # -------------------------------------------

        df_X = pd.DataFrame()
        for idx, var in enumerate(variables_name):
            scaler = SCALERS[idx]
            input_val = np.array(float(flask.request.form[var])).reshape(-1, 1)
            scaled_value = scaler.transform(input_val)[0][0]
            temp_df = pd.DataFrame({f'{var}': [scaled_value]})
            df_X = pd.concat((df_X, temp_df), axis=1)

        #x_pred = np.array(x_pred).reshape(1,-1)

        # ----------- код для XGBClassifier ---------
        #y_prob = loaded_model.predict_proba(df_X)
        #class_ = np.argmax(y_prob, axis=1)
        # -------------------------------------------

        # -------------- код для ANN ----------------
        y_prob = loaded_model.predict(df_X)
        class_ = (y_prob > 0.5).astype(int)
        # -------------------------------------------

        return render_template('main.html', output = True,
                                            prob_class = y_prob,
                                            class_ = class_)

if __name__ == '__main__':
    app.run()

