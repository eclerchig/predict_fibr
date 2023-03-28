import flask
from flask import render_template
import pickle
import sklearn
from keras.models import Sequential
import xgboost
import numpy as np
import joblib
import tensorflow as tf

SCALER = joblib.load('scaler.save')

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
        #Название модели
        with open('xgb.pkl', 'rb') as f:
            #loaded_model = joblib.load(f)
            loaded_model = tf.keras.models.load_model("ANN.keras")
        x_pred = SCALER.transform(np.array([float(flask.request.form[var]) for var in variables_name]).reshape(-1,1))
        #x_pred = np.array(x_pred).reshape(1,-1)
        y_prob = loaded_model.predict(x_pred.reshape(1,-1))
        #y_pred = loaded_model.predict(x_pred)
        #predict_class= np.argmax(y_pred, axis=1)
        class_ = (y_prob < 0.5).astype(int)

        return render_template('main.html', output = True,
                                            prob_class = y_prob,
                                            class_ = class_)

if __name__ == '__main__':
    app.run()

