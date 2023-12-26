import base64  # encoding of graph in binary format
import io #buffer
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
import joblib #to create or convert files doesn't need to open or close files
import pandas as pd
import numpy as np

app = Flask(__name__, template_folder="templete") #to run on only this page
le0 = joblib.load('le0.pkl') # to load file
model = joblib.load('model.pkl')


@app.route('/', methods=['GET', 'POST']) #url
def index():
    if request.method == "POST":
        input1 = float(request.form.get('number1'))
        input2 = float(request.form.get('number2'))
        input3 = float(request.form.get('number3'))
        input4 = request.form.get('number4')
        if input4 == 'Select Influencer':
            return render_template('index.html', a="Select correct Influencer!")
        else:
            columns = ['TV', 'Radio', 'Social Media']
            amounts = [input1, input2, input3]

            # amounts = [500,230,300]
            total_amount = sum(amounts)
            print(total_amount)
            a = [input1, input2, input3, input4, total_amount]
            array1 = np.array(a).reshape(1, -1)
            data = pd.DataFrame(array1, columns=['A', 'B', 'C', 'D', 'E'])

            data['A'] = data['A'].astype(float)
            data['B'] = data['B'].astype(float)
            data['C'] = data['C'].astype(float)
            data['E'] = data['E'].astype(float)
            print(data)
            array = data.iloc[:, [0, 1, 2, 3, 4]].values
            # #Generating Bar Chart
            plt.bar(columns,amounts)
            plt.xlabel("Platforms")
            plt.ylabel("Amount spent")
            plt.title("Amount spent on diff. platforms")
            #
            # #Saving it to buffer
            buffer = io.BytesIO()
            plt.savefig(buffer,format='png')
            buffer.seek(0)

            chart = base64.b64encode(buffer.getvalue()).decode('utf-8') #to decode
            # print(chart)
            # array  = np.array(a)
            print(array)

            array = array.reshape(1, -1)
            print(array)

            array[:, 3] = le0.transform(array[:, 3])
            print(array)
            # print(type(array[0][0]))
            # print(type(array[0][1]))
            # print(type(array[0][2]))
            # print(type(array[0][3]))
            # print(type(array[0][4]))
            prediction = model.predict(array)
            print(prediction[0])
            return render_template('index.html', a=prediction[0][0], plot=chart)
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
