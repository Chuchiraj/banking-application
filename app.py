from flask import Flask, render_template, request
import pickle
import sklearn
import pandas as pd
model = pickle.load(open('banking.pkl','rb'))

app = Flask(__name__)

@app.route('/', methods = ['GET','POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        form_inputs = pd.DataFrame(request.form.to_dict(), index=[0])
        prediction = model.predict(form_inputs.astype('float'))
        if prediction == 1:
            return "Person will likely to default"
        else:
            return "Person is good to go for loan"


if __name__ == "__main__":
    app.run(debug=True)