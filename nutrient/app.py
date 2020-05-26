from flask import Flask,request ,render_template
import pickle
import numpy as np
app =Flask(__name__)

model=pickle.load(open('finalModel.pkl','rb'))



@app.route('/')
def av():
    return render_template('login.html')
database={'abhishek':'1234','amit':'1234','am':'av','av':'am'}


@app.route('/form_login',methods=['POST','GET'])
def login():
    name1=request.form['username']
    psd=request.form['password']
    if name1 not in database:
        return render_template('login.html',info= f'You are not Register {name1} please Register First ')
    else:
        if database[name1]!=psd:
            return render_template('login.html',info=f'You have enter invalid password {psd} please enter correct password')
        else:
            return render_template('main.html',name=name1)


# this main file connect my page where i am get data

@app.route('/form_avmain',methods=['POST','GET'])
def predict():
    #print(request.form)
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    #print(int_features)
    #print(final)
    #prediction=model.predict(final)
    #print(prediction)

    #input=np.array([[5]])
    array=model.predict(final)
    #print(array)
    pickle.dump(model,open('finalModel.pkl','wb'))
    finalModel=pickle.load(open('finalModel.pkl','rb'))

    output='{0:.{1}f}'.format(array[0][1], 2)

    print(finalModel)
    #output='{0:.{11}f}'.format(prediction)
    #print(final)
    print(output)

    if output<str(6):
        return render_template('main.html',pred='Your crop is in Danger.\nProbability of crop occuring is {}'.format(output),bro="this side is under construction ")
    else:
        return render_template('main.html',pred='Your crop is safe.\n Probability of crop occuring is {}'.format(output),bro="this side is under construction ")


if __name__=='__main__':
    app.run()
