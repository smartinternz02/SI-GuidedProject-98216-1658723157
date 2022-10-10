from flask import Flask,render_template,request
# Flask-It is our framework which we are going to use to run/serve our application.
#request-for accessing file which was uploaded by the user on our application.
import os
import numpy as np #used for numerical analysis
from tensorflow.keras.models import load_model#to load our trained model
from tensorflow.keras.preprocessing import image



app = Flask(__name__,template_folder="templates") # initializing a flask app
# Loading the model
model=load_model('rock.h5')
print("Loaded model from disk")


@app.route('/')# route to display the home page
def home():
    return render_template('home.html')#rendering the home page


@app.route('/intro') # routes to the intro page
def intro():
    return render_template('intro.html')#rendering the intro page

@app.route('/image1',methods=['GET','POST'])# routes to the index html
def image1():
    return render_template("image.html")



@app.route('/predict',methods=['GET', 'POST'])# route to show the predictions in a web UI
def launch():
    if request.method=='POST':
        f=request.files['file'] #requesting the file
        basepath=os.path.dirname('__file__')#storing the file directory
        filepath=os.path.join(basepath,"uploads",f.filename)#storing the file in uploads folder
        f.save(filepath)#saving the file
        
        img=image.load_img(filepath,target_size=(64,64)) #load and reshaping the image
        x=image.img_to_array(img)#converting image to array
        x=np.expand_dims(x,axis=0)#changing the dimensions of the image
        
        pred=model.predict(x)
        pred=np.argmax(pred,axis=1)#predicting classes
        print("prediction",pred)#printing the prediction
        index=['Blue Calcite', 'Limestone', 'Marble', 'Olivine', 'Red Crystal']
        result=str(index[pred[0]])
        
        #result = str(index[pred[0]])
        
        if (result=='Blue Calcite'):
            return render_template("0.html",showcase =  str(result))
        elif (result=='Limestone'):
            return render_template("1.html",showcase =  str(result))
        elif (result=='Marble'):
            return render_template("2.html",showcase =  str(result))
        elif (result=='Olivine'):
            return render_template("3.html",showcase =  str(result))
        else:
            return render_template("4.html",showcase =  str(result))
        #return result#resturing the result
    else:
        return None
     
if __name__ == "__main__":
   # running the app
    app.run(debug=False,port=8000)
