
from flask import Flask, request, jsonify, flash,render_template,send_from_directory, redirect, url_for,session
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score 
from PIL import Image
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, render_template, jsonify, request
from src.helper import loading_embedding_model, CustomEmbeddingFunction
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from src.prompt import *
from database_connection.connection import register_user,validate_login,insert_cancer_data,insert_malaria_data,insert_pneumonia_data
from database_connection.connection import insert_diabetes_data,insert_hert_data,insert_kidney_data,insert_liver_data
import os
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_urlsafe(16)
# Import Logistic model and standard scaler pickle file
cancer_model = pickle.load(open('Model/cancer/cancer.pkl', 'rb'))
cancer_standard_scaler = pickle.load(open('Model/cancer/scaler.pkl', 'rb'))

#Importing RandomForest model and it standard scaler for Diabetes Prediction
Diabetes_model=pickle.load(open('Model\Diabetes\Diabetes.pkl','rb'))
Diabetes_Standard_scaler=pickle.load(open('Model/Diabetes/Diabetes_scaler.pkl','rb'))

#Importing RandomForest model and it standard scaler for Heart Prediction
Heart_model=pickle.load(open("Model\Heart\Heart.pkl",'rb'))
Heart_Standard_scaler=pickle.load(open("Model\Heart\Heart_scaler.pkl",'rb'))

#Importing LogisticRegression Model for Liver prediction
Liver_model=pickle.load(open("Model\liver\Liver.pkl",'rb'))
Liver_scaler=pickle.load(open("Model\liver\liver_scaler.pkl",'rb'))

##Importing RandomForest Model ans its Standard Scalar for kidney prediction
kidney_model=pickle.load(open("Model\kidney\kidney.pkl",'rb'))
kidney_scaler=pickle.load(open("Model\kidney\kidney_scaler.pkl",'rb'))

# Route for home page
@app.route('/')
def index():
    return render_template('login.html')

#Below this line all the code related to DataBase

#user regiser
@app.route('/register',methods=['GET','POST'])
def signup():
    if request.method == 'POST':
        username= str(request.form.get('username'))
        email= str(request.form.get('email'))
        password= str(request.form.get('password'))
        age= int(request.form.get('age'))

        #now storing these information inside our database
        register_user(username,age,email,password)
        
        return render_template('register.html')
    else:
        return render_template('register.html')
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        if validate_login(email, password) == 1:
            # Set session data
            session['logged_in'] = True
            session['email'] = email
            # Redirect to a dashboard or home page after successful login
            return render_template('home.html')
        else:
            # Show an error message and redirect to login page
            flash('User not found / Incorrect email or password')
            return render_template("login.html")
    else:
        return render_template('login.html')
@app.route('/logout')
def logout():
    # Clear session data
    session.pop('logged_in', None)
    session.pop('email', None)
    # Redirect to login page
    return render_template('login.html')

#Protect Route that require Authenticaton
# Ensure that only logged-in users can access certain routes.
from functools import wraps

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/home')
@login_required
def home():
    return render_template('home.html')

@app.route("/about")
@login_required
def about():
    return render_template("about.html")

@app.route('/cancer', methods=['GET', 'POST'])
@login_required
def predict_datapoint():
    if request.method == 'POST':
        # Extract data from form with standardized feature names
        radius_mean = float(request.form.get('Radius_mean'))
        texture_mean = float(request.form.get('texture_mean'))
        smoothness_mean = float(request.form.get('Smoothness_mean'))
        compactness_mean = float(request.form.get('Compactness_mean'))
        symmetry_mean = float(request.form.get('symmetry_mean'))
        fractal_dimension_mean = float(request.form.get('fractal_dimension_mean'))
        radius_se = float(request.form.get('radius_se'))
        texture_se = float(request.form.get('texture_se'))
        smoothness_se = float(request.form.get('smoothness_se'))
        compactness_se = float(request.form.get('compactness_se'))
        concavity_se = float(request.form.get('concavity_se'))
        concave_points_se = float(request.form.get('concave points_se'))
        symmetry_se = float(request.form.get('symmetry_se'))
        fractal_dimension_se = float(request.form.get('fractal_dimension_se'))
        smoothness_worst = float(request.form.get('smoothness_worst'))
        symmetry_worst = float(request.form.get('symmetry_worst'))
        fractal_dimension_worst = float(request.form.get('fractal_dimension_worst'))

        # Create a DataFrame for scaling and prediction with correct feature names
        input_data = pd.DataFrame([{
            'radius_mean': radius_mean,
            'texture_mean': texture_mean,
            'smoothness_mean': smoothness_mean,
            'compactness_mean': compactness_mean,
            'symmetry_mean': symmetry_mean,
            'fractal_dimension_mean': fractal_dimension_mean,
            'radius_se': radius_se,
            'texture_se': texture_se,
            'smoothness_se': smoothness_se,
            'compactness_se': compactness_se,
            'concavity_se': concavity_se,
            'concave points_se': concave_points_se,
            'symmetry_se': symmetry_se,
            'fractal_dimension_se': fractal_dimension_se,
            'smoothness_worst': smoothness_worst,
            'symmetry_worst': symmetry_worst,
            'fractal_dimension_worst': fractal_dimension_worst
        }])

        # Scale the input data
        new_data_scaled = cancer_standard_scaler.transform(input_data)

        # Predict the result
        result = cancer_model.predict(new_data_scaled)
        result_1 = "Sorry, You are Suffering with Cancer" if result[0] == 'True' else "Congrats! You are Healthy"

        # now here we are storing all the infromation inside our database which is given my user with it's result
        insert_cancer_data(radius_mean, texture_mean, smoothness_mean, compactness_mean,
        symmetry_mean, fractal_dimension_mean, radius_se, texture_se,
        smoothness_se, compactness_se, concavity_se, concave_points_se,
        symmetry_se, fractal_dimension_se, smoothness_worst, symmetry_worst,
        fractal_dimension_worst,result[0])

        return render_template('cancer.html', result=result_1)


    else:
        return render_template('cancer.html')


# Diabetes prediction related code for fetching the information form the form and 
# after that pass it inside the model and make predicton

@app.route('/diabetes', methods=['GET', 'POST'])
@login_required
def diabetes_prediction():
    if request.method == 'POST':
        Pregnancies = float(request.form.get('Pregnancies'))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))

        new_data_scaled = Diabetes_Standard_scaler.transform([[Pregnancies,Glucose,BloodPressure,
        SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])

        result = Diabetes_model.predict(new_data_scaled)

        if result[0] == 1:
            result1 = "Sorry, You are Suffering with Diabetes"
        else:
            result1 = "Congrats! You are Healthy"
        
        insert_diabetes_data(Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,result[0])

        return render_template('diabetes.html', result=result1)

    else:
        return render_template('diabetes.html')

# Heart Prediction code

@app.route('/heart', methods=['GET', 'POST'])
@login_required
def heart_prediction():
    if request.method == 'POST':
        Age= float(request.form.get('age'))
        Sex= float(request.form.get('sex'))
        
        chest_pain = request.form.get('chest pain')
        if chest_pain is not None:
            chest_pain = float(chest_pain.strip())
        else:
    # Handle the case when 'chest pain' is missing or empty
            chest_pain = 0.0  # or any other default value


        trestbps = float(request.form.get('trestbps'))
        serum_cholestoral_in_mg_dl = float(request.form.get('serum cholestoral in mg/dl'))
        restecg = float(request.form.get('restecg'))
        thalach = float(request.form.get('thalach'))
        Exang = float(request.form.get('exang'))
        oldpeak = float(request.form.get('oldpeak'))
        slope = float(request.form.get('slope'))
        thal = float(request.form.get('thal'))


        new_data_scaled = Heart_Standard_scaler.transform([[Age,Sex,chest_pain,trestbps,
        serum_cholestoral_in_mg_dl,restecg,thalach,Exang,oldpeak,slope,thal]])

        result= Heart_model.predict(new_data_scaled)

        if result[0] == 1:
            result = "Sorry, You are Suffering with Diabetes"
        else:
            result = "Congrats! You are Healthy"
        
        insert_hert_data(Age,Sex,chest_pain,trestbps,
        serum_cholestoral_in_mg_dl,restecg,thalach,Exang,oldpeak,slope,thal,result[0])
        
        return render_template('heart.html', result=result)

    else:
        return render_template('heart.html')

# Liver Prediction code
@app.route('/liver', methods=['GET', 'POST'])
@login_required
def liver_prediction():
    if request.method == 'POST':
        Age = float(request.form.get('Age'))
        Gender = float(request.form.get('Gender'))
        Total_Bilirubin = float(request.form.get('Total_Bilirubin'))
        Direct_Bilirubin = float(request.form.get('Direct_Bilirubin'))
        Alkaline_Phosphotase = float(request.form.get('Alkaline_Phosphotase'))
        Alamine_Aminotransferase = float(request.form.get('Alamine_Aminotransferase'))
        Aspartate_Aminotransferase = float(request.form.get('Aspartate_Aminotransferase'))
        Total_Protiens = float(request.form.get('Total_Protiens'))
        Albumin = float(request.form.get('Albumin'))
        Albumin_and_Globulin_Ratio = float(request.form.get('Albumin_and_Globulin_Ratio'))

        input_data = pd.DataFrame([{
            'Age': Age,
            'Gender': Gender,
            'Total_Bilirubin': Total_Bilirubin,
            'Direct_Bilirubin': Direct_Bilirubin,
            'Alkaline_Phosphotase': Alkaline_Phosphotase,
            'Alamine_Aminotransferase': Alamine_Aminotransferase,
            'Aspartate_Aminotransferase': Aspartate_Aminotransferase,
            'Total_Protiens': Total_Protiens,
            'Albumin': Albumin,
            'Albumin_and_Globulin_Ratio': Albumin_and_Globulin_Ratio
        }])

        new_data_scaled = Liver_scaler.transform(input_data)

        result = Liver_model.predict(new_data_scaled)

        if result[0] == 1:
            result1 = "Sorry, You are Suffering with Diabetes"
        else:
            result1 = "Congrats! You are Healthy"
        insert_liver_data(Age,Gender,Total_Bilirubin,Direct_Bilirubin,
        Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Protiens,
        Albumin,Albumin_and_Globulin_Ratio,result[0])
        return render_template('liver.html', result=result1)

    else:
        return render_template('liver.html')
    
@app.route('/kidney', methods=['GET', 'POST'])
@login_required
def kidney_prediction():
    if request.method == 'POST':
        Age= float(request.form.get('age'))
        bp= float(request.form.get('bp'))
        sg= float(request.form.get('sg'))
        al= float(request.form.get('al'))
        sugar= float(request.form.get('sugar'))
        RBC= float(request.form.get('RBC'))
        Pus_cell= float(request.form.get('Pus_cell'))
        pcc= float(request.form.get('pcc'))
        Bacteria= float(request.form.get('Bacteria'))
        bgr= float(request.form.get('bgr'))
        bu= float(request.form.get('bu'))
        sc= float(request.form.get('sc'))
        sodium= float(request.form.get('sodium'))
        potassium= float(request.form.get('potassium'))
        hemo= float(request.form.get('Haemoglobin'))
        pcv= float(request.form.get('pcv'))
        wbc= float(request.form.get('wbc'))
        rbcc= float(request.form.get('rbcc'))
        hp= float(request.form.get('hp'))
        dm= float(request.form.get('dm'))
        cad= float(request.form.get('cad'))
        ap= float(request.form.get('ap'))
        pe= float(request.form.get('pe'))
        aanemia= float(request.form.get('aanemia'))


        new_data_scaled = kidney_scaler.transform([[Age,bp,sg,al,sugar,RBC,Pus_cell,pcc,
        Bacteria,bgr,bu,sc,sodium,potassium,hemo,pcv,wbc,rbcc,hp,dm,cad,ap,pe,aanemia]])

        result= kidney_model.predict(new_data_scaled)

        if result[0] == 1:
            result1 = "Congrats! Your are Healthy"
        else:
            result1 = "Sorry, You are Suffering with Diabetes"
        
        insert_kidney_data(Age,bp,sg,al,sugar,RBC,Pus_cell,pcc,
        Bacteria,bgr,bu,sc,sodium,potassium,hemo,pcv,wbc,rbcc,hp,dm,cad,ap,pe,aanemia,result[0])
        return render_template('kidney.html', result=result1)

    else:
        return render_template('kidney.html')

# Below codes related to maleria prediction

# Create the uploads directory if it does not exist
uploads_dir = os.path.join(app.root_path, 'static/uploads')
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)

# Making Route for Malaria prediction page 
@app.route("/malaria", methods=['POST', 'GET'])
@login_required
def upload_file():
    if request.method == 'POST':
        img = request.files['image']
        uploaded_filename = img.filename
        img.save(os.path.join(uploads_dir, uploaded_filename))
        img_path = os.path.join(uploads_dir, uploaded_filename)

        # Load and resize the image
        img = Image.open(img_path)
        img = img.resize((224, 224))  # Resize to the expected input size

        img = tf.keras.utils.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        model = tf.keras.models.load_model("Model/model_vgg19.h5")
        
        # Predict the class
        pred = np.argmax(model.predict(img), axis=1)[0]
        message = "Malaria detected" if pred == 1 else "No malaria detected"
        label = "NEGATIVE" if pred == 1 else "POSITIVE"

        insert_malaria_data(img,pred)
        
        return render_template('malaria_predict.html', message=message, pred=pred, label=label, image_file_name=uploaded_filename)  # Example accuracy, replace with actual if available
    
    return render_template('malaria.html')


@app.route("/pneumonia", methods = ['POST', 'GET'])
@login_required
def pneumoniapredictPage():
    if request.method == 'POST':
        img = request.files['image']
        uploaded_filename = img.filename
        img.save(os.path.join(uploads_dir, uploaded_filename))
        img_path = os.path.join(uploads_dir, uploaded_filename)

        #load and resize the image
        img = tf.keras.utils.load_img(img_path, target_size=(128, 128))
        img = tf.keras.utils.img_to_array(img)
        img = np.expand_dims(img, axis=0)

        model = tf.keras.models.load_model("Model/pneumonia.h5")
        pred = np.argmax(model.predict(img))

        message = "Pneumonia detected" if pred == 1 else "No Pneumonia detected"
        label = "POSITIVE" if pred == 1 else "NEGATIVE"

        insert_pneumonia_data(img,pred)
        
        return render_template('pneumonia_predict.html', message=message,pred=pred,label=label, image_file_name=uploaded_filename)
    return render_template('pneumonia.html',)


#From Below here all codes are related to medical chatbot  


# Load the embedding model
model = loading_embedding_model()

# Ensure the model is correctly loaded before proceeding
if not model:
    raise RuntimeError("Failed to load the embedding model")

# Create the embedding function
embedding_function = CustomEmbeddingFunction({}, model)

# Load the saved Chroma vector store instance with the embedding function
persist_directory = 'ChromaDB'
db = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

llm = CTransformers(
    model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    config={'max_new_tokens': 512, 'temperature': 0.8}
)

# Now we need to create our question answering object
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

@app.route("/chat")
@login_required
def chat_index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
@login_required
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result = qa.invoke({"query": input})  # Use invoke instead of __call__
    print("Response: ", result["result"])
    return str(result["result"])

if __name__ == "__main__":
    app.run(debug=False) 

#The End 
#If you want to do some modification or want to build application like this 
# contact me : rajkr8369@gmail.com

