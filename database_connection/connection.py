import mysql.connector

# Defining the database connectivity information
host = "localhost"
u_name = 'root'
password = 'rajkr@803201' #Replace it with your password
port_no = 3306
db_name = 'HealthCare' #Replace database name with your desired table name

# Setting configuration
global mydb
mydb = mysql.connector.connect(
    host=host,
    user=u_name,
    passwd=password,
    port=port_no,
    database=db_name
)
print(mydb)

'''
def create_register_table():

    # Creating table for user registration
    mycursor = mydb.cursor()

    # Create table query
    tablename = "Register"   #Replace YourTableName with your desired table name
    query = f"""
    CREATE TABLE {tablename} (username VARCHAR(50),email VARCHAR(50),age INT,password VARCHAR(50));"""

    # Execute query
    mycursor.execute(f"USE {db_name}")
    mycursor.execute(query)

    # Commit the changes
    mydb.commit()

    # Close the cursor and connection
    mycursor.close()
    mydb.close()

    print(f"Table '{tablename}' created successfully in the database '{db_name}'.")

def create_cancer_table():
    # Creating table for your specified columns
    mycursor = mydb.cursor()

    # Create table query
    tablename = "cancerdata" #Replace YourTableName with your desired table name
    query = f"""
    CREATE TABLE IF NOT EXISTS {tablename} (Radious_mean FLOAT,texture_mean FLOAT,Smoothness_mean FLOAT,
    Compactness_mean FLOAT,symmetry_mean FLOAT,fractal_dimension_mean FLOAT,radius_se FLOAT,texture_se FLOAT,
    Smoothness_se FLOAT,Compactness_se FLOAT,concavity_se FLOAT,concave_points_se FLOAT,symmetry_se FLOAT,
    fractal_dimension_se FLOAT,smoothness_worst FLOAT,symmetry_worst FLOAT,fractal_dimension_worst FLOAT,
    result VARCHAR(10)
    );
"""

    # Execute query
    mycursor.execute(query)

    # Commit the changes
    mydb.commit()

    # Close the cursor and connection
    mycursor.close()
    mydb.close()

    print(f"Table '{tablename}' created successfully in the database '{db_name}'.")

def create_diabetes_table():
    # Creating table for your specified columns
    mycursor = mydb.cursor()

    # Create table query
    tablename = "diabetesdata"  # Replace YourTableName with your desired table name
    query = f"""CREATE TABLE IF NOT EXISTS {tablename} (username VARCHAR(50),o_age INT,email VARCHAR(50),
    Pregnancies FLOAT,Glucose FLOAT,BloodPressure FLOAT,SkinThickness FLOAT,Insulin FLOAT,BMI FLOAT,
    DiabetesPedigreeFunction FLOAT,Age FLOAT,result VARCHAR(50));"""

    # Execute query
    mycursor.execute(query)

    # Commit the changes
    mydb.commit()

    # Close the cursor and connection
    mycursor.close()
    mydb.close()

    print(f"Table '{tablename}' created successfully in the database '{db_name}'.")

def create_heart_table():
    # Creating table for your specified columns
    mycursor = mydb.cursor()

    # Create table query
    tablename = "heartdata"  # Replace YourTableName with your desired table name
    query = f"""CREATE TABLE IF NOT EXISTS {tablename} (username VARCHAR(50),o_age INT,email VARCHAR(50),
        Age FLOAT,Sex VARCHAR(10),chest_pain FLOAT,trestbps FLOAT,serum_cholestoral_in_mg_dl FLOAT,restecg FLOAT,thalach FLOAT,
        Exang FLOAT,oldpeak FLOAT,slope FLOAT,thal FLOAT,result VARCHAR(50));"""

    # Execute query
    mycursor.execute(query)

    # Commit the changes
    mydb.commit()

    # Close the cursor and connection
    mycursor.close()
    mydb.close()

    print(f"Table '{tablename}' created successfully in the database '{db_name}'.")

def create_kidney_table():
    # Creating table for your specified columns
    mycursor = mydb.cursor()

    # Create table query
    tablename = "kidneydata"  # Replace YourTableName with your desired table name
    query = f"""
    CREATE TABLE IF NOT EXISTS {tablename} (username VARCHAR(50),o_age INT,email VARCHAR(50),Age FLOAT,
        BP FLOAT,Specific_Gravity FLOAT,Albumin FLOAT,Sugar FLOAT,RBC FLOAT,Pus_cell FLOAT,PCC FLOAT,
        Bacteria FLOAT,BGR FLOAT,Blood_Urea FLOAT,Serum_Creatinine FLOAT,Sodium FLOAT,Potassium FLOAT,
        Haemoglobin FLOAT,PCV FLOAT,WBC FLOAT,RBCC FLOAT,Hypertension FLOAT,Diabetes_Mellitus FLOAT,CAD FLOAT,
        Appetite FLOAT,Peda_Edema FLOAT,Aanemia FLOAT,result VARCHAR(50));"""

    # Execute query
    mycursor.execute(query)

    # Commit the changes
    mydb.commit()

    # Close the cursor and connection
    mycursor.close()
    mydb.close()

    print(f"Table '{tablename}' created successfully in the database '{db_name}'.")

def create_liver_table():
    # Creating table for your specified columns
    mycursor = mydb.cursor()

    # Create table query
    tablename = "liverdata"  # Replace YourTableName with your desired table name
    query = f"""
    CREATE TABLE IF NOT EXISTS {tablename} (username VARCHAR(50),o_age INT,email VARCHAR(50),Age FLOAT,
        Gender VARCHAR(10),Total_Bilirubin FLOAT,Direct_Bilirubin FLOAT,Alkaline_Phosphotase FLOAT,Alamine_Aminotransferase FLOAT,
        Aspartate_Aminotransferase FLOAT,Total_Proteins FLOAT,Albumin FLOAT,Albumin_and_Globulin_Ratio FLOAT,Result VARCHAR(50));"""

    # Execute query
    mycursor.execute(query)

    # Commit the changes
    mydb.commit()

    # Close the cursor and connection
    mycursor.close()
    mydb.close()

    print(f"Table '{tablename}' created successfully in the database '{db_name}'.")

def create_malaria_table():
    # Creating table for the specified columns
    mycursor = mydb.cursor()

    # Create table query
    tablename = "malariaprediction"  # Replace YourTableName with your desired table name
    query = f"""
    CREATE TABLE IF NOT EXISTS {tablename} (username VARCHAR(50),email VARCHAR(50),age INT,picture LONGBLOB,result VARCHAR(50));"""

    # Execute query
    mycursor.execute(query)

    # Commit the changes
    mydb.commit()

    # Close the cursor and connection
    mycursor.close()
    mydb.close()

    print(f"Table '{tablename}' created successfully in the database '{db_name}'.")

def create_pneumonia_table():
    # Creating table for the specified columns
    mycursor = mydb.cursor()

    # Create table query
    tablename = "pneumoniadata"  # Replace YourTableName with your desired table name
    query = f"""
    CREATE TABLE IF NOT EXISTS {tablename} (username VARCHAR(50),email VARCHAR(50),age INT,picture LONGBLOB,result VARCHAR(50));"""

    # Execute query
    mycursor.execute(query)

    # Commit the changes
    mydb.commit()

    # Close the cursor and connection
    mycursor.close()
    mydb.close()

    print(f"Table '{tablename}' created successfully in the database '{db_name}'.")
'''

def register_user(username, age, email, password):
    table_name = 'register'

    # Prepare the SQL query
    query = f"INSERT INTO {table_name} (username, email, age, password) VALUES (%s, %s, %s, %s)"
    values = (username, email, age, password)

    # Execute the query
    mycursor = mydb.cursor()
    mycursor.execute(query, values)

    # Commit the changes
    mydb.commit()

    # Close the cursor and connection
    mycursor.close()
    mydb.close()

    print("User registered successfully")

def validate_login(email, password):
    global e_mail
    global pas
    e_mail=email
    pas=password
    mycursor = mydb.cursor()
    query = "SELECT * FROM register WHERE email = %s AND password = %s"
    mycursor.execute(query, (email, password))
    user = mycursor.fetchone()  # Fetch one matching row

    if user:  # If user exists
        return 1  # Login successful
    else:
        return 0  # User not found or incorrect password
def fetch_user_info():
    # Fetch user details using email and password
    mycursor = mydb.cursor()
    query = "SELECT username, age FROM register WHERE email = %s AND password = %s"
    mycursor.execute(query, (e_mail, pas))
    user = mycursor.fetchone()
    return user

def insert_cancer_data(Radious_mean, texture_mean, Smoothness_mean, Compactness_mean,
                       symmetry_mean, fractal_dimension_mean, radius_se, texture_se,
                       Smoothness_se, Compactness_se, concavity_se, concave_points_se,
                       symmetry_se, fractal_dimension_se, smoothness_worst, symmetry_worst,
                       fractal_dimension_worst, result):
    
    # Fetch user details using email and password
    user=fetch_user_info()
    
    if user:
        username, age = user  # Unpack the user details
        res = 'Yes' if result == 'True' else 'No'
        
        # Define the SQL query for inserting data into cancerdata table
        query = f"""
        INSERT INTO cancerdata(username, age, email, radius_mean, texture_mean, smoothness_mean, compactness_mean,
                       symmetry_mean, fractal_dimension_mean, radius_se, texture_se,
                       smoothness_se, compactness_se, concavity_se, concave_points_se,
                       symmetry_se, fractal_dimension_se, smoothness_worst, symmetry_worst,
                       fractal_dimension_worst, result) 
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        # Define the values to be inserted
        values = (username, age, e_mail, Radious_mean, texture_mean, Smoothness_mean, Compactness_mean,
                  symmetry_mean, fractal_dimension_mean, radius_se, texture_se,
                  Smoothness_se, Compactness_se, concavity_se, concave_points_se,
                  symmetry_se, fractal_dimension_se, smoothness_worst, symmetry_worst,
                  fractal_dimension_worst, res)
        
        # Execute the query
        mycursor.execute(query, values)
        
        # Commit the changes
        mydb.commit()
        
        # Close the cursor
        mycursor.close()
        
        print("Cancer Data Inserted successfully")
    else:
        print("Invalid login credentials. Cancer data not inserted.")

def insert_diabetes_data(Pregnancies,Glucose,BloodPressure,
        SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,result):

    # Fetch user details using email and password
    user=fetch_user_info()

    if user:
        res = 'Yes' if result == 'True' else 'No'
        # Unpack the user details like username,age
        username, age = user
        # now defining sql query
        query = f"""INSERT INTO diabetesdata(username,o_age,emial,Pregnancies,Glucose,BloodPressure,
            SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,result) 
            VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
    
        # Define the values to be inserted
        values = (username,age,e_mail,Pregnancies,Glucose,BloodPressure,
            SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,res)
        
        mycursor = mydb.cursor()
        # Execute the query
        mycursor.execute(query, values)
        
        # Commit the changes
        mydb.commit()
        
        # Close the cursor
        mycursor.close()
        
        print("Diabetes Data Inserted successfully")
    else:
        print("Invalid login credentials. Diabetes data not inserted.")

def insert_hert_data(Age,Sex,chest_pain,trestbps,
        serum_cholestoral_in_mg_dl,restecg,thalach,Exang,oldpeak,slope,thal,result):
    # Fetch user details using email and password
    user=fetch_user_info()

    if user:
        res = 'Yes' if result == 'True' else 'No'
        sex='Male' if Sex==1 else 'female'
        # Unpack the user details like username,age
        username, age = user
        # now defining sql query
        query = f"""INSERT INTO heartdata(username,o_age,email,Age,Sex,chest_pain,trestbps,
        serum_cholestoral_in_mg_dl,restecg,thalach,Exang,oldpeak,slope,thal,result)
        VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""

        # Define the values to be inserted
        values = (username,age,e_mail,Age,sex,chest_pain,trestbps,
        serum_cholestoral_in_mg_dl,restecg,thalach,Exang,oldpeak,slope,thal,result)

        mycursor = mydb.cursor()
        # Execute the query
        mycursor.execute(query, values)
        
        # Commit the changes
        mydb.commit()
        
        # Close the cursor
        mycursor.close()
        
        print("Hert Data Inserted successfully")
    else:
        print("Invalid login credentials. Hert data not inserted.")

def insert_kidney_data(Age,bp,sg,al,sugar,RBC,Pus_cell,pcc,
        Bacteria,bgr,bu,sc,sodium,potassium,hemo,pcv,wbc,rbcc,hp,dm,cad,ap,pe,aanemia,result):
    user=fetch_user_info()
    result=int(result)

    if user:
        res = 'Yes' if result == 'True' else 'No'
        # Unpack the user details like username,age
        username, age = user
        #defining sql query
        query = f"""INSERT INTO kidneydata (username,o_age,email,Age,BP,Specific_Gravity,Albumin,Sugar,RBC,Pus_cell,
            PCC,Bacteria,BGR,Blood_Urea,Serum_Creatinine,Sodium,Potassium,Haemoglobin,PCV,WBC,
            RBCC,Hypertension,Diabetes_Mellitus,CAD,Appetite,Peda_Edema,Aanemia,result)
            VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""

        # Define the values to be inserted
        values = (username,age,e_mail,Age,bp,sg,al,sugar,RBC,Pus_cell,pcc,
        Bacteria,bgr,bu,sc,sodium,potassium,hemo,pcv,wbc,rbcc,hp,dm,cad,ap,pe,aanemia,result)

        mycursor = mydb.cursor()
        # Execute the query
        mycursor.execute(query, values)
        
        # Commit the changes
        mydb.commit()
        
        # Close the cursor
        mycursor.close()
        
        print("Kidney Data Inserted successfully")
    else:
        print("Invalid login credentials. Kidney data not inserted.")

def insert_liver_data(Age, Gender, Total_Bilirubin, Direct_Bilirubin,
                      Alkaline_Phosphotase, Alamine_Aminotransferase, Aspartate_Aminotransferase,
                      Total_Proteins, Albumin, Albumin_and_Globulin_Ratio, result):

    # Convert numpy types to native Python types
    result = int(result)

    user = fetch_user_info()

    if user:
        res = 'Yes' if result == 'True' else 'No'
        # Unpack the user details like username,age
        username, age = user
        # defining sql query
        query = f"""INSERT INTO liverdata (username, o_age, email, Age, Gender, Total_Bilirubin,
        Direct_Bilirubin, Alkaline_Phosphotase, Alamine_Aminotransferase, Aspartate_Aminotransferase,
        Total_Proteins, Albumin, Albumin_and_Globulin_Ratio, Result)
        VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""

        # Define the values to be inserted
        values = (username, age, e_mail, Age, Gender, Total_Bilirubin, Direct_Bilirubin,
                  Alkaline_Phosphotase, Alamine_Aminotransferase, Aspartate_Aminotransferase, Total_Proteins,
                  Albumin, Albumin_and_Globulin_Ratio, result)

        # calling database object and making cursor
        mycursor = mydb.cursor()
        # Execute the query
        mycursor.execute(query, values)

        # Commit the changes
        mydb.commit()

        # Close the cursor
        mycursor.close()

        print("Liver Data Inserted successfully")
    else:
        print("Invalid login credentials. Liver data not inserted.")

def insert_malaria_data(img, result):
    # Fetching user data
    user = fetch_user_info()

    if user:
        res = 'Yes' if result == 1 else 'No'
        # Unpack the user details like username, age
        username, age = user
        # Defining sql query
        query = """INSERT INTO malariaprediction (username, email, age, picture, result) VALUES (%s, %s, %s, %s, %s)"""

        # Convert numpy array to bytes for storing in the database
        img_bytes = img.tobytes()

        # Define the values to be inserted
        values = (username, e_mail, age, img_bytes, res)

        # Calling database object and making cursor
        mycursor = mydb.cursor()
        # Execute the query
        mycursor.execute(query, values)

        # Commit the changes
        mydb.commit()

        # Close the cursor
        mycursor.close()

        print("Malaria Data Inserted successfully")
    else:
        print("Invalid login credentials. Malaria data not inserted.")



def insert_pneumonia_data(img,result):

    #fetching user data
    user=fetch_user_info()

    if user:
        res = 'Yes' if result == 1 else 'No'
        # Unpack the user details like username,age
        username, age = user
        #defining sql query
        query=f""" INSERT INTO pneumoniadata(username,email,age,picture,result) VALUES(%s,%s,%s,%s,%s)"""

        # Convert numpy array to bytes for storing in the database
        img_bytes = img.tobytes()

        # Define the values to be inserted
        values = (username, e_mail, age, img_bytes, res)

        #calling database object and making cursor
        mycursor = mydb.cursor()
        # Execute the query
        mycursor.execute(query, values)
        
        # Commit the changes
        mydb.commit()
        
        # Close the cursor
        mycursor.close()
        
        print("Pneumonia Data Inserted successfully")
    else:
        print("Invalid login credentials. Pneumonia data not inserted.")


# Here all the database connectivity codes are end 
# if you want to do modification , please feel free to do it
# if you want to collaborate with me then contact with me 
#via : email:- "rajkr8369@gmail.com" , linkedin :- "https://www.linkedin.com/in/raj-kumar-b81b4625a/"





    
    







    









