# End-To-End-Medical-Chatbot

# How to run?

Clone the Repository

'''bash
Project repo :https://github.com/RajKumarSingh69/End-To-End-Medical-Chatbot.git

'''
# Step 01- Create a conda environment after opening the repository

'''bash
conda create -n mchatbot python=3.8 -y
'''
# Activate Environment
'''bash
conda activate mchatbot
'''

## Step 02- Install the Requirements
'''bash
pip install -r requirements.txt
'''
## Downlaod the quantize model form the link provided in model folder & keep the model in the model directory

## Downlaod the Llama2 Model:
    llama2-7b-chat.ggmlv3.q4_0.bin
# From the following link
    https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main


# Step 03- Run the Setup.py File
 ''' By running the setup.py fie all the pacakges that we build are ready to use in codes where we call it

# Step 04- Run Store_index.py (optional)
    ''' if you want to generate new vector embeddings for our pdf data and store these into your own ChromaDb vector database , unless this step is not maindatory to do'''

# Step 05- Run connection.py file from database_connection

    ''' Here we are using SQL database , in which we are storing all the information regarding Registration, User provided data on each disease page .Make sure you have latest version of mysql workbench installed in your system'''

    ''' Afetr installation , create database and follow below steps:-

        ''' for these you need to do some changes:-
            # Define the database connectivity information
            host = "localhost"
            u_name = 'root'
            password = 'YOUR PASSWORD'
            port_no = 3306 or your Port No
            db_name = 'YOUR DATABASE NAME'
        '''

        '''After these you need to create a table according to our fields , for this i provided all the code for table creation and these code are commented by me . Un-comment all the table creation code and run these , so that all the tables are created.'''
    '''

# Setp 06 - Final
    ''' Run app.py file for start our web-application'''
    '''open Terminal and write : python app.py '''

# Thanks
# If you want to Connect with me , open my Github Profile and connect with me on Linkedin




    


