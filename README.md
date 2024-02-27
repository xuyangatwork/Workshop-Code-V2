## Generative AI Workshop Kit
Enabling teachers to experiment with LLM/ Generative AI in a Q&A chatbot

This kit will automatically:
 - Create a login page and all the features of a full stack application using ant_components 
 - Create an administrator account, password is pass1234 in a SQL Database


## Key Features:
 -  Upload documents and build a knowledge base using the OpenAI embeddings
 -  Enable semantic search on uploaded documents via [LanceDB](https://lancedb.com/)
 -  Preset custom prompt engineering to guide interactions for Q&A
 -  Latest features of Generative AI


## User-role specific features
> [!NOTE]  
> This app comes with the following user roles: admins, teachers, and students
 - **Admins** can reset passwords of students and teachers
 - **Teachers** can add and remove documents 
 - **Teachers** can build and remove knowledge base (VectorStores)
 - **Admins** can edit knowledge base and documents
 - **Students** can load their own knowledge base for their own chatbot

You can fork it at streamlit community cloud, it can be used straight away, just add the following to your streamlit secrets

> [!IMPORTANT]  
> The following env variables are required for setup. You can add this to the secrets.toml file in your streamlit deployment 
```

openai_key = "YOUR_OPEN_API_KEY"
cohere_key = "YOUR_OPEN_API_KEY - COHERE LLM API KEY"
google_key = "YOUR_OPEN_API_KEY - GEMINI PRO API KEY"
default_db = "chergpt.db"
default_temp = 0.0
default_frequency_penalty = 0.0
default_presence_penalty = 0.0
default_k_memory = 4
default_model = "gpt-4-1106-preview"
default_password = "default_password"
student_password = "studentp@sswrd"
teacher_password = "teacherp@sswrd"
super_admin_password = "pass1234"
super_admin = "super_admin"
default_title = "GenAI Workshop Framework V2"
sql_ext_path = "None"
NLTK_DATA ="./resources/nltk_data_dir/"
TWILIO_ACCOUNT_SID = "OPTIONAL YOU CAN REMOVE"
TWILIO_AUTH_TOKEN = "OPTIONAL YOU CAN REMOVE"
```

