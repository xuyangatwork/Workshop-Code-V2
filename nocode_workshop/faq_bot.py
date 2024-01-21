import streamlit as st
import openai
from openai import OpenAI
import sqlite3
from basecode.authenticate import return_api_key
import configparser
import os
import pandas as pd

client = OpenAI(
	# defaults to os.environ.get("OPENAI_API_KEY")
	api_key=return_api_key(),
)

config = configparser.ConfigParser()
config.read('config.ini')

NEW_PLAN  = config['constants']['NEW_PLAN']
FEEDBACK_PLAN = config['constants']['FEEDBACK_PLAN']
PERSONAL_PROMPT = config['constants']['PERSONAL_PROMPT']
DEFAULT_TEXT = config['constants']['DEFAULT_TEXT']

# Create or check for the 'database' directory in the current working directory
cwd = os.getcwd()
WORKING_DIRECTORY = os.path.join(cwd, "database")

if not os.path.exists(WORKING_DIRECTORY):
	os.makedirs(WORKING_DIRECTORY)

if st.secrets["sql_ext_path"] == "None":
	WORKING_DATABASE= os.path.join(WORKING_DIRECTORY , st.secrets["default_db"])
else:
	WORKING_DATABASE= st.secrets["sql_ext_path"]

def clear_session_states():
	st.session_state.msg = []
	if "memory" not in st.session_state:
		pass
	else:
		del st.session_state["memory"]


def retrieve_faqs(chatbot_selection):
	conn = sqlite3.connect(WORKING_DATABASE)
	cursor = conn.cursor()
	cursor.execute("""
			SELECT ctr.chatbot_name, ctr.prompt, ctr.response 
			FROM Chatbot_Training_Records ctr 
			LEFT JOIN Users u ON ctr.user_id = u.user_id 
			WHERE ctr.chatbot_name = ?
		""", (chatbot_selection,))
	rows = cursor.fetchall()
	df = pd.DataFrame(rows, columns=['chatbot_name', 'prompt', 'response'])
	all_records_string = "\n".join([f"Prompt: {row['prompt']}, Response: {row['response']}" for index, row in df.iterrows()])
	conn.close()
	return all_records_string 

def faq_bot():
	conn = sqlite3.connect(WORKING_DATABASE)
	cursor = conn.cursor()

	st.write("Rules for the chatbot:")

	# Select the chatbot from rb_chatbot1 to rb_chatbot10
	chatbot_selection = st.selectbox("Select a Chatbot", [f"faq_chatbot{i}" for i in range(1, 16)])
	
	# Extract the data from the database
	# Extract the data from the database and join with Users table to get the username
	cursor.execute("""
		SELECT ctr.id, u.username, ctr.chatbot_name, ctr.prompt, ctr.response 
		FROM Chatbot_Training_Records ctr 
		LEFT JOIN Users u ON ctr.user_id = u.user_id 
		WHERE ctr.chatbot_name = ?
	""", (chatbot_selection,))
	rows = cursor.fetchall()
	df = pd.DataFrame(rows, columns=['id', 'user_id', 'chatbot_name', 'prompt', 'response'])
	with st.expander("View Data"):
		# Display and edit data
		st.write(df)

		# Adding new rules
		new_prompt = st.text_input("Enter new prompt")
		new_response = st.text_input("Enter new response")
		if st.button("Add Rule"):
			if new_prompt and new_response:
				cursor.execute("INSERT INTO Chatbot_Training_Records (chatbot_type, chatbot_name, prompt, response, user_id, school_id) VALUES (?, ?, ?, ?, ?, ?)",
							('faq_ai_bot', chatbot_selection, new_prompt, new_response, st.session_state.user['id'], st.session_state.user['school_id']))
				conn.commit()
				st.success("New rule added")
				conn.close()
				st.rerun()

		# Select a row ID to delete
		if not df.empty:
			delete_id = st.selectbox("Select a row ID to delete", df['id'])
			if st.button("Delete Row"):
				cursor.execute("DELETE FROM Chatbot_Training_Records WHERE id = ?", (delete_id,))
				conn.commit()
				st.success(f"Row with ID {delete_id} deleted successfully!")
				conn.close()
				st.rerun()
	
	if st.toggle("Clear Chat"):
		clear_session_states()

	st.divider()
	st.subheader("FAQ Chatbot with AI")
	basebot("FAQ Bot", chatbot_selection)
	

	# Concatenate all records into a string


#below ------------------------------ base bot , no memory ---------------------------------------------
#chat completion for streamlit function
def chat_completion(prompt, faq):
	faq = "This is a list of frequently answered questions:\n" + faq
	openai.api_key = return_api_key()
	os.environ["OPENAI_API_KEY"] = return_api_key()
	response = client.chat.completions.create(
		model=st.session_state.openai_model,
		messages=[
			{"role": "system", "content":faq},
			{"role": "user", "content": prompt},
		],
		temperature=st.session_state.temp, #settings option
		stream=True #settings option
	)
	return response

#integration API call into streamlit chat components
def basebot(bot_name, chatbot_selection):
	full_response = ""
	greetings_str = f"Hi, I am {bot_name}"
	help_str = "How can I help you today?"
	# Check if st.session_state.msg exists, and if not, initialize with greeting and help messages
	if 'msg' not in st.session_state:
		st.session_state.msg = [
			{"role": "assistant", "content": greetings_str},
			{"role": "assistant", "content": help_str}
		]
	elif st.session_state.msg == []:
		st.session_state.msg = [
			{"role": "assistant", "content": greetings_str},
			{"role": "assistant", "content": help_str}
		]
	for message in st.session_state.msg:
		with st.chat_message(message["role"]):
			st.markdown(message["content"])
	try:
		if prompt := st.chat_input("What is up?"):
			faq = retrieve_faqs(chatbot_selection)

			st.session_state.msg.append({"role": "user", "content": prompt})
			with st.chat_message("user"):
				st.markdown(prompt)

			with st.chat_message("assistant"):
				message_placeholder = st.empty()
				full_response = ""
				for response in chat_completion(prompt, faq):
					full_response += (response.choices[0].delta.content or "")
					message_placeholder.markdown(full_response + "▌")
				message_placeholder.markdown(full_response)
			
			st.session_state.msg.append({"role": "assistant", "content": full_response})
				
	except Exception as e:
		st.error(e)