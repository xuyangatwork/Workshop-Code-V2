import streamlit as st
import openai
from openai import OpenAI
import sqlite3
from basecode.authenticate import return_api_key
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI
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

def extract_and_combine_responses():
	# Connect to the SQLite database
	conn = sqlite3.connect(WORKING_DATABASE)
	cursor = conn.cursor()

	# SQL query to select all responses for discussion_bot
	query = "SELECT response FROM Chatbot_Training_Records WHERE chatbot_type = 'discussion_bot'"
	
	try:
		cursor.execute(query)
		responses = cursor.fetchall()
		
		# Combine all responses into a single string
		combined_responses = ' '.join([response[0] for response in responses if response[0]])

		return combined_responses
	except sqlite3.Error as e:
		st.write(f"An error occurred: {e}")
		return ""
	finally:
		# Close the database connection
		conn.close()

#below ------------------------------ base bot , summary memory for long conversation---------------------------------------------
#summary of conversation , requires another LLM call for every input, useful for feedback and summarising what was spoken
def memory_summary_component(prompt, prompt_design): #currently not in use
	conn = sqlite3.connect(WORKING_DATABASE)
	cursor = conn.cursor()

	if "memory" not in st.session_state:
		llm = ChatOpenAI(model_name=st.session_state.openai_model,temperature=st.session_state.temp)
		st.session_state.memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=2000)
	messages = st.session_state["memory"].chat_memory.messages
	previous_summary = ""
	mem = st.session_state["memory"].predict_new_summary(messages, previous_summary)
	#vectorstore available
	if st.session_state.vs:
		docs = st.session_state.vs.similarity_search(prompt)
		resource = docs[0].page_content
		source = docs[0].metadata
		prompt_template = prompt_design + f"""
						Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. 
						Search Result:
						{resource}
						{source}
						History of conversation:
						{mem}
						"""
			
	#vectorstore not available
	else:
		prompt_template = prompt_design + f"""
						Summary of current conversation:
						{mem}"""
	chatbot_name = "chatbot" + str(st.session_state.user['id'])
	user_id = st.session_state.user['id']
	school_id = st.session_state.user['school_id']

	# Check if a record exists
	cursor.execute("SELECT COUNT(*) FROM Chatbot_Training_Records WHERE user_id = ? AND school_id = ?",
				   (user_id, school_id))
	record_exists = cursor.fetchone()[0] > 0

	if record_exists:
		# Update the existing record
		cursor.execute("UPDATE Chatbot_Training_Records SET response = ? WHERE user_id = ? AND school_id = ?",
					   (mem, user_id, school_id))
	else:
		# Insert a new record
		cursor.execute("INSERT INTO Chatbot_Training_Records (chatbot_type, chatbot_name, prompt, response, user_id, school_id) VALUES (?, ?, ?, ?, ?, ?)",
					   ('discussion_bot', chatbot_name, "NIL", mem, user_id, school_id))
	conn.commit()
	conn.close()
	return prompt_template


#chat completion memory for streamlit using memory buffer
def chat_completion_qa_memory(prompt, prompt_design):
	openai.api_key = return_api_key()
	os.environ["OPENAI_API_KEY"] = return_api_key()
	prompt_template = memory_summary_component(prompt, prompt_design)
	response = client.chat.completions.create(
		model=st.session_state.openai_model,
		messages=[
			{"role": "system", "content":prompt_template },
			{"role": "user", "content": prompt},
		],
		temperature=st.session_state.temp, #settings option
		presence_penalty=st.session_state.presence_penalty, #settings option
		frequency_penalty=st.session_state.frequency_penalty, #settings option
		stream=True #settings option
	)
	return response

#integration API call into streamlit chat components with memory and qa

def discussion_bot(bot_name, prompt_design):
	if st.button("Clear Chat"):
		clear_session_states()
	full_response = ""
	greetings_str = st.session_state.discussion_greetings
	#st.write(greetings_str)
	# Check if st.session_state.msg exists, and if not, initialize with greeting and help messages
	if 'msg' not in st.session_state:
		st.session_state.msg = [
			
			{"role": "assistant", "content": greetings_str}
		]
	elif st.session_state.msg == []:
		st.session_state.msg = [
			
			{"role": "assistant", "content": greetings_str}
		]
	#lesson collaborator
	for message in st.session_state.msg:
		with st.chat_message(message["role"]):
			st.markdown(message["content"])

	try:
		if prompt := st.chat_input("Enter your thoughts"):
			st.session_state.msg.append({"role": "user", "content": prompt})
			with st.chat_message("user"):
				st.markdown(prompt)

			with st.chat_message("assistant"):
				message_placeholder = st.empty()
				full_response = ""
				for response in chat_completion_qa_memory(prompt, prompt_design):
					full_response += (response.choices[0].delta.content or "")
					message_placeholder.markdown(full_response + "▌")
				message_placeholder.markdown(full_response)
				#Response Rating
				
			st.session_state.msg.append({"role": "assistant", "content": full_response})
			st.session_state["memory"].save_context({"input": prompt},{"output": full_response})
			
			
			
	except Exception as e:
		st.exception(e)

