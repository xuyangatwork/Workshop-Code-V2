import openai
from openai import OpenAI
import streamlit as st
from basecode.authenticate import return_api_key
import google.generativeai as genai
import os
import pandas as pd
import sqlite3
import string
import cohere


client = OpenAI(
	# defaults to os.environ.get("OPENAI_API_KEY")
	api_key=return_api_key(),
)

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
	st.session_state.messages = []
	if "memory" not in st.session_state:
		pass
	else:
		del st.session_state["memory"]


def call_api():
	st.subheader("Calling the LLM API")
	prompt_design = st.text_input("Enter your the prompt design for the API call:", value="You are a helpful assistant.")
	prompt_query = st.text_input("Enter your user input:", value="Tell me about Singapore in the 1970s in 50 words.")
	select_model = st.selectbox("Select a model", ["gpt-3.5-turbo", "gpt-4-1106-preview", "cohere", "gemini-pro"])	
	if st.button("Call the API"):
		if prompt_design and prompt_query:
			if select_model == "cohere":
				call_cohere_api(prompt_design, prompt_query)
			elif select_model == "gemini-pro":
				call_google_api(prompt_design, prompt_query)
			else:
				api_call(prompt_design, prompt_query, select_model)
		else:
			st.warning("Please enter a prompt design and user input.")


def call_google_api(prompt_design, prompt_query):
	# Initialize the Cohere client
	genai.configure(api_key = st.secrets["google_key"])

	with st.status("Calling the Google API..."):
		# Call the Cohere API
		
		chat_model = genai.GenerativeModel('gemini-pro')
		response = chat_model.generate_content(prompt_design + prompt_query)
		# Check if the response has the expected structure
		
		st.write(response.text)




def call_cohere_api(prompt_design, prompt_query):
	# Initialize the Cohere client
	co = cohere.Client(st.secrets["cohere_key"])

	with st.status("Calling the Cohere API..."):
		# Call the Cohere API
		response = co.generate(prompt=prompt_design + "\n" + prompt_query, max_tokens=1000)
		
		# Check if the response has the expected structure
		if response and response.generations:
			# Extract the text of the first generation
			generation_text = response.generations[0].text

			# Display the raw response (optional)
			st.markdown("**This is the raw response:**")
			st.write(response)

			# Display the extracted response
			st.markdown("**This is the extracted response:**")
			st.write(generation_text)

			# Display token usage information
			# Display token usage information
			if 'meta' in response and 'billed_units' in response['meta']:
				completion_tokens = response['meta']['billed_units']['output_tokens']
				prompt_tokens = response['meta']['billed_units']['input_tokens']
				st.write(f"Completion Tokens: {completion_tokens}")
				st.write(f"Prompt Tokens: {prompt_tokens}")
		else:
			st.error("No response or unexpected response format received from the API.")

def api_call(p_design, p_query, model):
	openai.api_key = return_api_key()
	os.environ["OPENAI_API_KEY"] = return_api_key()
	st.title("Api Call")
	#MODEL = "gpt-3.5-turbo"
	with st.status("Calling the OpenAI API..."):
		response = client.chat.completions.create(
			model=model,
			messages=[
				{"role": "system", "content": p_design},
				{"role": "user", "content": p_query},
			],
			temperature=0,
		)

		st.markdown("**This is the raw response:**") 
		st.write(response)
		st.markdown("**This is the extracted response:**")
		st.write(response.choices[0].message.content)
		completion_tokens = response.usage.completion_tokens
		prompt_tokens = response.usage.prompt_tokens
		total_tokens = response.usage.total_tokens

		st.write(f"Completion Tokens: {completion_tokens}")
		st.write(f"Prompt Tokens: {prompt_tokens}")
		st.write(f"Total Tokens: {total_tokens}")

def rule_based():
	st.write("Rules for the chatbot:")
	df = pd.DataFrame(
		[
			{"prompt": "Hello", "response": "Hi there what can I do for you"},
			{
				"prompt": "What is your name?",
				"response": "My name is EAI , an electronic artificial being"
			},
			{"prompt": "How old are you?", "response": "Today is my birthday!"},
		]
	)

	edited_df = st.data_editor(df, num_rows="dynamic")
	st.divider()
	# Initialize chat history
	if "messages" not in st.session_state:
		st.session_state.messages = []

	# Display chat messages from history on app rerun
	for message in st.session_state.messages:
		with st.chat_message(message["role"]):
			st.markdown(message["content"])

	# React to user input
	if prompt := st.chat_input("Enter your prompt"):
		if prompt in edited_df["prompt"].values:
			reply = edited_df.loc[edited_df["prompt"] == prompt]["response"].values[0]
		else:
			reply = "I don't understand"

		with st.chat_message("user"):
			st.write(prompt)
			st.session_state.messages.append({"role": "user", "content": prompt})
		with st.chat_message("assistant"):
			st.write(reply)
			st.session_state.messages.append({"role": "assistant", "content": reply})

def init_training_data():
	# Base data for initialization
	initial_data = [
		{"prompt": "Hello", "response": "Hi there what can I do for you"},
		{"prompt": "What is your name?", "response": "My name is EAI, an electronic artificial being"},
		{"prompt": "How old are you?", "response": "Today is my birthday!"}
	]

	# Creating a list of 10 DataFrames for each chatbot
	global_dfs = []
	for i in range(1, 16):
		chatbot_name = f"rb_chatbot{i}"
		df = pd.DataFrame(initial_data)
		df['chatbot_type'] = 'rule_base'
		df['chatbot_name'] = chatbot_name
		global_dfs.append(df)

	with sqlite3.connect(WORKING_DATABASE) as conn:
		cursor = conn.cursor()

		# Delete existing data
		cursor.execute('DELETE FROM Chatbot_Training_Records')

		# Insert data into Chatbot_Training_Records
		for df in global_dfs:
			for _, row in df.iterrows():
				cursor.execute('''
					INSERT INTO Chatbot_Training_Records (chatbot_type, chatbot_name, prompt, response, user_id, school_id) 
					VALUES (?, ?, ?, ?, ?, ?)
				''', (row['chatbot_type'], row['chatbot_name'], row['prompt'], row['response'], 0, 0))

		conn.commit()

def clean_string(input_str):
	return input_str.strip(string.punctuation + string.whitespace).lower()

def group_rule_based():
	# Database connection
	conn = sqlite3.connect(WORKING_DATABASE)
	cursor = conn.cursor()

	st.write("Rules for the chatbot:")

	# Select the chatbot from rb_chatbot1 to rb_chatbot10
	chatbot_selection = st.selectbox("Select a Chatbot", [f"rb_chatbot{i}" for i in range(1, 16)])

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
							('rule_base', chatbot_selection, new_prompt, new_response, st.session_state.user['id'], st.session_state.user['school_id']))
				conn.commit()
				st.success("New rule added")
				st.rerun()

		# Select a row ID to delete
		if not df.empty:
			delete_id = st.selectbox("Select a row ID to delete", df['id'])
			if st.button("Delete Row"):
				cursor.execute("DELETE FROM Chatbot_Training_Records WHERE id = ?", (delete_id,))
				conn.commit()
				st.success(f"Row with ID {delete_id} deleted successfully!")
				st.rerun()

		conn.close()
	
	if st.button("Clear Chat"):
		clear_session_states()

	st.divider()
	st.subheader("Rule based Chatbot")

	# Initialize chat history
	if "messages" not in st.session_state:
		st.session_state.messages = []

	# Display chat messages from history on app rerun
	for message in st.session_state.messages:
		with st.chat_message(message["role"]):
			st.markdown(message["content"])

	# React to user input
	if prompt := st.chat_input("Enter your prompt"):
		cleaned_prompt = clean_string(prompt)

		# Find a matching response by comparing cleaned prompts
		matching_responses = df[df['prompt'].apply(clean_string) == cleaned_prompt]['response']
		reply = matching_responses.iloc[0] if not matching_responses.empty else "I don't understand"

		with st.chat_message("user"):
			st.write(prompt)
			st.session_state.messages.append({"role": "user", "content": prompt})
		with st.chat_message("assistant"):
			st.write(reply)
			st.session_state.messages.append({"role": "assistant", "content": reply})

	# Close the database connection
	conn.close()


