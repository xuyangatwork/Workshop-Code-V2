import streamlit as st
import openai
from openai import OpenAI
import sqlite3
from basecode.authenticate import return_api_key
from datetime import datetime
from langchain.memory import ConversationSummaryBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
import streamlit_antd_components as sac
# from nocode_workshop.k_map import (
# 	map_prompter_with_plantuml,
# 	generate_plantuml_mindmap,
# 	render_diagram
# )
import configparser
import os
from Markdown2docx import Markdown2docx

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

# def metacognitive_prompter(full_response):
# 	with st.status("Generating visuals..."):
# 		input = map_prompter_with_plantuml(full_response)
# 		uml = generate_plantuml_mindmap(input)
# 		image = render_diagram(uml)
# 		st.image(image, use_column_width=True)
# 		#input = map_prompter_with_mermaid_syntax(full_response)
# 		#generate_mindmap(input)

def response_download():
	docx_name = "crp" + st.session_state.user['username'] + ".docx"
	docx_path = os.path.join("chatbot_response", docx_name)
	
	if os.path.exists(docx_path):
# Provide the docx for download via Streamlit
		with open(docx_path, "rb") as docx_file:
			docx_bytes = docx_file.read()
			st.success("File is ready for downloading")
			st.download_button(
				label="Download document as DOCX",
				data=docx_bytes,
				file_name=docx_name,
				mime='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
			)
		os.remove(docx_path)
		st.session_state.button_text = 'Reset'
	else:
		st.warning("There is no lesson plan available for download.")

def complete_my_lesson():
	plan_action = sac.buttons([sac.ButtonsItem(label='Preview Responses', icon='eye', color='#00BFFF'),
							sac.ButtonsItem(label='Download Responses', icon='file-earmark-arrow-down', color='#40826D'),
							sac.ButtonsItem(label='Clear Responses', icon='file-earmark-arrow-down', color='#FF7F50')
								], index=None, format_func='title', size='small')
	
	
	if plan_action == 'Preview Responses':
		st.write(st.session_state.data_doc)

	elif plan_action == 'Download Responses':
		st.write("Downloading your lesson plan")
		md_filename = "crp" + st.session_state.user['username'] + ".md"
		md_filepath = os.path.join("chatbot_response", md_filename)
		if not os.path.exists("chatbot_response"):
			os.makedirs("chatbot_response")
		with open(md_filepath, 'w', encoding='utf-8') as file:
			file.write(st.session_state.data_doc)
		# Convert the markdown file to a docx
		base_filepath = os.path.join("chatbot_response", "crp" + st.session_state.user['username'])
		project = Markdown2docx(base_filepath)
		project.eat_soup()
		project.save()  # Assuming it saves the file with the same name but a .docx extension
		response_download()
	elif plan_action == 'Clear Responses':
		if st.checkbox("Clear Responses"):
			st.session_state.data_doc = ""
			st.success("Responses cleared")
	
def add_response(response): #add responses to the data_doc
	opt = sac.buttons([sac.ButtonsItem(label='Save Response', color='#40826D')], format_func='title', index=None, size='small')
	if add_response:
		st.session_state.data_doc = st.session_state.data_doc + "\n\n" + response
	return opt

#response rating component - consider for asynch feedback running on a separate thread
def rating_component():
	rating_value = sac.rate(label='Response ratings:', position='left', clear=True, value=2.0, align='left', size=15, color='#25C3B0')
	return rating_value

#insert data into the data table
def insert_into_data_table(date, chatbot_ans,user_prompt, tokens, function_name, value=0):
	conn = sqlite3.connect(WORKING_DATABASE)
	cursor = conn.cursor()

	# Insert data into Data_Table using preloaded session state value
	cursor.execute('''
		INSERT INTO Data_Table (date, user_id, profile_id, chatbot_ans, user_prompt, function_name, tokens, response_rating)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?)
	''', (date, st.session_state.data_profile["user_id"], st.session_state.data_profile["profile_id"],  chatbot_ans, user_prompt, function_name, tokens, value))

	conn.commit()
	conn.close()

#clear messages and memory
def clear_session_states():
	st.session_state.msg = []
	if "memory" not in st.session_state:
		pass
	else:
		del st.session_state["memory"]

def prompt_template_function(prompt, memory_flag, rag_flag):
	#check if there is kb loaded
	if st.session_state.vs:
		docs = st.session_state.vs.similarity_search(prompt)
		resource = docs[0].page_content
		source = docs[0].metadata
		st.session_state.rag_response = resource, source
	else:
		resource = ""
		source = ""

	if memory_flag:
		if "memory" not in st.session_state:
			st.session_state.memory = ConversationBufferWindowMemory(k=st.session_state.k_memory)
		mem = st.session_state.memory.load_memory_variables({})

	if rag_flag and memory_flag: #rag and memory only
		prompt_template = st.session_state.chatbot + f"""
							Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. 
							Search Result:
							{resource}
							{source}
							History of conversation:
							{mem}
							You must quote the source of the Search Result if you are using the search result as part of the answer"""
	
		return prompt_template
	
	elif rag_flag and not memory_flag: #rag kb only
		prompt_template = st.session_state.chatbot + f"""
						Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. 
						Search Result:
						{resource}
						{source}
						You must quote the source of the Search Result if you are using the search result as part of the answer"""
		return prompt_template
	
	elif not rag_flag and memory_flag: #memory only
		prompt_template = st.session_state.chatbot + f""" 
						History of conversation:
						{mem}"""
		return prompt_template
	else: #base bot nothing
		return st.session_state.chatbot


def base_bot(bot_name, memory_flag, rag_flag):
	openai.api_key = return_api_key()
	os.environ["OPENAI_API_KEY"] = return_api_key()
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
	messages = st.container(border=True)
		#showing the history of the chatbots
	for message in st.session_state.msg:
		with messages.chat_message(message["role"]):
			st.markdown(message["content"])
	#chat bot input
	try:
		if prompt := st.chat_input("Enter your query"):
			st.session_state.msg.append({"role": "user", "content": prompt})
			with messages.chat_message("user"):
				st.markdown(prompt)
			with messages.chat_message("assistant"):
				prompt_template = prompt_template_function(prompt, memory_flag, rag_flag)
				stream = client.chat.completions.create(
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
				response = st.write_stream(stream)
			st.session_state.msg.append({"role": "assistant", "content": response})
			if memory_flag:
				st.session_state["memory"].save_context({"input": prompt},{"output": response})
			# Insert data into the table
			now = datetime.now() # Using ISO format for date
			num_tokens = len(full_response + prompt)*1.3
			insert_into_data_table(now.strftime("%d/%m/%Y %H:%M:%S"),  response, prompt, num_tokens, bot_name)
			if st.session_state.download_response_flag == True:
				st.session_state.chat_response = add_response(response)
			
			
	except Exception as e:
		st.exception(e)
	



#below ------------------------------ base bot , summary memory for long conversation---------------------------------------------
#summary of conversation , requires another LLM call for every input, useful for feedback and summarising what was spoken
def memory_summary_component(prompt): #currently not in use
	if "memory" not in st.session_state:
		llm = ChatOpenAI(model_name=st.session_state.openai_model,temperature=st.session_state.temp)
		st.session_state.memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=1000)
	messages = st.session_state["memory"].chat_memory.messages
	#st.write("Messages ", messages)
	previous_summary = ""
	mem = st.session_state["memory"].predict_new_summary(messages, previous_summary)
	prompt_template = st.session_state.chatbot + f"""
						Summary of current conversation:
						{mem}"""
	

