#No need SQLite
import nltk
import streamlit as st
from streamlit_antd_components import menu, MenuItem
import streamlit_antd_components as sac
from basecode.main_bot import clear_session_states, complete_my_lesson, base_bot
from basecode.files_module import display_files,docs_uploader, delete_files
from basecode.kb_module import display_vectorstores, create_vectorstore, delete_vectorstores
from basecode.authenticate import login_function,check_password
from basecode.class_dash import download_data_table_csv
#from nocode_workshop.machine import upload_csv, plot_prices, prepare_data_and_train, plot_predictions, load_teachable_machines
from nocode_workshop.agent import agent_bot, agent_management, wiki_search, YouTubeSearchTool, DuckDuckGoSearchRun
from nocode_workshop.rule_base_api_chatbot import call_api, rule_based, group_rule_based, init_training_data
from nocode_workshop.faq_bot import faq_bot
from nocode_workshop.discussion_bot import discussion_bot, extract_and_combine_responses
from nocode_workshop.prompt_designs import prompt_designs_llm
from nocode_workshop.prototype_application import my_first_app, prototype_settings, my_first_app_advance
from nocode_workshop.analytics_dashboard import pandas_ai
from nocode_workshop.educational_bots import starting_bot,network_bot, language_bot, linking_bot
from nocode_workshop.empathy_bot import empathy_bot
#from nocode_workshop.assistant import assistant_demo, init_session_state
from nocode_workshop.k_map import map_creation_form
#New schema move function fom settings
from basecode.database_schema import create_dbs
from nocode_workshop.tool_bots import basic_analysis_bot
from nocode_workshop.knowledge_bot import rag_bot
import pandas as pd
import os
from basecode.database_module import (
	manage_tables, 
	delete_tables, 
	download_database, 
	upload_database, 
	upload_s3_database, 
	download_from_s3_and_unzip, 
	check_aws_secrets_exist,
	backup_s3_database,
	db_was_modified
	)
from basecode.org_module import (
	has_at_least_two_rows,
	initialise_admin_account,
	load_user_profile,
	display_accounts,
	create_org_structure,
	check_multiple_schools,
	process_user_profile,
	remove_or_reassign_teacher_ui,
	reassign_student_ui,
	change_teacher_profile_ui,
	add_user,
	streamlit_delete_interface,
	add_class,
	add_level,
)

from basecode.pwd_module import reset_passwords, password_settings
from basecode.users_module import (
	link_users_to_app_function_ui,
	set_function_access_for_user,
	create_prompt_template,
	update_prompt_template,
	vectorstore_selection_interface,
	pre_load_variables,
	load_and_fetch_vectorstore_for_user,
	link_profiles_to_vectorstore_interface
)

from basecode.bot_settings import bot_settings_interface, load_bot_settings
#from nocode_workshop.openai_features import generate_image, record_myself, upload_audio, analyse_image, text_to_speech, images_features,voice_features
from PIL import Image
import configparser
import ast
import ssl
              
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context





class ConfigHandler:
	def __init__(self):
		self.config = configparser.ConfigParser()
		self.config.read('config.ini')

	def get_value(self, section, key):
		value = self.config.get(section, key)
		try:
			# Convert string value to a Python data structure
			return ast.literal_eval(value)
		except (SyntaxError, ValueError):
			# If not a data structure, return the plain string
			return value

# Initialization
config_handler = ConfigHandler()

# Setting Streamlit configurations
st.set_page_config(layout="wide")

# Fetching secrets from Streamlit
DEFAULT_TITLE = st.secrets["default_title"]
SUPER_PWD = st.secrets["super_admin_password"]
SUPER = st.secrets["super_admin"]
DEFAULT_DB = st.secrets["default_db"]

# Fetching values from config.ini
DEFAULT_TEXT = config_handler.get_value('constants', 'DEFAULT_TEXT')
TCH = config_handler.get_value('constants', 'TCH')
STU = config_handler.get_value('constants', 'STU')
SA = config_handler.get_value('constants', 'SA')
AD = config_handler.get_value('constants', 'AD')
COTF = config_handler.get_value('constants', 'COTF')
META = config_handler.get_value('constants', 'META')
PANDAI = config_handler.get_value('constants', 'PANDAI')
MENU_FUNCS = config_handler.get_value('menu_lists', 'MENU_FUNCS')
META_BOT = config_handler.get_value('constants', 'META_BOT')
QA_BOT = config_handler.get_value('constants', 'QA_BOT')
LESSON_BOT = config_handler.get_value('constants', 'LESSON_BOT')
LESSON_COLLAB = config_handler.get_value('constants', 'LESSON_COLLAB')
LESSON_COMMENT = config_handler.get_value('constants', 'LESSON_COMMENT')
LESSON_MAP = config_handler.get_value('constants', 'LESSON_MAP')
REFLECTIVE = config_handler.get_value('constants', 'REFLECTIVE')
CONVERSATION = config_handler.get_value('constants', 'CONVERSATION')
MINDMAP = config_handler.get_value('constants', 'MINDMAP')
METACOG = config_handler.get_value('constants', 'METACOG')
ACK = config_handler.get_value('application_agreement', 'ACK')
PROTOTYPE = config_handler.get_value('constants', 'PROTOTYPE')
SEARCH = config_handler.get_value('constants', 'SEARCH')
DISCUSSION = config_handler.get_value('constants', 'DISCUSSION')

os.environ['TIKTOKEN_CACHE_DIR'] = st.secrets["NLTK_DATA"]
os.environ['NLTK_DATA'] = st.secrets["NLTK_DATA"]

def is_function_disabled(function_name):
	#st.write("Function name: ", function_name)
	#st.write("Function options: ", st.session_state.func_options.get(function_name, True))
	return st.session_state.func_options.get(function_name, True)

def return_function_name(function_name, default_name = ""):
	if st.session_state.func_options.get(function_name, True):
		return "-"
	else:
		if default_name == "":
			return function_name
		else:
			return default_name

def initialize_session_state( menu_funcs, default_value):
	st.session_state.func_options = {key: default_value for key in menu_funcs.keys()}

def main():
	try:
		if "title_page"	not in st.session_state:
			st.session_state.title_page = DEFAULT_TITLE 

		st.title(st.session_state.title_page)
		sac.divider(label='Exploring Generative Artificial Intelligence - Author Joe Tay', icon='house')

		# Define the NLTK data directory
		nltk_data_dir = st.secrets["NLTK_DATA"]

		# Ensure the NLTK data directory exists
		if not os.path.exists(nltk_data_dir):
			os.makedirs(nltk_data_dir, exist_ok=True)

		# Update the NLTK data path to include the custom directory
		nltk.data.path.append(nltk_data_dir)

		def download_nltk_data_if_absent(package_name):
			try:
				# Try loading the package to see if it exists in the custom directory
				nltk.data.find("tokenizers/" + package_name)
			except LookupError:
				# If the package doesn't exist, download it to the specified directory
				nltk.download(package_name, download_dir=nltk_data_dir)

		# Example usage
		download_nltk_data_if_absent('punkt')
		download_nltk_data_if_absent('stopwords')
		
		if "api_key" not in st.session_state:
			st.session_state.api_key = ""

		if "option" not in st.session_state:
			st.session_state.option = False
		
		if "login" not in st.session_state:
			st.session_state.login = False
		
		if "user" not in st.session_state:
			st.session_state.user = None
		
		if "start" not in st.session_state:
			st.session_state.start = 0
		
		if "openai_model" not in st.session_state:
			st.session_state.openai_model = st.secrets["default_model"]

		if "msg" not in st.session_state:
			st.session_state.msg = []

		if "rating" not in st.session_state:
			st.session_state.rating = False

		# if "lesson_plan" not in st.session_state:
		# 	st.session_state.lesson_plan = ""

		if "temp" not in st.session_state:
			st.session_state.temp = st.secrets["default_temp"]
		
		if "acknowledgement" not in st.session_state:
			st.session_state.acknowledgement = False
		
		if "frequency_penalty" not in st.session_state:
			st.session_state.frequency_penalty = st.secrets["default_frequency_penalty"]

		if "presence_penalty" not in st.session_state:
			st.session_state.presence_penalty = st.secrets["default_presence_penalty"]

		if "k_memory" not in st.session_state:
			st.session_state.k_memory = st.secrets["default_k_memory"]
		
		if "memoryless" not in st.session_state:
			st.session_state.memoryless = False

		if "vs" not in st.session_state:
			st.session_state.vs = False
		
		# if "visuals" not in st.session_state:
		# 	st.session_state.visuals = False
		
		if "svg_height" not in st.session_state:
			st.session_state["svg_height"] = 1000
			
		if "current_model" not in st.session_state:
			st.session_state.current_model = "No KB loaded"

		if "func_options" not in st.session_state:
			st.session_state.func_options = {}
			initialize_session_state(MENU_FUNCS, True)
		
		if "tools" not in st.session_state:
			st.session_state.tools = []
		
		# if "lesson_col_prompt" not in st.session_state:
		# 	st.session_state.lesson_col_prompt = False

		# if "lesson_col_option" not in st.session_state:
		# 	st.session_state.lesson_col_option = 'Cancel'
		
		# if "generated_flag" not in st.session_state:
		# 	st.session_state.generated_flag = False
		
		# if "button_text" not in st.session_state:
		# 	st.session_state.button_text = "Cancel"
		
		if "data_doc" not in st.session_state:
			st.session_state.data_doc = ""
		
		if "download_response_flag" not in st.session_state:
			st.session_state.download_response_flag = False
		
		if "chatbot_index" not in st.session_state:
			st.session_state.chatbot_index = 1

		if "chat_response" not in st.session_state:
			st.session_state.chat_response = ""
		
		if "analyse_discussion" not in st.session_state:
			st.session_state.analyse_discussion = False

		if "rag_response" not in st.session_state:
			st.session_state.rag_response  = "", ""
		
		# if "test2"	not in st.session_state:
		# 	st.session_state.test2 = ""
		
		#These functions below will create the initial database and administator account
		create_dbs()
		initialise_admin_account()

		#PLEASE REMOVE THIS or COMMENT IT 
		#st.write("User Profile: ", st.session_state.user)
		
		#PLEASE REMOVE ABOVE
		with st.sidebar: #options for sidebar
			
			if st.session_state.login == False:
				st.image("app_logo/AI logo.png")
				st.session_state.option = menu([MenuItem('Users login', icon='people')])
			else:
				#can do a test if user is school is something show a different logo and set a different API key
				if st.session_state.user['profile_id'] == SA: #super admin login feature
					# Initialize the session state for function options	
					initialize_session_state(MENU_FUNCS, False)
				else:
					if st.session_state.acknowledgement == False:
						initialize_session_state(MENU_FUNCS, True)
					else:
						set_function_access_for_user(st.session_state.user['id'])
						#st.write("Function options: ", st.session_state.func_options)
					# Using the is_function_disabled function for setting the `disabled` attribute
				st.session_state.option = sac.menu([
					sac.MenuItem('Home', icon='house', children=[
						sac.MenuItem(return_function_name('Personal Dashboard'), icon='person-circle', disabled=is_function_disabled('Personal Dashboard')),
						#sac.MenuItem('Class Dashboard', icon='clipboard-data', disabled=is_function_disabled('Class Dashboard')),
					]),

					sac.MenuItem('GenAI Capabilities', icon='book', children=[
						sac.MenuItem(return_function_name('AI Analytics'), icon='graph-up', disabled=is_function_disabled('AI Analytics')),
						sac.MenuItem(return_function_name('Knowledge Graph Bot'), icon='diagram-3', disabled=is_function_disabled('Knowledge Graph Bot')),
						sac.MenuItem(return_function_name('Empathy Bot'), icon='chat-heart', disabled=is_function_disabled('Empathy Bot')),
					]),	

					sac.MenuItem('Educational Chatbots', icon='book', children=[
						sac.MenuItem(return_function_name('Starting Bot'), icon='skip-start',disabled=is_function_disabled('Starting Bot')), #visual image
						sac.MenuItem(return_function_name('Connecting Bot','Learning Bot'), icon='diagram-3',disabled=is_function_disabled('Connecting Bot')), #Graph agent lang graph
						sac.MenuItem(return_function_name('Language Support Bot', 'Language Bot'), icon='chat', disabled=is_function_disabled('Language Support Bot')), #Language
						sac.MenuItem(return_function_name('Linking Bot'), icon='emoji-smile',disabled=is_function_disabled('Linking Bot')), #Image generator and voice
					]),	

					sac.MenuItem('Types of ChatBots', icon='person-fill-gear', children=[
						sac.MenuItem(return_function_name('Discussion Chatbot'), icon='people', disabled=is_function_disabled('Discussion Chatbot')),
						sac.MenuItem(return_function_name('Rule Based Chatbot'), icon='chat-dots', disabled=is_function_disabled('Rule Based Chatbot')),
						sac.MenuItem(return_function_name('FAQ AI Chatbot'), icon='chat-dots', disabled=is_function_disabled('FAQ AI Chatbot')),
						sac.MenuItem(return_function_name('Open AI API Call', 'LLM API call'), icon='arrow-left-right', disabled=is_function_disabled('Open AI API Call')),
						sac.MenuItem(return_function_name('Prompt Designs'), icon='arrow-left-right', disabled=is_function_disabled('Prompt Designs')),
						sac.MenuItem(return_function_name('AI Chatbot'), icon='chat-dots', disabled=is_function_disabled('AI Chatbot')),
						sac.MenuItem(return_function_name('Agent Chatbot'), icon='chat-dots', disabled=is_function_disabled('Agent Chatbot')),
						sac.MenuItem(return_function_name('Chatbot Management', 'Bot & Prompt Management'), icon='wrench', disabled=is_function_disabled('Chatbot Management')),
						sac.MenuItem(return_function_name('Prototype Application'), icon='star-fill', disabled=is_function_disabled('Prototype Application')),
						sac.MenuItem(return_function_name('Prototype Settings'), icon='wrench', disabled=is_function_disabled('Prototype Settings')),
						
					]),
					sac.MenuItem('Knowledge Base Tools', icon='book', children=[
						sac.MenuItem(return_function_name('Files management', 'Files Management'), icon='file-arrow-up', disabled=is_function_disabled('Files management')),
						sac.MenuItem(return_function_name('KB management', 'Knowledge Base Editor'), icon='database-fill-up',disabled=is_function_disabled('KB management')),
					]),
					sac.MenuItem('Organisation Tools', icon='buildings', children=[
						sac.MenuItem(return_function_name( 'Organisation Management','Org Management'), icon='building-gear', disabled=is_function_disabled('Organisation Management')),
						sac.MenuItem(return_function_name('School Users Management', 'Users Management'), icon='house-gear', disabled=is_function_disabled('School Users Management')),
					]),
					sac.MenuItem(type='divider'),
					sac.MenuItem('Profile Settings', icon='gear'),
					sac.MenuItem('Application Info', icon='info-circle'),
					sac.MenuItem('Logout', icon='box-arrow-right'),
				], index=st.session_state.start, format_func='title', open_all=True)
		
		if st.session_state.option == 'Users login':
				col1, col2 = st.columns([3,4])
				placeholder = st.empty()
				with placeholder:
					with col1:
						if login_function() == True:
							st.session_state.user = load_user_profile(st.session_state.user)
							pre_load_variables(st.session_state.user['id'])
							load_and_fetch_vectorstore_for_user(st.session_state.user['id'])
							load_bot_settings(st.session_state.user['id'])
							st.session_state.login = True
							placeholder.empty()
							st.rerun()
					with col2:
						pass
		elif st.session_state.option == 'Home':
			col1, col2 = st.columns([3,1])
			with col1:
				st.subheader("Acknowledgement on the use of Generative AI with Large Language Models")
				initialize_session_state(MENU_FUNCS, True)
				st.write(ACK)
				ack = st.checkbox("I acknowledge the above information")
				if ack:
					st.session_state.acknowledgement = True
					set_function_access_for_user(st.session_state.user['id'])
					st.session_state.start = 1
					st.rerun()
				else:
					st.warning("Please acknowledge the above information before you proceed")
					initialize_session_state(MENU_FUNCS, True)
					st.stop()
				pass
			with col2:
				pass
		
		#Personal Dashboard
		elif st.session_state.option == 'Personal Dashboard':
			st.subheader(f":green[{st.session_state.option}]")
			if st.session_state.user['profile_id'] == SA:
				sch_id, msg = process_user_profile(st.session_state.user["profile_id"])
				st.write(msg)
				download_data_table_csv(st.session_state.user["id"], sch_id, st.session_state.user["profile_id"])
			else:
				download_data_table_csv(st.session_state.user["id"], st.session_state.user["school_id"], st.session_state.user["profile_id"])
			display_vectorstores()
			vectorstore_selection_interface(st.session_state.user['id'])
	
		elif st.session_state.option == 'AI Analytics':
			# Code for AI Analytics
			st.subheader(f":green[{st.session_state.option}]")
			if st.toggle('Switch on to PandasAI Analytics'):
				pandas_ai(st.session_state.user['id'], st.session_state.user['school_id'], st.session_state.user['profile_id'])
			else:
				basic_analysis_bot()
		elif st.session_state.option == 'Knowledge Graph Bot':
			# Code for Image Generator
			st.subheader(f":green[{st.session_state.option}]")
			st.session_state.chatbot = st.session_state.knowledge_graph_bot
			rag_bot()
			pass
		elif st.session_state.option == 'Empathy Bot':
			if "TWILIO_ACCOUNT_SID" in st.secrets and "TWILIO_AUTH_TOKEN" in st.secrets:
				st.subheader(f":green[{st.session_state.option}]")
				st.session_state.chatbot = st.session_state.empathy_bot
				empathy_bot()
			else:
				st.warning("Please set up your Twilio account in the secrets.toml file")
				st.write("Feature not available")
			pass
		elif st.session_state.option == 'Starting Bot':
			# Code for Starting Bot
			st.subheader(f":green[{st.session_state.option}]")
			st.session_state.chatbot = st.session_state.start_bot
			starting_bot()
		
			pass
		elif st.session_state.option == 'Learning Bot':
			# Code for Starting Bot
			st.subheader(f":green[{st.session_state.option}]")
			st.session_state.chatbot = st.session_state.connecting_bot
			network_bot()
			pass
		elif st.session_state.option == 'Language Bot':
			# Code for Language Bot
			st.subheader(f":green[{st.session_state.option}]")
			st.session_state.chatbot = st.session_state.language_support_bot
			language_bot()
			pass

		elif st.session_state.option == 'Linking Bot':
			# Code for Linking Bot
			st.subheader(f":green[{st.session_state.option}]")
			st.session_state.chatbot = st.session_state.linking_bot
			linking_bot()
		#========================Workshop Tools=======================================================#
		elif st.session_state.option == 'Rule Based Chatbot':
			# Code for Rule Based Chatbot - Zerocode
			if st.session_state.user['profile_id'] == SA:
				with st.expander("Rule Based Chatbot Settings"):
					rb_chatbot = st.checkbox("I will delete and initialise training data for rule based chatbot")
					if st.button("Initialise Training Data") and rb_chatbot:
						init_training_data()
					pass

			personal = st.toggle('Switch on to access the Personal Chatbot')
			if personal:
				rule_based()
			else:
				group_rule_based()
		elif st.session_state.option == 'Discussion Chatbot':
			# Code for FAQ AI chatbot
			if "extract_data" not in st.session_state:
				st.session_state.extract_data = ""
			if st.session_state.user['profile_id'] == SA:
				with st.expander("Discussion Bot Settings"):
					analyse_responses = st.toggle('Switch on to analyse responses')
					if analyse_responses:
						if st.button("Extract Responses"):
							st.session_state.extract_data = extract_and_combine_responses()
						st.session_state.analyse_discussion = True
						st.write("Discussion Data: ", st.session_state.extract_data)
					else:
						st.session_state.analyse_discussion = False
			
					dis_bot = st.checkbox("I will delete and initialise training data for discussion bot")
					if st.button("Initialise Training Data") and dis_bot:
						init_training_data()
					pass

			if st.session_state.analyse_discussion:
				prompt = st.session_state.extraction_prompt + "/n" + st.session_state.extract_data  + "/n" + "Please analyse the response and answer the questions below"
			else:		
				prompt = st.session_state.discussion_bot
			
			if st.session_state.user['profile_id'] == SA:
				if st.button("Generate Report"):
					prompt = st.session_state.discussion_bot_report + "/n" + st.session_state.extract_data
			discussion_bot(DISCUSSION, prompt)
			pass
		elif st.session_state.option == 'FAQ AI Chatbot':
			if st.session_state.user['profile_id'] == SA:
				with st.expander("FAQ Bot Settings"):
					faq = st.checkbox("I will delete and initialise training data for FAQ bot")
					if st.button("Initialise Training Data") and faq:
						init_training_data()
					pass
			# Code for FAQ AI chatbot
			faq_bot()
			pass
		elif st.session_state.option == 'LLM API call':
			# Code for Open AI API Call
			call_api()
			pass
		elif st.session_state.option == 'Prompt Designs':
			# Code for Open AI API Call
			prompt_designs_llm()
			pass
		elif st.session_state.option == 'Prototype Application':
			# Code for Prototype Application - Zerocode
			st.subheader(f":green[{st.session_state.option}]")
			options = sac.chip(items=[
								sac.ChipItem(label='Form prototype', icon='card-text'),
								sac.ChipItem(label='Chatbot prototype', icon='chat'),
							], format_func='title', radius='sm', size='sm', align='left', variant='light')
			if options == 'Form prototype':
				my_first_app(PROTOTYPE)
			else:
				my_first_app_advance(PROTOTYPE)
			pass
		elif st.session_state.option == 'Prototype Settings':
			# Code for Prototype Settings - Zerocode
			st.subheader(f":green[{st.session_state.option}]")
			prototype_settings()
			pass

		elif st.session_state.option == 'AI Chatbot':
			#Code for AI Chatbot - ZeroCode
			#st.write("Current Chatbot Template: ", st.session_state.chatbot)
			#check if API key is entered
			with st.expander("Chatbot Settings"):
				vectorstore_selection_interface(st.session_state.user['id'])
				#new options --------------------------------------------------------
				if st.session_state.vs:
					vs_flag = False
				else:
					vs_flag = True
				options = sac.chip(
							items=[
								sac.ChipItem(label='Raw Search', icon='search', disabled=vs_flag),
								sac.ChipItem(label='Enable Memory', icon='memory'),
								#sac.ChipItem(label='Rating Function', icon='star-fill'),
								sac.ChipItem(label='Capture Responses', icon='camera-fill'),
								sac.ChipItem(label='Download Responses', icon='download'),
							], index=[1, 2], format_func='title', radius='sm', size='sm', align='left', variant='light', multiple=True)
				# Update state based on new chip selections
				raw_search = 'Raw Search' in options
				st.session_state.memoryless = 'Enable Memory' not in options
				st.session_state.rating = 'Rating Function' in options
				st.session_state.download_response_flag = 'Capture Responses' in options
				preview_download_response = 'Download Responses' in options
				
				if preview_download_response:
					complete_my_lesson()

				if st.button("Clear Chat"):
						clear_session_states()

			b1, b2 = st.columns([3,1])

			with b1:

				if st.session_state.vs:#chatbot with knowledge base
					if st.session_state.memoryless: #memoryless chatbot with knowledge base but no memory
						base_bot(QA_BOT, False, True)
					else:
						base_bot(QA_BOT, True, True) #chatbot with knowledge base and memory
				else:#chatbot with no knowledge base
					if st.session_state.memoryless: #memoryless chatbot with no knowledge base and no memory
						base_bot(QA_BOT, False, False)
					else:
						base_bot(QA_BOT, True, False) #chatbot with no knowledge base but with memory
			with b2:
				with st.container(border=True):
					st.write("RAG Results")
					resource, source = st.session_state.rag_response
					st.write("Resource: ", resource)
					st.write("Source : ", source)
				with st.container(border=True):
					st.write("Chat Memory")
					if "memory" not in st.session_state:
						st.write("No memory")
					else:
						st.write(st.session_state.memory.load_memory_variables({}))
				
		elif st.session_state.option == "Agent Chatbot":
			if st.session_state.tools == []:
				st.warning("Loading Wikipedia Search, Internet Search and YouTube Search, you may select your tools in Bot & Prompt management")
				st.session_state.tools =  [wiki_search, DuckDuckGoSearchRun(name="Internet Search"), YouTubeSearchTool()]
				agent_bot()
			else:
				agent_bot()
			
		elif st.session_state.option == 'Bot & Prompt Management': #ensure that it is for administrator or super_admin
			if st.session_state.user['profile_id'] == SA or st.session_state.user['profile_id'] == AD:
				st.subheader(f":green[{st.session_state.option}]")
				templates = create_prompt_template(st.session_state.user['id'])
				st.divider()
				# st.write("Templates created: ", templates)
				update_prompt_template(st.session_state.user['profile_id'], templates)
				st.subheader("Agent Management")
				agent_management()
				if st.session_state.user['profile_id'] == SA:
					st.subheader("OpenAI Chatbot Parameters Settings")
					bot_settings_interface(st.session_state.user['profile_id'], st.session_state.user['school_id'])	
			else:
				st.subheader(f":red[This option is accessible only to administrators only]")
		
		#Knowledge Base Tools
		elif st.session_state.option == 'Files Management':
			st.subheader(f":green[{st.session_state.option}]") 
			display_files()
			docs_uploader()
			delete_files()

		elif st.session_state.option == "Knowledge Base Editor":
			st.subheader(f":green[{st.session_state.option}]") 
			options = sac.steps(
				items=[
					sac.StepsItem(title='Step 1', description='Create a new knowledge base'),
					sac.StepsItem(title='Step 2', description='Assign a knowledge base to a user'),
					sac.StepsItem(title='Step 3', description='Delete a knowledge base (Optional)'),
				],
				format_func='title',
				placement='vertical',
				size='small'
			)
			if options == "Step 1":
				st.subheader("KB created in the repository")
				display_vectorstores()
				st.subheader("Files available in the repository")
				display_files()
				create_vectorstore()
			elif options == "Step 2":
				st.subheader("KB created in the repository")
				display_vectorstores()
				vectorstore_selection_interface(st.session_state.user['id'])
				link_profiles_to_vectorstore_interface(st.session_state.user['id'])
	
			elif options == "Step 3":
				st.subheader("KB created in the repository")
				display_vectorstores()
				delete_vectorstores()

		#Organisation Tools
		elif st.session_state.option == "Users Management":
			if st.session_state.user['profile_id'] == SA or st.session_state.user['profile_id'] == AD:	
				st.subheader(f":green[{st.session_state.option}]") 
				sch_id, msg = process_user_profile(st.session_state.user["profile_id"])
				rows = has_at_least_two_rows()
				if rows >= 2:
					#Password Reset
					st.subheader("User accounts information")
					df = display_accounts(sch_id)
					st.warning("Password Management")
					st.subheader("Reset passwords of users")
					reset_passwords(df)
					add_user(sch_id)
			else:
				st.subheader(f":red[This option is accessible only to administrators only]")
		
		elif st.session_state.option == "Org Management":
			if st.session_state.user['profile_id'] == SA:
				st.subheader(f":green[{st.session_state.option}]") 
				#direct_vectorstore_function()
				
				if check_password(st.session_state.user["username"], SUPER_PWD):
						st.write("To start creating your teachers account, please change the default password of your administrator account under profile settings")
				else:
					sch_id, msg = process_user_profile(st.session_state.user["profile_id"])
					create_flag = False
					rows = has_at_least_two_rows()
					if rows >= 2:
						create_flag = check_multiple_schools()
					st.markdown("###")
					st.write(msg)
					st.markdown("###")
					steps_options = sac.steps(
								items=[
									sac.StepsItem(title='Create new school', disabled=create_flag),
									sac.StepsItem(title='Assign Teachers'),
									sac.StepsItem(title='Change Teachers Profile'),
									sac.StepsItem(title='Set function access'),
									sac.StepsItem(title='Reassign Students'),
									sac.StepsItem(title='Edit Classes and Levels'),
									sac.StepsItem(title='Manage SQL Tables',icon='radioactive'),
								], color='lime'
							)
					if steps_options == "Create new school":
						if create_flag:
							st.write("School created, click on Step 2")
						else:
							create_org_structure()
					elif steps_options == "Assign Teachers":
						remove_or_reassign_teacher_ui(sch_id)
					elif steps_options == "Change Teachers Profile":
						change_teacher_profile_ui(sch_id)
					elif steps_options == "Set function access":
						link_users_to_app_function_ui(sch_id)
					elif steps_options == "Reassign Students":
						reassign_student_ui(sch_id)
					elif steps_options == "Edit Classes and Levels":
						add_level(sch_id)
						st.divider()
						add_class(sch_id)
						st.divider()
						streamlit_delete_interface()
					elif steps_options == "Manage SQL Tables":
						st.subheader(":red[Managing SQL Schema Tables]")
						st.warning("Please do not use this function unless you know what you are doing")
						if st.checkbox("I know how to manage SQL Tables"):
							st.subheader(":red[Zip Database - Download and upload a copy of the database]")
							download_database()
							upload_database()
							if check_aws_secrets_exist():
								st.subheader(":red[Upload Database to S3 - Upload a copy of the database to S3]")
								upload_s3_database()
								download_from_s3_and_unzip()
							st.subheader(":red[Display and Edit Tables - please do so if you have knowledge of the current schema]")
							manage_tables()
							st.subheader(":red[Delete Table - Warning please use this function with extreme caution]")
							delete_tables()
			else:
				st.subheader(f":red[This option is accessible only to super administrators only]")
						
		
		elif st.session_state.option == "Profile Settings":
			st.subheader(f":green[{st.session_state.option}]") 
			#direct_vectorstore_function()
			password_settings(st.session_state.user["username"])

		elif st.session_state.option == 'Application Info':
			st.subheader(f":green[{st.session_state.option}]") 
			col1, col2 = st.columns([3,1])
			with col1:
				st.subheader("Acknowledgement on the use of Generative AI with Large Language Models")
				initialize_session_state(MENU_FUNCS, True)
				st.write(ACK)
				if st.session_state.acknowledgement == True:
					st.success("You have acknowledged the above information")
				else:
					ack = st.checkbox("I acknowledge the above information")
					if ack:
						st.session_state.acknowledgement = True
						set_function_access_for_user(st.session_state.user['id'])
						st.session_state.start = 1
						st.rerun()
					else:
						st.warning("Please acknowledge the above information before you proceed")
						initialize_session_state(MENU_FUNCS, True)
						st.stop()
					pass
			with col2:
				pass

		elif st.session_state.option == 'Logout':
			if db_was_modified(DEFAULT_DB):
				if check_aws_secrets_exist():
					backup_s3_database()
					for key in st.session_state.keys():
						del st.session_state[key]
					st.rerun()
				elif st.session_state.user['profile_id'] == SA:
					on = st.toggle('I do not want to download a copy of the database')
					if on:
						for key in st.session_state.keys():
							del st.session_state[key]
						st.rerun()
					else:
						download_database()
						for key in st.session_state.keys():
							del st.session_state[key]
						st.rerun()
				else:
					for key in st.session_state.keys():
						del st.session_state[key]
					st.rerun()
			else:
				for key in st.session_state.keys():
					del st.session_state[key]
				st.rerun()
					
	except Exception as e:
		st.exception(e)

if __name__ == "__main__":
	main()
