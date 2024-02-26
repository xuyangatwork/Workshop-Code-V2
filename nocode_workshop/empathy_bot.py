import streamlit as st
import av
from streamlit_webrtc import webrtc_streamer
from basecode.authenticate import return_api_key
from basecode.users_module import vectorstore_selection_interface
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationSummaryBufferMemory
from nocode_workshop.k_map import map_prompter_with_plantuml, generate_plantuml_mindmap
from datetime import datetime
from langchain.chat_models import ChatOpenAI
from basecode.main_bot import add_response, insert_into_data_table, complete_my_lesson, response_download
#from st_audiorec import st_audiorec
import os
from PIL import Image
import openai
import google.generativeai as genai
import requests
import base64
import tempfile
import io
from openai import OpenAI
import streamlit_antd_components as sac
from nocode_workshop.k_map import map_creation_form, map_prompter_with_plantuml_form, generate_plantuml_mindmap, render_diagram
import requests
from Markdown2docx import Markdown2docx
import configparser
# import spacy_streamlit
# import spacy
import ast
from fer import FER
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import imageio
import matplotlib
import time
import threading
from twilio.rest import Client
import collections
import tempfile

# Initialize a dictionary to store emotion statistics
emotion_summary = collections.defaultdict(lambda: {'count': 0, 'score': 0})



lock = threading.Lock()
img_container = {"img": None}
#nlp = spacy.load("en_core_web_sm")

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

START_BOT = config_handler.get_value('constants', 'START_BOT')
EMPATHY_BOT = config_handler.get_value('constants', 'EMPATHY_BOT')
QA_BOT = config_handler.get_value('constants', 'QA_BOT')
CONNECT_BOT = config_handler.get_value('constants', 'CONNECT_BOT')
LANGUAGE_BOT = config_handler.get_value('constants', 'LANGUAGE_BOT')
LINKING_BOT = config_handler.get_value('constants', 'LINKING_BOT')
START_PROMPT1 = config_handler.get_value('Prompt_Design_Templates', 'START_PROMPT1')
START_PROMPT2 = config_handler.get_value('Prompt_Design_Templates', 'START_PROMPT2')
START_PROMPT3 = config_handler.get_value('Prompt_Design_Templates', 'START_PROMPT3')
START_PROMPT4 = config_handler.get_value('Prompt_Design_Templates', 'START_PROMPT4')

cwd = os.getcwd()
AUDIO_DIRECTORY = os.path.join(cwd, "audio_files")

if not os.path.exists(AUDIO_DIRECTORY):
	os.makedirs(AUDIO_DIRECTORY)

openai.api_key = return_api_key()

client = OpenAI(
	# defaults to os.environ.get("OPENAI_API_KEY")
	api_key=return_api_key(),
)

# Function to encode the image
def encode_image(image_path):
	with open(image_path, "rb") as image_file:
		return base64.b64encode(image_file.read()).decode('utf-8')

# Function to get file extension
def get_file_extension(file_name):
	return os.path.splitext(file_name)[-1]

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

def clear_session_states():
	st.session_state.msg = []
	if st.session_state.overall_emotion:
		st.session_state.overall_emotion = "Neutral"
	if "memory" not in st.session_state:
		pass
	else:
		del st.session_state["memory"]


#===============================Empathy Bot===========================================

def empathy_bot():
	with st.expander("Chatbot Settings"):
		vectorstore_selection_interface(st.session_state.user['id'])
		options = sac.chip(
					items=[
						sac.ChipItem(label='Capture Responses', icon='camera-fill'),
						sac.ChipItem(label='Download Responses', icon='download'),
					], index=[1, 2], format_func='title', radius='sm', size='sm', align='left', variant='light', multiple=True)
		# Update state based on new chip selections
		st.session_state.download_response_flag = 'Capture Responses' in options
		preview_download_response = 'Download Responses' in options
		
		if preview_download_response:
			complete_my_lesson()

		if st.button("Clear Chat"):
			clear_session_states()
	
	j1, j2 = st.columns([3,2])

	with j1:
		if st.session_state.vs:
			empathy_base_bot(EMPATHY_BOT, True, True)
		else:
			empathy_base_bot(EMPATHY_BOT, True, False)
	with j2:
		with st.container(border=True):
			#web_capture()
			image_capture()
		pass
	
def prompt_template_function_empathy(prompt, memory_flag, rag_flag):
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




def empathy_base_bot(bot_name, memory_flag, rag_flag):

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
				prompt_template = prompt_template_function_empathy(prompt, memory_flag, rag_flag)
				stream = client.chat.completions.create(
					model=st.session_state.openai_model,
					messages=[
						{"role": "system", "content":prompt_template },
						{"role": "user", "content": f"Take note of the following user emotion which is currently {st.session_state.overall_emotion} and adjust your conversation accordingly" + prompt},
					],
					temperature=st.session_state.temp, #settings option
					presence_penalty=st.session_state.presence_penalty, #settings option
					frequency_penalty=st.session_state.frequency_penalty, #settings option
					stream=True #settings option
				)
				response = st.write_stream(stream)
			
			st.session_state.msg.append({"role": "assistant", "content": response})
			st.session_state["memory"].save_context({"input": prompt},{"output": response})
			# Insert data into the table
			now = datetime.now() # Using ISO format for date
			num_tokens = len(full_response + prompt)*1.3
			insert_into_data_table(now.strftime("%d/%m/%Y %H:%M:%S"),  response, prompt, num_tokens, bot_name)
			if st.session_state.download_response_flag == True:
				st.session_state.chat_response = add_response(response)
			
			
	except Exception as e:
		st.exception(e)


def update_emotion_statistics(current_emotions):
	"""Update the emotion statistics with the current emotions detected."""
	for emotion, score in current_emotions.items():
		emotion_summary[emotion]['count'] += 1
		emotion_summary[emotion]['score'] += score

def get_overall_emotion():
	"""Determine the overall emotion based on the aggregated statistics."""
	# Ensure there is data to process
	if not emotion_summary:
		return "No Data", 0  # Or any other default or indicative value

	# Calculate the average score for each emotion
	for emotion in emotion_summary:
		if emotion_summary[emotion]['count'] > 0:  # Ensure division by zero is not possible
			emotion_summary[emotion]['average'] = emotion_summary[emotion]['score'] / emotion_summary[emotion]['count']
		else:
			emotion_summary[emotion]['average'] = 0

	# Find the emotion with the highest average score
	overall_emotion = max(emotion_summary, key=lambda e: emotion_summary[e]['average'])
	return overall_emotion, emotion_summary[overall_emotion]['average']

def image_capture():
	if "overall_emotion" not in st.session_state:
		st.session_state.overall_emotion = "Neutral"
	if "overall_score" not in st.session_state:
		st.session_state.overall_score = 0
	# Display instructions to the user
	st.write("Take a picture of yourself before we start the conversation")

	# Capture an image from the user
	uploaded_file = st.camera_input("Capture")
	
	# Ensure an image was captured
	if uploaded_file is not None:
		bytes_data = uploaded_file.getvalue()
		
		# Convert the bytes data to a PIL Image
		img = Image.open(io.BytesIO(bytes_data))
		# Initialize the FER detector with MTCNN
		img = Image.open(io.BytesIO(bytes_data))
		
		# Save the PIL Image to a temporary file
		with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
			img.save(tmp, format="JPEG")
			tmp_path = tmp.name  # Store the temporary file path

		detector = FER(mtcnn=True)
		
		# Convert the captured image to the format expected by FER (if necessary)
		# This might involve converting the image from a PIL format to an array, etc.
		# The conversion depends on how Streamlit returns the camera input and how FER expects it.
		
		# Detect emotions in the image
		result = detector.detect_emotions(tmp_path)
		largest_face = None
		max_area = 0
		
		# Find the largest face for primary emotion analysis
		for face in result:
			box = face["box"]
			x, y, w, h = box
			area = w * h
			if area > max_area:
				max_area = area
				largest_face = face
		
		# If a face is detected, display the emotion
		if largest_face:
			current_emotions = largest_face["emotions"]
			emotion_type = max(current_emotions, key=current_emotions.get)
			emotion_score = current_emotions[emotion_type]
			
			# Display the detected emotion and its score
			emotion_text = f"Detected emotion: {emotion_type} with a confidence of {emotion_score:.2f}"
			st.session_state.overall_emotion = emotion_type
			st.session_state.overall_score = emotion_score
			st.write(emotion_text)
		else:
			st.write("No face detected. Please try again.")
	else:
		st.write("No image captured. Please take a picture to proceed.")
	

def web_capture():
	if "overall_emotion" not in st.session_state:
		st.session_state.overall_emotion = ""
	if "overall_score" not in st.session_state:
		st.session_state.overall_score = 0
	account_sid = st.secrets['TWILIO_ACCOUNT_SID']
	auth_token = st.secrets['TWILIO_AUTH_TOKEN']
	client = Client(account_sid, auth_token)
	token = client.tokens.create()
	detector = FER(mtcnn=True)
	emotion_statistics = []
	ctx = webrtc_streamer(key="example", video_frame_callback=video_frame_callback, rtc_configuration={"iceServers": token.ice_servers})

	# Set up a matplotlib figure for displaying live emotion detection results
	plt.ion()  # Turn on interactive mode for live updates
	fig, ax = plt.subplots()
	emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
	bars = ax.bar(emotion_labels, [0]*7, color='lightblue') # Initialize bars for each emotion
	plt.ylim(0, 1)
	plt.ylabel('Confidence')
	plt.title('Real-time Emotion Detection')
	ax.set_xticklabels(emotion_labels, rotation=45)

	# Initialize imageio writer to save live chart updates as a GIF
	gif_writer = imageio.get_writer('emotion_chart.gif', mode='I', duration=0.1)

	# List to store cumulative emotion statistics for each frame
	emotion_statistics = []

	emotion_text_placeholder = st.empty()

	fig_place = st.empty()
	fig, ax = plt.subplots(1, 1)

	while ctx.state.playing:
		with lock:
			img = img_container["img"]
		if img is None:
			continue

		result = detector.detect_emotions(img)
		largest_face = None
		max_area = 0

		# Find the largest face in the frame for primary emotion analysis
		for face in result:
			box = face["box"]
			x, y, w, h = box
			area = w * h
			if area > max_area:
				max_area = area
				largest_face = face

		# If a face is detected, display the emotion and update the chart
		if largest_face:
			box = largest_face["box"]
			current_emotions = largest_face["emotions"]
			update_emotion_statistics(current_emotions)

			# Store the emotion data
			emotion_statistics.append(current_emotions)
			emotion_type = max(current_emotions, key=current_emotions.get)
			emotion_score = current_emotions[emotion_type]

			emotion_text = f"{emotion_type}: {emotion_score:.2f}"
			emotion_text_placeholder.empty()  # Clear previous content (optional, as the next line overwrites it anyway)
			emotion_text_placeholder.write(emotion_text)
				
			# ax.clear()
			# ax.bar(emotion_labels, [current_emotions.get(emotion, 0) for emotion in emotion_labels], color='lightblue')
			# plt.ylim(0, 1)
			# plt.ylabel('Confidence')
			# plt.title('Real-time Emotion Detection')
			# ax.set_xticklabels(emotion_labels, rotation=45)
			# fig_place.pyplot(fig)

	overall_emotion, overall_score = get_overall_emotion()
	st.session_state.overall_emotion = overall_emotion
	st.session_state.overall_score = overall_score
	st.write(f"Overall emotion: {overall_emotion} ({overall_score:.2f})")		


		
	#webrtc_streamer(key="example", video_frame_callback=callback_model, rtc_configuration={"iceServers": token.ice_servers})

def video_frame_callback(frame):
	img = frame.to_ndarray(format="bgr24")
	with lock:
		img_container["img"] = img

	return frame
