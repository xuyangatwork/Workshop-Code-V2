import streamlit as st
from basecode.authenticate import return_api_key
from langchain.memory import ConversationBufferWindowMemory
from st_audiorec import st_audiorec
import os
import openai
import requests
import base64
import tempfile
import io
from openai import OpenAI
import streamlit_antd_components as sac
from nocode_workshop.k_map import map_creation_form, map_prompter_with_plantuml_form, generate_plantuml_mindmap, render_diagram
import requests

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

if "voice_image_file_exist" not in st.session_state:
	st.session_state.voice_image_file_exist = None

def clear_session_states():
	st.session_state.msg = []
	if "memory" not in st.session_state:
		pass
	else:
		del st.session_state["memory"]


def images_features():
	options = sac.chip(items=[
								sac.ChipItem(label='Image Generator', icon='image'),
								sac.ChipItem(label='Knowledge Map Generator', icon='diagram-3'),
								sac.ChipItem(label='Image analyser with chat', icon='clipboard-data'),
							], format_func='title', radius='sm', size='sm', align='left', variant='light')
	if options == 'Image Generator':
		st.subheader("Image Generator using DALL-E 3")
		generate_image()
	elif options == 'Knowledge Map Generator':
		st.subheader("Knowledge Map Generator using PlantUML")
		subject, topic, levels = map_creation_form()
		if subject and topic and levels:
			kb_prompt = map_prompter_with_plantuml_form(subject, topic, levels)
			if kb_prompt:
				with st.spinner("Generating knowledge map..."):
					kb_syntax = generate_plantuml_mindmap(kb_prompt)
					st.write(kb_syntax)
					st.image(render_diagram(kb_syntax))
	elif options == 'Image analyser with chat':
		st.subheader("Image analyser with chat")
		if st.toggle("Clear chat"):
			clear_session_states()
		visual_basebot_memory("VISUAL BOT")


def voice_features():
	options = sac.chip(items=[
								sac.ChipItem(label='Conversation Helper', icon='mic'),
								sac.ChipItem(label='Call Agent', icon='headset'),
							], format_func='title', radius='sm', size='sm', align='left', variant='light')
	if options == 'Conversation Helper':
		st.subheader("Conversation Helper")
		if st.toggle("Upload Audio"):
			transcript = upload_audio()
		else:
			transcript = record_myself()
		if transcript:
			st.write("Providing conversation feedback")
			with st.spinner("Constructing feedback"):
				analyse_audio(transcript)
				pass
	elif options == 'Call Agent':
		st.subheader("Call Agent")
		phone = st.text_input("Enter your phone number")
		st.write("Call Agent - work in progress")



def analyse_audio(prompt):
	prompt_design = """You are listening to the student's speech and you are giving feedback in the content and the way the sentences are structured to the student.
					provide feedback to the student on the following speech, tell the student what is good and what can be improved as well as provide guidance and pointers:"""
	if prompt_design and prompt:
		try:
			prompt = prompt_design + "\n" + prompt
			os.environ["OPENAI_API_KEY"] = return_api_key()
			# Generate response using OpenAI API
			response = client.chat.completions.create(
											model=st.session_state.openai_model, 
											messages=[{"role": "user", "content": prompt}],
											temperature=st.session_state.temp, #settings option
											presence_penalty=st.session_state.presence_penalty, #settings option
											frequency_penalty=st.session_state.frequency_penalty #settings option
											)
			if response.choices[0].message.content != None:
				st.write(response.choices[0].message.content)
	
		except Exception as e:
			st.error(e)
			st.error("Please type in a new topic or change the words of your topic again")
			return False


		

def analyse_image():
	st.subheader("Analyse an image")
	api_key = return_api_key()
	# Streamlit: File Uploader
	uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
	img_file_buffer = st.camera_input("Take a picture")
	prompt = st.text_input("Enter a prompt", value="This is a photo of a")
	if st.button("Analyse"):
		if uploaded_file is not None or img_file_buffer is not None:
			# Save the file to a temporary file
			if img_file_buffer is not None:
				uploaded_file = img_file_buffer
			extension = get_file_extension(uploaded_file.name)
			with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
				temp_file.write(uploaded_file.getvalue())
				temp_file_path = temp_file.name

			# Encode the image
			base64_image = encode_image(temp_file_path)

			# Prepare the payload
			headers = {
				"Content-Type": "application/json",
				"Authorization": f"Bearer {api_key}"
			}

			payload = {
				"model": "gpt-4-vision-preview",
				"messages": [
					{
						"role": "user",
						"content": [
							{
								"type": "text",
								"text": prompt
							},
							{
								"type": "image_url",
								"image_url": {
									"url": f"data:image/jpeg;base64,{base64_image}"
								}
							}
						]
					}
				],
				"max_tokens": 500
			}

			# Send the request
			response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

			# Display the response
			if response.status_code == 200:
				st.write(response.json())
				st.write(response.json()["choices"][0]["message"]["content"])
			else:
				st.error("Failed to get response")

			# Clean up the temporary file
			os.remove(temp_file_path)

def detect_file_upload():
	uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
	img_file_buffer = st.camera_input("Take a picture")
	if uploaded_file is not None or img_file_buffer is not None:
		# Save the file to a temporary file
		if img_file_buffer is not None:
			uploaded_file = img_file_buffer
		extension = get_file_extension(uploaded_file.name)
		with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
			temp_file.write(uploaded_file.getvalue())
			temp_file_path = temp_file.name
			st.session_state.voice_image_file_exist = temp_file_path
			st.write(st.session_state.voice_image_file_exist)
			return temp_file_path
	else:
		return False

def analyse_image_chat(temp_file_path, prompt):
	# Encode the image
	api_key = return_api_key()
	base64_image = encode_image(temp_file_path)

	# Prepare the payload
	headers = {
		"Content-Type": "application/json",
		"Authorization": f"Bearer {api_key}"
	}

	payload = {
		"model": "gpt-4-vision-preview",
		"messages": [
			{
				"role": "user",
				"content": [
					{
						"type": "text",
						"text": prompt
					},
					{
						"type": "image_url",
						"image_url": {
							"url": f"data:image/jpeg;base64,{base64_image}"
						}
					}
				]
			}
		],
		"max_tokens": 500
	}

	# Send the request
	response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

	# Display the response
	if response.status_code == 200:
		#st.write(response.json())
		#st.write(response.json()["choices"][0]["message"]["content"])
		os.remove(temp_file_path)
		return response.json()["choices"][0]["message"]["content"]
	else:
		os.remove(temp_file_path)
		st.session_state.voice_image_file_exist = None
		st.error("Failed to get response")
		return False

def transcribe_audio(file_path):
	with open(file_path, "rb") as audio_file:
		transcript = client.audio.transcriptions.create(
			model="whisper-1", 
			file=audio_file, 
			response_format="text"
		)
	return transcript

def translate_audio(file_path):
	with open(file_path, "rb") as audio_file:
		transcript = client.audio.translations.create(
		model="whisper-1", 
		file=audio_file
		)
		return transcript


def upload_audio():
	# Streamlit: File Uploader
	st.subheader("Transcribe an audio file")
	uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

	if uploaded_file is not None:
		# Save the file to a temporary file
		extension = os.path.splitext(uploaded_file.name)[-1]
		with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
			temp_file.write(uploaded_file.getvalue())
			temp_file_path = temp_file.name

		# Transcribe the audio
		if st.button("Transcribe"):
			with st.spinner("Transcribing..."):
				transcription_result = transcribe_audio(temp_file_path)
				st.write(transcription_result)

		# Clean up the temporary file
		os.remove(temp_file_path)

def record_myself():
	# Audio recorder
	st.subheader("Record and Transcribe an audio file")
	wav_audio_data = st_audiorec()

	if st.button("Transcribe (Maximum: 30 Seconds)") and wav_audio_data is not None:
		memory_file = io.BytesIO(wav_audio_data)
		memory_file.name = "recorded_audio.wav"

		with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
			tmpfile.write(wav_audio_data)

		with st.spinner("Transcribing..."):
			transcription_result = transcribe_audio(tmpfile.name)
			os.remove(tmpfile.name)  # Delete the temporary file manually after processing
			st.write(transcription_result)
			return transcription_result
	# elif st.button("Translation (Maximum: 30 Seconds)") and wav_audio_data is not None:
	# 	memory_file = io.BytesIO(wav_audio_data)
	# 	memory_file.name = "recorded_audio.wav"

	# 	with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
	# 		tmpfile.write(wav_audio_data)

	# 	with st.spinner("Translating..."):
	# 		transcription_result = translate_audio(tmpfile.name)
	# 		os.remove(tmpfile.name) 
	# 		st.write(transcription_result)
	# 		return transcription_result

def generate_image():
	st.subheader("Generate an image")
	i_prompt = st.text_input("Enter a prompt", value="Generate a photo of a")
	if st.button("Generate"):
		if i_prompt is not None or i_prompt != "Generate a photo of a":
			response = client.images.generate(
			model="dall-e-3",
			prompt=i_prompt,
			size="1024x1024",
			quality="standard",
			n=1,
			)

			image_url = response.data[0].url
			st.image(image_url)
		else:
			st.write("Please enter a prompt")


def text_speech(input_text):
	# Create a temporary file within the 'audio_files' directory
	temp_file = tempfile.NamedTemporaryFile(delete=False, dir=AUDIO_DIRECTORY, suffix='.mp3')
	
	# Generate speech
	response = client.audio.speech.create(
		model="tts-1",
		voice="alloy",
		input=input_text
	)

	# Write the response content to the temporary file
	with open(temp_file.name, 'wb') as file:
		file.write(response.content)

	# Return the path of the temporary file
	return temp_file.name


def text_to_speech():
	st.subheader("Text to Speech")
	if 'audio_file_path' not in st.session_state:
		st.session_state.audio_file_path = None

	user_input = st.text_area("Enter your text here:")

	if user_input and st.button("Generate Speech from Text"):
		st.session_state.audio_file_path = text_speech(user_input)
		st.audio(st.session_state.audio_file_path)

	if st.session_state.audio_file_path and st.button("Reset"):
		# Remove the temporary file
		os.remove(st.session_state.audio_file_path)
		st.session_state.audio_file_path = None
		st.experimental_rerun()



#below ------------------------------ base bot , K=2 memory for short term memory---------------------------------------------
#faster and more precise but no summary
def memory_buffer_component():
	if "memory" not in st.session_state:
		st.session_state.memory = ConversationBufferWindowMemory(k=st.session_state.k_memory)
	#st.write("Messages ", messages)
	mem = st.session_state.memory.load_memory_variables({})
	#For more customisation, this can be in the config.ini file
	prompt_template = st.session_state.chatbot + f""" 
						History of conversation:
						{mem}"""
				
	return prompt_template


#chat completion memory for streamlit using memory buffer
def chat_completion_memory(prompt):
	openai.api_key = return_api_key()
	os.environ["OPENAI_API_KEY"] = return_api_key()	
	prompt_template = memory_buffer_component()
	#st.write("Prompt Template ", prompt_template)
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

#integration API call into streamlit chat components with memory
def visual_basebot_memory(bot_name):
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
	left_col, right_col = st.columns(2)
	with left_col:
		
		for message in st.session_state.msg:
			with st.chat_message(message["role"]):
				st.markdown(message["content"])
	
	with right_col:
		detect_file_upload()
		#st.write(file_upload)
	try:
		if st.session_state.voice_image_file_exist != None:
			if prompt := st.chat_input("What is up?", key=1):
				with st.spinner("Analysing image..."):
					response = analyse_image_chat(st.session_state.voice_image_file_exist, prompt)
				st.session_state.msg.append({"role": "user", "content": prompt})
				with st.chat_message("user"):
					st.markdown(prompt)
				with st.chat_message("assistant"):
					message_placeholder = st.empty()
					message_placeholder.markdown(response)
					st.session_state.msg.append({"role": "assistant", "content": response})
					st.session_state["memory"].save_context({"input": prompt},{"output": response})	
					st.session_state.voice_image_file_exist = None

		elif prompt := st.chat_input("What is up?", key=2):
			st.session_state.msg.append({"role": "user", "content": prompt})
			with st.chat_message("user"):
				st.markdown(prompt)

			with st.chat_message("assistant"):
				message_placeholder = st.empty()
				full_response = ""
				for response in chat_completion_memory(prompt):
					full_response += (response.choices[0].delta.content or "")
					message_placeholder.markdown(full_response + "▌")
				message_placeholder.markdown(full_response)
		
			st.session_state.msg.append({"role": "assistant", "content": full_response})
			st.session_state["memory"].save_context({"input": prompt},{"output": full_response})
			
	except Exception as e:
		st.error(e)

def call_agent(phone):
	# Headers
	headers = {
	'Authorization': st.secrets["bland_key"],
	}

	# Data
	data = {
	'phone_number': phone,
	'task': """Call Flow:

		Introduce yourself as Mr Joe Tay and say you are calling from the YIJC Geography Department.

		Verify you are speaking with the parent or guardian of the student and mention the student's name.

		Explain the upcoming changes in the examination format for the geography paper, highlighting the new practical component.

		Inform them about the project work that the students are required to complete.

		Ask if they have any questions or concerns regarding these changes and offer to provide additional information or resources.

		Thank the parent or guardian for their time and provide contact information for further inquiries.

		Example Dialogue:

		R: Hello, this is Mr. Joe Tay calling from YIJC Geography Department. May I speak with the parent or guardian of Samantha Tan, please?

		P: Yes, this is her mother speaking.

		R: Great, thank you for taking my call. I wanted to discuss some important updates regarding the geography paper that Samantha will be taking this semester. We're introducing a practical component to the examination, which will account for 20% of the final grade.

		P: Oh, I see. What does that involve?

		R: Students will be asked to conduct a small field study and present their findings. We believe this hands-on experience will greatly benefit their understanding of geographic research methods. In addition to this, there is also a project component that the students will need to complete in groups.

		P: That sounds interesting. What kind of support will they receive for the project?

		R: Teachers will be providing guidance throughout the term, and we'll have a few dedicated sessions to discuss project ideas and execution. We're also providing extra materials on our online platform.

		P: Thank you for letting me know. I'll discuss this with Samantha tonight.

		R: You're welcome. If you or Samantha have any questions, please feel free to contact me. I'm here to help. Here's my email and the school's phone number.

		P: Got it. Thanks for the call.

		R: My pleasure. Have a wonderful day!

		This dialogue maintains the structure of providing important information and seeking confirmation while adapting the content to the educational context.""",
		'voice_id': 2,
		'reduce_latency': True,
		'request_data': {},
		'voice_settings':{
			speed: 1
		},
		'interruption_threshold': null
		}

		# API request 
	requests.post('https://api.bland.ai/call', json=data, headers=headers)
			