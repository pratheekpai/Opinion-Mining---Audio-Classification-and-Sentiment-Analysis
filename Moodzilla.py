import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk
from time import sleep

import customtkinter as ctk
import tkinter as tk
import sounddevice as sd

import numpy as np
import keyboard
import tensorflow
from tensorflow.keras.models import load_model
import cv2
from PIL import Image, ImageTk
import threading
import multiprocessing

from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation

import io
import sys
import os
import time
from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep

import whisper

import speech_recognition as sr
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import torch
import torch.nn.functional as F

import librosa

import gc
gc.enable()


# https://stackoverflow.com/questions/31836104/pyinstaller-and-onefile-how-to-include-an-image-in-the-exe-file
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS2
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


class VolumeMeter(ctk.CTkFrame):
    def __init__(self, parent, *args, **kwargs):
        ctk.CTkFrame.__init__(self, parent, *args, **kwargs)
        self.canvas = tk.Canvas(self, width=3, height=200, highlightthickness=0, borderwidth=0, relief='ridge')
        self.canvas.pack()
        self.meter = self.canvas.create_rectangle(0, 200, 3, 200, fill='#1FAB89')

    def update_meter(self, indata, frames, time, status):
        volume_norm = np.linalg.norm(indata)/20
        volume_norm *= 200
        self.canvas.coords(self.meter, 0, 200 - volume_norm, 3, 200)

class SlidePanelLeft(ctk.CTkFrame):
    def __init__(self, parent, start_pos, end_pos):
        super().__init__(master=parent)

        self.parent = parent

        # general attributes
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.width = 0.20

        # animation logic
        self.pos = start_pos
        self.btn_pos = 0
        # self.bg_frame_pos = 0.5
        # self.bg_frame_width = 0.9
        # self.bg_frame_height = 0.85
        self.in_start_pos = True

        # layout
        self.place(relx=self.start_pos, rely=0.03, relwidth=self.width, relheight=0.94)

    def animate(self):
        if self.in_start_pos:
            self.animate_forward()
        else:
            self.animate_backward()

    def animate_forward(self):
        if self.pos < self.end_pos:
            self.pos += 0.008
            self.btn_pos += 0.008
            
            # if self.bg_frame_width > 0.65:
            #     self.bg_frame_pos += 0.004 
            #     self.bg_frame_width -= 0.008
            #     self.bg_frame_height -= 0.008

            self.place(relx=self.pos, rely=0.03, relwidth=self.width, relheight=0.94)

            self.parent.slidepanel_left_btn.place(relx=self.btn_pos, rely=0.38, relwidth=0.03, relheight=0.2)
            # self.parent.bg_frame.place(relx=self.bg_frame_pos, rely=0.5, relwidth=self.bg_frame_width, relheight=self.bg_frame_height, anchor='center')
            
            self.after(10, self.animate_forward)
        else:
            self.in_start_pos = False
            img = ctk.CTkImage(dark_image=Image.open(resource_path('icons\\slide_left_icon.png')))
            self.parent.slidepanel_left_btn.configure(image=img)

    def animate_backward(self):
        if self.pos > self.start_pos:

            self.pos -= 0.008
            self.btn_pos -= 0.008

            # if self.bg_frame_width < 0.9:
            #     self.bg_frame_pos -= 0.004 
            #     self.bg_frame_width += 0.008
            #     self.bg_frame_height += 0.008    


            self.place(relx=self.pos, rely=0.03, relwidth=self.width, relheight=0.94)

            self.parent.slidepanel_left_btn.place(relx=self.btn_pos, rely=0.38, relwidth=0.03, relheight=0.2)
            # self.parent.bg_frame.place(relx=self.bg_frame_pos, rely=0.5, relwidth=self.bg_frame_width, relheight=self.bg_frame_height, anchor='center')
            

            self.after(10, self.animate_backward)
        else:
            self.in_start_pos = True
            img = ctk.CTkImage(dark_image=Image.open(resource_path('icons\\slide_right_icon.png')))
            self.parent.slidepanel_left_btn.configure(image=img)

class SlidePanelRight(ctk.CTkFrame):
    def __init__(self, parent, start_pos, end_pos):
        super().__init__(master=parent)

        self.parent = parent

        # general attributes
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.width = 0.25

        # animation logic
        self.pos = start_pos
        self.btn_pos = 0.97
        # self.bg_frame_pos = 0.5
        # self.bg_frame_width = 0.9
        # self.bg_frame_height = 0.85
        self.in_start_pos = True

        # layout
        self.place(relx=self.start_pos, rely=0.03, relwidth=self.width, relheight=0.94)

    def animate(self):
        if self.in_start_pos:
            self.animate_forward()
            self.parent.slidepanel_right.lift()
        else:
            self.animate_backward()

    def animate_forward(self):
        if self.pos > self.end_pos:
            self.pos -= 0.008
            self.btn_pos -= 0.008
            
            # if self.bg_frame_width > 0.65:
            #     self.bg_frame_pos += 0.004 
            #     self.bg_frame_width -= 0.008
            #     self.bg_frame_height -= 0.008

            self.place(relx=self.pos, rely=0.03, relwidth=self.width, relheight=0.94)

            self.parent.slidepanel_right_btn.place(relx=self.btn_pos, rely=0.38, relwidth=0.03, relheight=0.2)
            # self.parent.bg_frame.place(relx=self.bg_frame_pos, rely=0.5, relwidth=self.bg_frame_width, relheight=self.bg_frame_height, anchor='center')
            
            self.after(10, self.animate_forward)
        else:
            self.in_start_pos = False
            img = ctk.CTkImage(dark_image=Image.open(resource_path('icons\\slide_right_icon2.png')))
            self.parent.slidepanel_right_btn.configure(image=img)

    def animate_backward(self):
        if self.pos < self.start_pos:

            self.pos += 0.008
            self.btn_pos += 0.008

            # if self.bg_frame_width < 0.9:
            #     self.bg_frame_pos -= 0.004 
            #     self.bg_frame_width += 0.008
            #     self.bg_frame_height += 0.008    


            self.place(relx=self.pos, rely=0.03, relwidth=self.width, relheight=0.94)

            self.parent.slidepanel_right_btn.place(relx=self.btn_pos, rely=0.38, relwidth=0.03, relheight=0.2)
            # self.parent.bg_frame.place(relx=self.bg_frame_pos, rely=0.5, relwidth=self.bg_frame_width, relheight=self.bg_frame_height, anchor='center')
            

            self.after(10, self.animate_backward)
        else:
            self.in_start_pos = True
            img = ctk.CTkImage(dark_image=Image.open(resource_path('icons\\slide_left_icon2.png')))
            self.parent.slidepanel_right_btn.configure(image=img)


def capture_speech_feed(queue, speech_queue, text_emotion_queue, tone_audio_queue, model_ready_queue, energy_threshold, whisper_model, clearcontext):
    startstop_speech_flag = queue.get()
    if startstop_speech_flag == True:


        # Loading Models
        config = AutoConfig.from_pretrained(resource_path('models\\config.json'))
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        text_model = AutoModelForSequenceClassification.from_pretrained(resource_path("models\\pytorch_model.bin"), config=config)


        audio_model = whisper.load_model(whisper_model)

        print('Models loaded')
        # The last time a recording was retreived from the queue.
        phrase_time = None
        # Current raw audio bytes.
        last_sample = bytes()
        # Thread safe Queue for passing data from the threaded recording callback.
        data_queue = Queue()
        recorder = sr.Recognizer()
        # We use SpeechRecognizer to record our audio because it has a nice feauture where it can detect when speech ends.            
        recorder.energy_threshold = energy_threshold
        # Definitely do this, dynamic energy compensation lowers the energy threshold dramtically to a point where the SpeechRecognizer never stops recording.
        recorder.dynamic_energy_threshold = False

        record_timeout = 2
        phrase_timeout = 3

        temp_file = NamedTemporaryFile().name
        transcription = ['']


        def record_callback(_, audio:sr.AudioData) -> None:
            """
            Threaded callback function to recieve audio data when recordings finish.
            audio: An AudioData containing the recorded bytes.
            """
            tone_audio_queue.put(audio)
            # Grab the raw bytes and push it into the thread safe queue.

            data = audio.get_raw_data()
            data_queue.put(data)

        # open the microphone and start recording
        with sr.Microphone() as source:
            recorder.adjust_for_ambient_noise(source)
        stop_listening = recorder.listen_in_background(source, record_callback, phrase_time_limit=2)
        # speech_text_label.configure(text='You can speak now...')


        model_ready_queue.put(True)

        print('Speak')

        while True:
            if queue.empty() == False:
                startstop_speech_flag = queue.get()

            if startstop_speech_flag == True:
                now = datetime.utcnow()
                # Pull raw recorded audio from the queue.
                if not data_queue.empty():
                    phrase_complete = False
                    # If enough time has passed between recordings, consider the phrase complete.
                    # Clear the current working audio buffer to start over with the new data.
                    if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                        last_sample = bytes()
                        phrase_complete = True
                    # This is the last time we received new audio data from the queue.
                    phrase_time = now

                    # Concatenate our current audio data with the latest audio data.
                    while not data_queue.empty():
                        data = data_queue.get()
                        last_sample += data

                    # print('Started: ', datetime.utcnow().time())

                    # Use AudioData to convert the raw data to wav data.
                    audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                    wav_data = io.BytesIO(audio_data.get_wav_data())

                    # Write wav data to the temporary file as bytes.
                    with open(temp_file, 'w+b') as f:
                        f.write(wav_data.read())

                    # Read the transcription.
                    result = audio_model.transcribe(temp_file, fp16=torch.cuda.is_available())
                    text = result['text'].strip()

                    # print('Transcription End: ', datetime.utcnow().time())

                    # If we detected a pause between recordings, add a new item to our transcripion.
                    # Otherwise edit the existing one.

                    if clearcontext.empty() == False:
                        clearcontext.get()
                        transcription = []

                    if phrase_complete:
                        transcription.append(text)
                    else:
                        transcription[-1] = text

                    # Clear the text label to reprint the updated transcription.
                    complete_text = ''

                    for line in transcription:
                        if line == 'Thank you.' or line == ' Thank You.' or line == ' See you next time!' or line == 'See you next time!'or line == 'Thank You.':
                            continue 
                        complete_text += line + " "
                    

                    if len(complete_text) > 2500:                
                        transcription = []

                    while not speech_queue.empty():
                        speech_queue.get()
                    speech_queue.put(complete_text)

                    def preprocess_sentence(sentence):
                        return tokenizer(sentence, padding="max_length", truncation=True, return_tensors="pt")

                    preprocessed_input = preprocess_sentence(complete_text)

                    with torch.no_grad():
                        logits = text_model(**preprocessed_input)[0]
                        probs = F.softmax(logits, dim=1)

                    text_prediction = np.array(probs)[0].tolist()


                    while not text_emotion_queue.empty():
                        text_emotion_queue.get()
                    text_emotion_queue.put(text_prediction)
                    # print(self.speech_text)


            else:
                stop_listening()
                break


def tone_predict(tone_emotion_queue, tone_audio_queue, tone_predict_close):
    tone_model = load_model(resource_path('models\\speech_emotion_model_i2.h5'))
    while True:
        if tone_predict_close.empty() == False:
            tone_predict_close.get()
            break
        if tone_audio_queue.empty() == False:

            audio = tone_audio_queue.get()
            sample_rate = 44100

            # Define the spectrogram parameters
            n_fft = 2048
            hop_length = 512
            n_mels = 128

            # Convert the recorded audio to a numpy array
            tone_audio = np.frombuffer(audio.frame_data, dtype=np.float16)
            tone_audio = np.nan_to_num(tone_audio, nan=0.0, posinf=0.0, neginf=0.0)


            # Convert the audio to spectrogram
            spectrogram = librosa.feature.melspectrogram(y=tone_audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)


            spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

            # Preprocess the audio data
            spectrogram = spectrogram[..., np.newaxis] # Add a new axis for the channel dimension
            spectrogram = tensorflow.image.resize(spectrogram, (128, 128))

            probs = tone_model.predict(spectrogram[np.newaxis, ...], verbose=0)[0].tolist()

            while not tone_emotion_queue.empty():
                tone_emotion_queue.get()
            tone_emotion_queue.put(probs)



class Moodzilla(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # title
        self.title('Moodzilla')
        
        # ico
        self.iconbitmap(resource_path("icons\\Moodzilla.ico"))

        # app geometry
        self.screen_width = self.winfo_screenwidth()
        self.screen_height = self.winfo_screenheight()

        self.width = 1200
        self.height = 600

        self.left = int((self.screen_width/2) - (self.width/2))
        self.top = int((self.screen_height/2) - (self.height/2))

        self.geometry(f"{self.width}x{self.height}+{self.left-10}+{self.top-32}")
        
        # background frame
        self.bg_frame = ctk.CTkFrame(master = self)
        self.bg_frame.place(relx=0.5, rely=0.5, relwidth=0.9, relheight=0.85, anchor='center')        

        # slide panels

        # left panel
        self.slidepanel_left = SlidePanelLeft(self, -0.2, 0.001)
        self.slide_right_image = ctk.CTkImage(dark_image=Image.open(resource_path("icons\\slide_right_icon.png")))
        self.slidepanel_left_btn = ctk.CTkButton(self, text='', fg_color='transparent', image=self.slide_right_image, command=self.slidepanel_left.animate, hover=False)
        self.slidepanel_left_btn.place(relx=0, rely=0.38, relwidth=0.03, relheight=0.2)


        # left panel widgets
        self.left_panel_img = ctk.CTkImage(dark_image=Image.open(resource_path("icons\\Moodzilla.png")), size=(200, 200))
        self.left_panel_imglabel = ctk.CTkLabel(self.slidepanel_left, text='', image=self.left_panel_img)
        self.left_panel_imglabel.place(relx=0.5, rely=0.2, anchor='center')

        self.left_panel_titlelabel = ctk.CTkLabel(self.slidepanel_left, text='Moodzilla', font=('Open Sans', 30), text_color='#FFF')
        self.left_panel_titlelabel.place(relx=0.5, rely=0.4, anchor='center')

        self.appearance_mode_label = ctk.CTkLabel(self.slidepanel_left, text="Appearance Mode:", anchor="center")
        self.appearance_mode_label.place(relx=0.5, rely=0.7, anchor='center')
        self.appearance_mode_optionemenu = ctk.CTkOptionMenu(self.slidepanel_left, values=["System", "Light", "Dark"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.place(relx=0.5, rely=0.75, anchor='center')

        # switch cameras
        self.chosen_cam = 0
        self.avail_cam_label = ctk.CTkLabel(self.slidepanel_left, text="Switch Camera:", anchor="center")
        self.avail_cam_label.place(relx=0.5, rely=0.83, anchor='center')
        self.avail_cam_btn = ctk.CTkButton(self.slidepanel_left, text='Check For Available Cameras', fg_color='#850E35', hover=True, hover_color='#EE6983', anchor='center', command=self.get_available_cameras)
        self.avail_cam_btn.place(relx=0.5, rely=0.88, anchor='center')
        self.avail_cam_optionemenu = ctk.CTkOptionMenu(self.slidepanel_left, values=['Camera 0'], fg_color='#222831', button_color='#30475E', button_hover_color='#F05454', dropdown_fg_color='#14274E', dropdown_hover_color='#155263',
                                                                       command=self.change_camera, state='disabled')
        self.avail_cam_optionemenu.place(relx=0.5, rely=0.95, anchor='center')

        # right panel
        self.slidepanel_right = SlidePanelRight(self, 1, 0.748)
        self.slide_left_image = ctk.CTkImage(dark_image=Image.open(resource_path("icons\\slide_left_icon2.png")))
        self.slidepanel_right_btn = ctk.CTkButton(self, text='', fg_color='transparent', image=self.slide_left_image, command=self.slidepanel_right.animate, hover=False)
        self.slidepanel_right_btn.place(relx=0.97, rely=0.38, relwidth=0.03, relheight=0.2)

        self.speech_text_label_title_frame = ctk.CTkFrame(self.slidepanel_right, border_color='#FFF', border_width=2)
        self.speech_text_label_title_frame.place(relx=0.5, rely=0.05, relwidth=0.85, relheight=0.05, anchor='center')
        self.speech_text_label_title = ctk.CTkLabel(self.speech_text_label_title_frame, text='Speech Text', corner_radius=5, font=('Sans Serif', 15), fg_color='#0D1117', text_color='#FFF', anchor='center')
        self.speech_text_label_title.place(relx=0.01, rely=0.065, relwidth=0.978, relheight=0.85)

        self.speech_textbox = ctk.CTkTextbox(self.slidepanel_right, text_color='#FFF', font=('Sans Serif', 12), border_color='#FFF', border_width=2, fg_color='#171B23', corner_radius=3, wrap='word', state='disabled')
        self.speech_textbox.place(relx=0.5, rely=0.33, relwidth=0.85, relheight=0.5, anchor='center')

        self.speech_textbox_clr_btn = ctk.CTkButton(self.slidepanel_right, text='Clear Textbox', border_width=2, border_color='#FFF', fg_color='#E23E57', font=('Sans Serif', 15), text_color='#FFF', command=self.speech_textbox_clr)
        self.speech_textbox_clr_btn.place(relx=0.5, rely=0.62, anchor='center')


        self.options_frame = ctk.CTkFrame(self.slidepanel_right, fg_color='#222831', border_color='#FFF', border_width=2)
        self.options_frame.place(relx=0.5, rely=0.83, relwidth=0.85, relheight=0.3, anchor='center')
        self.model_label = ctk.CTkLabel(self.options_frame, text="Choose Model :", anchor="center", font=('Sans Serif', 15))
        self.model_label.place(relx=0.5, rely=0.15, anchor='center')
        self.model_optionmenu = ctk.CTkOptionMenu(self.options_frame, values=["Tiny", "Base", "Small", "Medium", "Large"], text_color='#181823', dropdown_text_color='#FFF', fg_color='#CBE4DE', button_color='#0E8388', button_hover_color='#2E4F4F', dropdown_fg_color='#171B23', dropdown_hover_color='#0B2447',
                                                                       command=self.change_model)
        self.model_optionmenu.place(relx=0.5, rely=0.32, anchor='center')

        self.energy_threshold = 1000
        self.energy_label = ctk.CTkLabel(self.options_frame, text='Energy Threshold : ', font=('Sans Serif', 15))
        self.energy_label.place(relx=0.38, rely=0.6, anchor='center')
        self.energy_label_value = ctk.CTkLabel(self.options_frame, text='1000', font=('Sans Serif', 17), text_color='#FFF', fg_color='#171B23', corner_radius=5, anchor='center')
        self.energy_label_value.place(relx=0.73, rely=0.6, relwidth=0.2, relheight=0.15, anchor='center')
        self.energyvar = tk.IntVar(self, value=1000)
        self.energy_slide = ctk.CTkSlider(self.options_frame, from_=0, to=4000, command=self.change_energy_threshold, variable=self.energyvar, number_of_steps=400, progress_color='#F38181', button_color='#F4EEFF', button_hover_color='#DCD6F7')
        self.energy_slide.place(relx=0.5, rely=0.78, relwidth=0.7, anchor='center')


        self.info_image = ctk.CTkImage(dark_image=Image.open(resource_path("icons\\info.png")), size=(20, 20))
        self.info_button = ctk.CTkButton(self.options_frame, text='', fg_color='transparent', image=self.info_image, hover=False, anchor='center', state='disabled')
        self.info_button.place(relx=0.9, rely=0.12, relwidth=0.15, anchor='center')
        self.info_button.bind('<Enter>', self.info_print)
        self.info_button.bind('<Leave>', self.info_print_leave)


        # Loading Models
        self.facial_model = load_model(resource_path('models\\fer2013n_model_i2.h5'))


        # Create the video feed display
        self.video_frame = ctk.CTkFrame(self.bg_frame, border_color='#F1F6F9', border_width=4, fg_color='#F1F6F9')
        self.video_frame.place(relx=0.065, rely=0.08, relwidth=0.4, relheight=0.4)
        self.video_label = ctk.CTkLabel(self.video_frame, text='')
        self.video_label.place(relx=0.01, rely=0.018, relwidth=0.978, relheight=0.965, anchor='nw')
        self.video_width = int(self.screen_width * 0.9 * 0.4)
        self.video_height = int(self.screen_height * 0.85 * 0.45)
        
        # Create the face not detected text display
        self.face_text_frame = ctk.CTkFrame(self.bg_frame, border_color='#19bd97', border_width=3, fg_color='transparent')
        self.face_text_frame.place(relx=0.53, rely=0.02, relwidth=0.4, relheight=0.1)
        self.face_text_label = ctk.CTkLabel(self.face_text_frame, text='', font=("Helvetica", 15), text_color='#FFF', fg_color='transparent')
        self.face_text_label.place(relx=0.5, rely=0.35, anchor='center')
        self.face_text_frame.lift()

        # Create the face sentiment text display
        self.face_senti_frame = ctk.CTkFrame(self.bg_frame, border_color='#19bd97', border_width=3, fg_color='transparent')
        self.face_senti_frame.place(relx=0.065, rely=0.02, relwidth=0.4, relheight=0.1)
        self.face_senti_label = ctk.CTkLabel(self.face_senti_frame, text='', font=("Helvetica", 15), text_color='#FFF', fg_color='transparent')
        self.face_senti_label.place(relx=0.5, rely=0.35, anchor='center')
        self.video_frame.lift()


        # Facial Sentiment Module

        self.face_flag = False
        self.facial_prediction = []

        
        # button to start/stop the video capture thread
        self.startstop_face_flag = False
        self.capture_video_thread = None
        self.startstop_face_btn = ctk.CTkButton(self, fg_color='#E76161', text_color='#FFF', corner_radius=5, border_width=2, border_color='#98D8AA', text='Start Face Recording', hover=True, hover_color='#E84545', anchor='center', command=self.start_stop_face_recording)
        self.startstop_face_btn.place(relx=0.21, rely=0.9375, relwidth=0.15, relheight=0.05)


        self.face_fig = Figure(figsize=(3, 2), dpi=100)
        self.face_ax = self.face_fig.add_subplot(111)
        self.face_canvas = FigureCanvasTkAgg(self.face_fig, master=self.bg_frame)
        self.face_canvas.get_tk_widget().place(relx=0.53, rely=0.08, relwidth=0.4, relheight=0.4)
        self.face_anim = FuncAnimation(self.face_fig, self.face_sentiment_plot, 1000)



        # Speech Sentiment Module

        self.queue = multiprocessing.Queue() 
        self.speech_queue = multiprocessing.Queue()
        self.audio_queue = multiprocessing.Queue()
        self.model_ready_queue = multiprocessing.Queue()
        self.text_emotion_queue = multiprocessing.Queue()
        self.tone_emotion_queue = multiprocessing.Queue()
        self.tone_audio_queue = multiprocessing.Queue()
        self.tone_predict_close = multiprocessing.Queue()
        self.clearcontext = multiprocessing.Queue()

        self.whisper_model = "tiny"
        self.audio = None
        self.source = None
        self.stop_listening = None
        self.text_flag = False
        self.text_prediction = []

        # button to start/stop the speech capture thread
        self.startstop_speech_flag = False
        self.capture_speech_process = None

        # initialize the recognizer
        self.recorder = sr.Recognizer()
        self.speech_text = ''
        self.preprocessed_input = None

        # Create the Loading Model text display
        self.speech_text_frame = ctk.CTkFrame(self.bg_frame, border_color='#FF6000', border_width=3, fg_color='transparent')
        self.speech_text_frame.place(relx=0.5, rely=0.548, relwidth=0.3, relheight=0.08, anchor='center')
        self.speech_text_label = ctk.CTkLabel(self.speech_text_frame, text='', font=("Helvetica", 15), text_color='#FFF', fg_color='transparent')
        self.speech_text_label.place(relx=0.5, rely=0.5, relwidth=0.978, relheight=0.80, anchor='center')
        self.speech_text_frame.lift()


        self.startstop_speech_btn = ctk.CTkButton(self, fg_color='#E76161', text_color='#FFF', corner_radius=5, border_width=2, border_color='#98D8AA', text='Start Speech Recording', hover=True, hover_color='#E84545', anchor="center", command=self.start_stop_speech_recording)
        self.startstop_speech_btn.place(relx=0.635, rely=0.9375, relwidth=0.15, relheight=0.05)


        self.text_fig = Figure(figsize=(3, 2), dpi=100)
        self.text_ax = self.text_fig.add_subplot(111)
        self.text_canvas = FigureCanvasTkAgg(self.text_fig, master=self.bg_frame)
        self.text_canvas.get_tk_widget().place(relx=0.53, rely=0.596, relwidth=0.4, relheight=0.4)
        self.text_anim = FuncAnimation(self.text_fig, self.text_sentiment_plot, 2000)

        self.tone_fig = Figure(figsize=(3, 2), dpi=100)
        self.tone_ax = self.tone_fig.add_subplot(111)
        self.tone_canvas = FigureCanvasTkAgg(self.tone_fig, master=self.bg_frame)
        self.tone_canvas.get_tk_widget().place(relx=0.065, rely=0.596, relwidth=0.4, relheight=0.4)



    # Function to capture the video feed
    def capture_video_feed(self):

        if self.startstop_face_flag == True:
            cap = cv2.VideoCapture(self.chosen_cam)
            
            while True:
                ret, frame = cap.read()
                if ret:
                    # Resize and convert the frame to PIL format
                    frame = cv2.resize(frame, (self.video_width, self.video_height))
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(image)
                    photo = ctk.CTkImage(light_image=image, size=(self.video_width, self.video_height))
                    
                        
                    self.video_label.configure(image=photo)

                    image.close()
                    del photo

                    # Perform facial sentiment analysis on the frame
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = cv2.CascadeClassifier(resource_path('classifiers\\haarcascade_frontalface_default.xml')).detectMultiScale(gray, 1.1, 8)
                    if(len(faces) == 0):
                        self.face_text_label.configure(text='No Face Detected')
                    else:
                        self.face_text_label.configure(text='')
                  
                    for (x,y,w,h) in faces:
                        # print('yes')
                        face_img = gray[y:y+h, x:x+w]
                        face_img = cv2.resize(face_img, (48, 48))
                        face_img = face_img.reshape(1, 48, 48, 1)
                        face_img = face_img.astype('float32')
                        face_img /= 255 
                        
                        self.facial_prediction = self.facial_model.predict(face_img, verbose=0)[0].tolist()
                        self.face_flag = True    

                    sentiment = ''
                    facial_emotions = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear"]
                    for i, prob in enumerate(self.facial_prediction):
                        if prob*100 > 10.0:
                            sentiment += facial_emotions[i] + ' '

                    self.face_senti_label.configure(text=sentiment)

                if self.startstop_face_flag == False:
                    self.face_senti_label.configure(text='')
                    break  

    # Function to plot the facial sentiment bar chart                    
    def face_sentiment_plot(self, i):
        # plt.close(self.face_fig)
        self.face_ax.clear()

        #  Only when the flag is true, it mean that we have gotten the prediction from the capture_video_feed()
        if self.face_flag == True:
            emotion_probs = [prob*100 for prob in self.facial_prediction]

            # Set bar color based on emotion
            colors = ['#FF5733', '#6A5ACD', '#800000', '#FFFF00', '#A9A9A9', '#1E90FF', '#FFA07A']
            facial_emotions = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear"]
                        
            # Create horizontal bar chart
            
            y_pos = np.arange(len(facial_emotions))
            self.face_ax.barh(y_pos, emotion_probs, align='center', color=colors)
            self.face_ax.set_yticks(y_pos)
            self.face_ax.set_yticklabels(facial_emotions, fontsize=6)
            self.face_ax.invert_yaxis()  # labels read top-to-bottom
            self.face_ax.set_xlabel('Probability')
            self.face_ax.set_title('Facial Sentiment Probabilities')


            # Add probability values as labels above each bar
            for i, prob in enumerate(emotion_probs):
                self.face_ax.text(prob+1, i, str(round(prob, 3)), color='black', va='center', fontsize=8)

            # Set the minimum and maximum limits of the x-axis
            self.face_ax.set_xlim(0, 100)
            self.face_ax.set_ylim(-0.5, len(facial_emotions) - 0.5)

            # plt.show()


            # Remove the old canvas, redraw the plot and create a new canvas
            # if self.face_canvas is not None:
            #     self.face_canvas.get_tk_widget().destroy()
            self.face_canvas.draw()

        return self.face_canvas

    # Function to set the speech record button flag
    def start_stop_face_recording(self):
        if self.startstop_face_flag == True:
            self.startstop_face_flag = False
            self.startstop_face_btn.configure(text='Start Video Recording')
            

            # giving timeout as a parameter is crucial to let the thread complete out its work and then close gracefully
            self.capture_video_thread.join(timeout=1)
            sleep(1)

            self.face_ax.clear()
            self.face_anim.event_source.stop()
            self.face_flag = False
            self.video_label.configure(image=None)
            
            self.face_text_label.configure(text='')
            self.face_senti_label.configure(text='')

            self.face_fig = Figure(figsize=(3, 2), dpi=100)
            self.face_ax = self.face_fig.add_subplot(111)
            self.face_canvas = FigureCanvasTkAgg(self.face_fig, master=self.bg_frame)
            self.face_canvas.get_tk_widget().place(relx=0.53, rely=0.08, relwidth=0.4, relheight=0.4)
            

        else:
            self.startstop_face_flag = True
            self.startstop_face_btn.configure(text='Stop Video Recording')
            self.face_anim.event_source.start()

            # Set up the video feed capture thread
            self.capture_video_thread = threading.Thread(target=self.capture_video_feed)
            self.capture_video_thread.start()

    # Function to plot the text sentiment bar chart
    def text_sentiment_plot(self, i):
    
        #  Only when the flag is true, it mean that we have gotten the prediction from the capture_video_feed()
        if self.text_emotion_queue.empty() == False:
            self.text_ax.clear()
        
            text = self.speech_queue.get()
            self.speech_textbox.configure(state='normal')
            self.speech_textbox.delete("0.0", "end")
            self.speech_textbox.insert("0.0", text)
            self.speech_textbox.configure(state='disabled')

            text_prediction = self.text_emotion_queue.get()
            text_emotions = ['sadness', 'happiness', 'love', 'anger', 'fear', 'surprise']
            
            emotion_probs = [prob*100 for prob in text_prediction]

            # Set bar color based on emotion
            colors = ['#FF5733', '#6A5ACD', '#800000', '#FFFF00', '#A9A9A9', '#1E90FF']

                        
            # Create horizontal bar chart
            
            y_pos = np.arange(len(text_emotions))
            self.text_ax.barh(y_pos, emotion_probs, align='center', color=colors)
            self.text_ax.set_yticks(y_pos)
            self.text_ax.set_yticklabels(text_emotions, fontsize=6)
            self.text_ax.invert_yaxis()  # labels read top-to-bottom
            self.text_ax.set_xlabel('Probability')
            self.text_ax.set_title('Textual Sentiment Probabilities')
   

            # Add probability values as labels above each bar
            for i, prob in enumerate(emotion_probs):
                self.text_ax.text(prob+1, i, str(round(prob, 3)), color='black', va='center', fontsize=8)

            # Set the minimum and maximum limits of the x-axis
            self.text_ax.set_xlim(0, 100)
            self.text_ax.set_ylim(-0.5, len(text_emotions) - 0.5)

            # plt.show()


            # Remove the old canvas, redraw the plot and create a new canvas
            # if self.face_canvas is not None:
            #     self.face_canvas.get_tk_widget().destroy()
            self.text_canvas.draw()

        return self.text_canvas

        # Function to plot the tone sentiment bar chart


    def tone_sentiment_plot(self, i):

        if self.tone_emotion_queue.empty() == False:

            self.tone_ax.clear()

            probs = self.tone_emotion_queue.get()

            # Convert the predicted label to the corresponding emotion category
            tone_emotions = ['neutral', 'calm', 'happiness', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

            emotion_probs = [prob*100 for prob in probs]

            # Set bar color based on emotion
            colors = ['#FF5733', '#6A5ACD', '#800000', '#FFFF00', '#A9A9A9', '#1E90FF', '#903749', '#815B5B']

                        
            # Create horizontal bar chart        
            y_pos = np.arange(len(tone_emotions))
            self.tone_ax.barh(y_pos, emotion_probs, align='center', color=colors)
            self.tone_ax.set_yticks(y_pos)
            self.tone_ax.set_yticklabels(tone_emotions, fontsize=6)
            self.tone_ax.invert_yaxis()  # labels read top-to-bottom
            self.tone_ax.set_xlabel('Probability')
            self.tone_ax.set_title('Tonal Sentiment Probabilities')


            # Add probability values as labels above each bar
            for i, prob in enumerate(emotion_probs):
                self.tone_ax.text(prob+1, i, str(round(prob, 3)), color='black', va='center', fontsize=8)

            # Set the minimum and maximum limits of the x-axis
            self.tone_ax.set_xlim(0, 100)
            self.tone_ax.set_ylim(-0.5, len(tone_emotions) - 0.5)


            self.tone_canvas.draw()

        return self.tone_canvas

    # Function to set the speech record button flag
    def start_stop_speech_recording(self):
        
        if self.startstop_speech_flag == True:
            self.startstop_speech_flag = False

            self.queue.put(self.startstop_speech_flag)
            self.startstop_speech_btn.configure(text='Start Speech Recording')
            self.speech_text_label.configure(text='')

            # giving timeout as a parameter is crucial to let the thread complete out its work and then close gracefully
            self.capture_speech_process.join(timeout=1)
            self.capture_speech_process.terminate()


            self.tone_predict_close.put(True)
            self.tone_predict_process.join(timeout=1)
            self.tone_predict_process.terminate()
            self.loading_thread.join()


            self.tone_plot_thread.join(timeout=1)


            self.text_anim.event_source.stop()
            self.tone_anim.event_source.stop()
            self.text_ax.clear()
            self.tone_ax.clear()

            self.speech_text_label.configure(text='')
            self.speech_textbox.configure(state='normal')
            self.speech_textbox.delete("0.0", "end")
            self.speech_textbox.configure(state='disabled')

            self.text_fig = Figure(figsize=(3, 2), dpi=100)
            self.text_ax = self.text_fig.add_subplot(111)
            self.text_canvas = FigureCanvasTkAgg(self.text_fig, master=self.bg_frame)
            self.text_canvas.get_tk_widget().place(relx=0.53, rely=0.596, relwidth=0.4, relheight=0.4)

            self.tone_fig = Figure(figsize=(3, 2), dpi=100)
            self.tone_ax = self.tone_fig.add_subplot(111)
            self.tone_canvas = FigureCanvasTkAgg(self.tone_fig, master=self.bg_frame)
            self.tone_canvas.get_tk_widget().place(relx=0.065, rely=0.596, relwidth=0.4, relheight=0.4)


            while not self.tone_emotion_queue.empty():
                self.tone_emotion_queue.get()

            while not self.tone_audio_queue.empty():
                self.tone_audio_queue.get()

            while not self.model_ready_queue.empty():
                self.model_ready_queue.get()

            while not self.queue.empty():
                self.queue.get()

        else:
            self.startstop_speech_flag = True
            self.startstop_speech_btn.configure(text='Stop Speech Recording')


            self.loading_thread = threading.Thread(target=self.loading_speech)
            self.loading_thread.start()
            


            
            self.queue.put(self.startstop_speech_flag)  
            self.capture_speech_process = multiprocessing.Process(target=capture_speech_feed, args=(self.queue, self.speech_queue, self.text_emotion_queue, self.tone_audio_queue, self.model_ready_queue, self.energy_threshold, self.whisper_model, self.clearcontext))
            self.capture_speech_process.start()

            self.tone_predict_process = multiprocessing.Process(target=tone_predict, args=(self.tone_emotion_queue, self.tone_audio_queue, self.tone_predict_close))
            self.tone_predict_process.start()


            self.tone_plot_thread = threading.Thread(target=self.tone_plotting)
            self.tone_plot_thread.start()


    def change_model(self, model_name: str):
        self.whisper_model = whisper.load_model(model_name)
        self.speech_text_label.configure(text='New Speech Model Loaded...')

    def change_energy_threshold(self, value):
        self.energy_label_value.configure(text=str(int(value)))

    def speech_textbox_clr(self):
        self.speech_textbox.configure(state='normal')
        self.speech_textbox.delete("0.0", "end")
        self.clearcontext.put(True)
        self.speech_textbox.configure(state='disabled')

    def loading_speech(self):
        self.speech_text_label.configure(text='Loading Model...')
        while self.model_ready_queue.empty():
            sleep(3)
            continue
        self.speech_text_label.configure(text='You can speak now...')

    # Function to switch cameras if available
    def get_available_cameras(self):
        index = 0
        arr = []
        while True:
            try:
                cap = cv2.VideoCapture(index)
                if not cap.read()[0]:
                    break
                else:
                    arr.append(f'Camera {index}')
                cap.release()
                index += 1
            except:
                break
        self.avail_cam_optionemenu.configure(state='normal', values=arr)

    def change_camera(self, new_camera_index: str):
        self.chosen_cam = new_camera_index.split()[1]

    def change_appearance_mode_event(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)

    def info_print(self, event):
        self.info_ = ctk.CTkLabel(self.slidepanel_right, text="Change options before 'Start Speech Recording'", text_color='#FFF', font=('Helvetica', 11))
        self.info_.place(relx=0.5, rely=0.67, anchor='center')

    def info_print_leave(self, event):
        self.info_.configure(text='')

    def tone_plotting(self):
        self.tone_fig = Figure(figsize=(3, 2), dpi=100)
        self.tone_ax = self.tone_fig.add_subplot(111)
        self.tone_canvas = FigureCanvasTkAgg(self.tone_fig, master=self.bg_frame)
        self.tone_canvas.get_tk_widget().place(relx=0.065, rely=0.596, relwidth=0.4, relheight=0.4)
        self.tone_anim = FuncAnimation(self.tone_fig, self.tone_sentiment_plot, 1000)
        self.tone_anim.event_source.start()

        self.text_fig = Figure(figsize=(3, 2), dpi=100)
        self.text_ax = self.text_fig.add_subplot(111)
        self.text_canvas = FigureCanvasTkAgg(self.text_fig, master=self.bg_frame)
        self.text_canvas.get_tk_widget().place(relx=0.53, rely=0.596, relwidth=0.4, relheight=0.4)
        self.text_anim = FuncAnimation(self.text_fig, self.text_sentiment_plot, 5000)
        self.text_anim.event_source.start()
        


if __name__ == "__main__":

    multiprocessing.freeze_support()
    app = Moodzilla()

    meter = VolumeMeter(app)
    meter.place(relx=0.5, rely=0.75, anchor='center')

    with sd.InputStream(callback=meter.update_meter):
        app.mainloop()
