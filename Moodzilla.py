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
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
import time

import speech_recognition as sr
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import torch
import torch.nn.functional as F

import librosa

import gc
gc.enable()

ctk.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("green")  # Themes: "blue" (standard), "green", "dark-blue"

class VolumeMeter(ctk.CTkFrame):
    def __init__(self, parent, *args, **kwargs):
        ctk.CTkFrame.__init__(self, parent, *args, **kwargs)
        self.canvas = tk.Canvas(self, width=3, height=200, highlightthickness=0, borderwidth=0, relief='ridge')
        self.canvas.pack()
        self.meter = self.canvas.create_rectangle(0, 200, 3, 200, fill='#1FAB89')

    def update_meter(self, indata, frames, time, status):
        volume_norm = np.linalg.norm(indata)/10
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
            img = ctk.CTkImage(dark_image=Image.open('Icons/slide_left_icon.png'))
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
            img = ctk.CTkImage(dark_image=Image.open('Icons/slide_right_icon.png'))
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
            img = ctk.CTkImage(dark_image=Image.open('Icons/slide_right_icon2.png'))
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
            img = ctk.CTkImage(dark_image=Image.open('Icons/slide_left_icon2.png'))
            self.parent.slidepanel_right_btn.configure(image=img)


class Moodzilla(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # title
        self.title('Moodzilla')
        
        # ico
        self.iconbitmap("Icons/Moodzilla.ico")

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
        self.slide_right_image = ctk.CTkImage(dark_image=Image.open("Icons/slide_right_icon.png"))
        self.slidepanel_left_btn = ctk.CTkButton(self, text='', fg_color='transparent', image=self.slide_right_image, command=self.slidepanel_left.animate, hover=False)
        self.slidepanel_left_btn.place(relx=0, rely=0.38, relwidth=0.03, relheight=0.2)


        # left panel widgets
        self.left_panel_img = ctk.CTkImage(dark_image=Image.open("Icons/Moodzilla.png"), size=(200, 200))
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
        self.slide_left_image = ctk.CTkImage(dark_image=Image.open("Icons/slide_left_icon2.png"))
        self.slidepanel_right_btn = ctk.CTkButton(self, text='', fg_color='transparent', image=self.slide_left_image, command=self.slidepanel_right.animate, hover=False)
        self.slidepanel_right_btn.place(relx=0.97, rely=0.38, relwidth=0.03, relheight=0.2)

        self.speech_text_label_title_frame = ctk.CTkFrame(self.slidepanel_right, border_color='#FFF', border_width=2)
        self.speech_text_label_title_frame.place(relx=0.5, rely=0.2, relwidth=0.85, relheight=0.05, anchor='center')
        self.speech_text_label_title = ctk.CTkLabel(self.speech_text_label_title_frame, text=' Speech Text', corner_radius=5, font=('Sans Serif', 15), fg_color='#0D1117', text_color='#FFF', anchor='w')
        self.speech_text_label_title.place(relx=0.01, rely=0.07, relwidth=0.978, relheight=0.85)
        
        self.speech_text_frame = ctk.CTkFrame(self.slidepanel_right, border_color='#FFF', border_width=2, fg_color='#171B23')
        self.speech_text_frame.place(relx=0.5, rely=0.5, relwidth=0.85, relheight=0.5, anchor='center')
        self.speech_text = ctk.CTkLabel(self.speech_text_frame, text='', wraplength=300, justify='center', corner_radius=2, font=('Sans Serif', 15), fg_color='#171B23', text_color='#FFF', anchor='nw')
        self.speech_text.place(relx=0.5, rely=0.5, relwidth=0.95, relheight=0.95, anchor='center')

        self.speech_text_clear_btn = ctk.CTkButton(self.slidepanel_right, text='Clear Text Box', border_color='#FFF', corner_radius=5, fg_color='#850E35', hover=True, hover_color='#EE6983', anchor='center', command=lambda: self.speech_text.configure(text=''))
        self.speech_text_clear_btn.place(relx=0.5, rely=0.8, relwidth=0.7, relheight=0.05, anchor='center')

        # Loading Models
        self.facial_model = load_model('Models/fer2013n_model_i2.h5')

        self.config = AutoConfig.from_pretrained('Models/config.json')
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.text_model = AutoModelForSequenceClassification.from_pretrained("Models/pytorch_model.bin", config=self.config)

        self.tone_model = load_model('Models/speech_emotion_model_i2.h5')

        
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

        # Create the no speech detected text display
        self.speech_text_frame = ctk.CTkFrame(self.bg_frame, border_color='#FF6000', border_width=3, fg_color='transparent')
        self.speech_text_frame.place(relx=0.5, rely=0.548, relwidth=0.3, relheight=0.08, anchor='center')
        self.speech_text_label = ctk.CTkLabel(self.speech_text_frame, text='', font=("Helvetica", 15), text_color='#FFF', fg_color='transparent')
        self.speech_text_label.place(relx=0.5, rely=0.5, relwidth=0.978, relheight=0.80, anchor='center')
        self.speech_text_frame.lift()


        # Facial Sentiment Module

        self.face_flag = False
        self.facial_prediction = []
        self.facial_emotions = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear"]

        
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

        self.audio = None

        self.text_flag = False
        self.text_prediction = []
        self.text_emotions = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

        # button to start/stop the speech capture thread
        self.startstop_speech_flag = False
        self.capture_speech_thread = None

        # initialize the recognizer
        self.r = sr.Recognizer()
        self.text = ''
        self.preprocessed_input = None

        self.startstop_speech_btn = ctk.CTkButton(self, fg_color='#E76161', text_color='#FFF', corner_radius=5, border_width=2, border_color='#98D8AA', text='Start Speech Recording', hover=True, hover_color='#E84545', anchor="center", command=self.start_stop_speech_recording)
        self.startstop_speech_btn.place(relx=0.635, rely=0.9375, relwidth=0.15, relheight=0.05)


        self.text_fig = Figure(figsize=(3, 2), dpi=100)
        self.text_ax = self.text_fig.add_subplot(111)
        self.text_canvas = FigureCanvasTkAgg(self.text_fig, master=self.bg_frame)
        self.text_canvas.get_tk_widget().place(relx=0.53, rely=0.596, relwidth=0.4, relheight=0.4)
        self.text_anim = FuncAnimation(self.text_fig, self.text_sentiment_plot, 100)

        self.tone_flag = False
        self.spectrogram = None
        self.tone_fig = Figure(figsize=(3, 2), dpi=100)
        self.tone_ax = self.tone_fig.add_subplot(111)
        self.tone_canvas = FigureCanvasTkAgg(self.tone_fig, master=self.bg_frame)
        self.tone_canvas.get_tk_widget().place(relx=0.065, rely=0.596, relwidth=0.4, relheight=0.4)
        self.tone_anim = FuncAnimation(self.tone_fig, self.tone_sentiment_plot, 100)
        
    
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
                    faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.1, 8)
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
                    for i, prob in enumerate(self.facial_prediction):
                        if prob*100 > 10.0:
                            sentiment += self.facial_emotions[i] + ' '

                    self.face_senti_label.configure(text=sentiment)

                if self.startstop_face_flag == False:
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

                        
            # Create horizontal bar chart
            
            y_pos = np.arange(len(self.facial_emotions))
            self.face_ax.barh(y_pos, emotion_probs, align='center', color=colors)
            self.face_ax.set_yticks(y_pos)
            self.face_ax.set_yticklabels(self.facial_emotions, fontsize=6)
            self.face_ax.invert_yaxis()  # labels read top-to-bottom
            self.face_ax.set_xlabel('Probability')
            self.face_ax.set_title('Facial Sentiment Probabilities')


            # Add probability values as labels above each bar
            for i, prob in enumerate(emotion_probs):
                self.face_ax.text(prob+1, i, str(round(prob, 3)), color='black', va='center', fontsize=8)

            # Set the minimum and maximum limits of the x-axis
            self.face_ax.set_xlim(0, 100)
            self.face_ax.set_ylim(-0.5, len(self.facial_emotions) - 0.5)

            # plt.show()


            # Remove the old canvas, redraw the plot and create a new canvas
            # if self.face_canvas is not None:
            #     self.face_canvas.get_tk_widget().destroy()
            self.face_canvas.draw()

        return self.face_canvas

    # Function to capture the speech feed
    def capture_speech_feed(self):

        if self.startstop_speech_flag == True:
            # open the microphone and start recording
            with sr.Microphone() as source:
                
                try:
                    print('Say Something, Time allowed = 2s')
                    self.audio = self.r.listen(source, timeout=2)
                    # Check if recording should be interrupted
                    # if keyboard.is_pressed('q'):  # Replace 'q' with your desired keyword or key
                    #     print("Recording interrupted.")
                    # else:
                    # # Process the audio data here
                    #     print("Audio recorded.")    
                    # transcribe the audio to text
                    if self.audio is not None:  
                        # Define the sample rate and duration
                        sample_rate = 44100

                        # Define the spectrogram parameters
                        n_fft = 2048
                        hop_length = 512
                        n_mels = 128

                        # Convert the recorded audio to a numpy array
                        tone_audio = np.frombuffer(self.audio.frame_data, dtype=np.float16)
                        tone_audio = np.nan_to_num(tone_audio, nan=0.0, posinf=0.0, neginf=0.0)


                        # Convert the audio to spectrogram
                        spectrogram = librosa.feature.melspectrogram(y=tone_audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)


                        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

                        # Preprocess the audio data
                        spectrogram = spectrogram[..., np.newaxis] # Add a new axis for the channel dimension
                        self.spectrogram = tensorflow.image.resize(spectrogram, (128, 128))


                        self.tone_flag = True
                        text = self.r.recognize_google(self.audio)
                        print(text)
                        self.text = self.text + '\n' + text
                        self.speech_text.configure(text=self.text)
                        def preprocess_sentence(sentence):
                            return self.tokenizer(sentence, padding="max_length", truncation=True, return_tensors="pt")

                        # perform classification on the transcribed text
                        input_sentence = self.text
                        self.preprocessed_input = preprocess_sentence(input_sentence)
                        self.text_flag = True

                    
                        self.speech_text_label.configure(text='')

                except (sr.UnknownValueError, sr.WaitTimeoutError):
                    self.speech_text_label.configure(text='No Speech Detected. Please try again')
                    self.tone_flag = False
                    self.text_flag = False


    # Function to plot the text sentiment bar chart
    def text_sentiment_plot(self, i):
    
        #  Only when the flag is true, it mean that we have gotten the prediction from the capture_video_feed()
        if self.text_flag == True:

            self.text_ax.clear()
                    

            with torch.no_grad():
                logits = self.text_model(**self.preprocessed_input)[0]
                probs = F.softmax(logits, dim=1)

            self.text_prediction = np.array(probs)[0].tolist()

            emotion_probs = [prob*100 for prob in self.text_prediction]

            # Set bar color based on emotion
            colors = ['#FF5733', '#6A5ACD', '#800000', '#FFFF00', '#A9A9A9', '#1E90FF']

                        
            # Create horizontal bar chart
            
            y_pos = np.arange(len(self.text_emotions))
            self.text_ax.barh(y_pos, emotion_probs, align='center', color=colors)
            self.text_ax.set_yticks(y_pos)
            self.text_ax.set_yticklabels(self.text_emotions, fontsize=6)
            self.text_ax.invert_yaxis()  # labels read top-to-bottom
            self.text_ax.set_xlabel('Probability')
            self.text_ax.set_title('Textual Sentiment Probabilities')
   

            # Add probability values as labels above each bar
            for i, prob in enumerate(emotion_probs):
                self.text_ax.text(prob+1, i, str(round(prob, 3)), color='black', va='center', fontsize=8)

            # Set the minimum and maximum limits of the x-axis
            self.text_ax.set_xlim(0, 100)
            self.text_ax.set_ylim(-0.5, len(self.text_emotions) - 0.5)

            # plt.show()


            # Remove the old canvas, redraw the plot and create a new canvas
            # if self.face_canvas is not None:
            #     self.face_canvas.get_tk_widget().destroy()
            self.text_canvas.draw()
            self.text_flag = False

        return self.text_canvas
    
    # Function to plot the tone sentiment bar chart
    def tone_sentiment_plot(self, i):

        if self.tone_flag == True:

            if self.audio is not None:
                self.tone_ax.clear()
                

                # Predict the emotion label of the audio sample
                probs = self.tone_model.predict(self.spectrogram[np.newaxis, ...], verbose=0)[0].tolist()

                # Convert the predicted label to the corresponding emotion category
                tone_emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

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

                # plt.show()


                # Remove the old canvas, redraw the plot and create a new canvas
                # if self.face_canvas is not None:
                #     self.face_canvas.get_tk_widget().destroy()
                self.tone_canvas.draw()
                self.tone_flag = False

        return self.tone_canvas

    # Function to set the speech record button flag
    def start_stop_face_recording(self):
        if self.startstop_face_flag == True:
            self.startstop_face_flag = False
            self.startstop_face_btn.configure(text='Start Video Recording')
            

            # giving timeout as a parameter is crucial to let the thread complete out its work and then close gracefully
            self.face_senti_label.configure(text='')
            self.face_senti_label.configure(text='')
            self.capture_video_thread.join(timeout=1)
            time.sleep(0.5)
            self.face_flag = False
            self.video_label.configure(image=None)
            self.face_anim.event_source.stop()
            self.face_ax.clear()
            self.face_ax.clear()
            self.face_ax.clear()
            self.face_text_label.configure(text='')
            self.face_senti_label.configure(text='')
            self.face_senti_label.configure(text='')
            self.face_senti_label.configure(text='')
            self.face_senti_label.configure(text='')
            

        else:
            self.startstop_face_flag = True
            self.startstop_face_btn.configure(text='Stop Video Recording')
            self.face_anim.event_source.start()

            # Set up the video feed capture thread
            self.capture_video_thread = threading.Thread(target=self.capture_video_feed)
            self.capture_video_thread.start()

    # Function to set the speech record button flag
    def start_stop_speech_recording(self):
        if self.startstop_speech_flag == True:
            self.startstop_speech_flag = False
            self.startstop_speech_btn.configure(text='Start Speech Recording')

            # giving timeout as a parameter is crucial to let the thread complete out its work and then close gracefully
            self.text_ax.clear()
            self.tone_ax.clear()
            self.text_anim.event_source.stop()
            self.tone_anim.event_source.stop()
            self.text_ax.clear()
            self.tone_ax.clear()
            self.capture_speech_thread.join(timeout=1)
            self.text_flag = False
            self.tone_flag = False
            self.text_ax.clear()
            self.tone_ax.clear()

            self.speech_text_label.configure(text='')
            self.text_ax.clear()
            self.tone_ax.clear()
            self.text_ax.clear()

        else:
            self.startstop_speech_flag = True
            self.startstop_speech_btn.configure(text='Stop Speech Recording')
            self.text_anim.event_source.start()
            self.tone_anim.event_source.start()

            # Set up the video feed capture thread
            self.capture_speech_thread = threading.Thread(target=self.capture_speech_feed)
            self.capture_speech_thread.start()

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




if __name__ == "__main__":
    app = Moodzilla()

    meter = VolumeMeter(app)
    meter.place(relx=0.496, rely=0.59)

    with sd.InputStream(callback=meter.update_meter):
        app.mainloop()