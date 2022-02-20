import streamlit as st
from save_audio import save_audio
from configure import auth_key
import pandas as pd
from time import sleep
import urllib.request
import plotly.express as px
import plotly.graph_objects as go
from urllib.request import urlopen
from bs4 import BeautifulSoup
import json
import requests
import pyaudio
import wave
import os



## AssemblyAI endpoints and headers
transcript_endpoint = "https://api.assemblyai.com/v2/transcript"
upload_endpoint = 'https://api.assemblyai.com/v2/upload'

headers_auth_only = {'authorization': auth_key}
headers = {
   "authorization": auth_key,
   "content-type": "application/json"
}



## App explanation
st.title('Sentiment analysis of real-time 30s speech')
st.caption('With this app you can analyse the sentiment of 30s speech')

##record audio
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 30
WAVE_OUTPUT_FILENAME = "voice.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

st.caption("* recording start -----")
st.caption("recording...")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

st.caption("* done recording -----")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

# Save audio locally
save_location = WAVE_OUTPUT_FILENAME


## Upload audio to AssemblyAI

def read_file(filename):
	with open(filename, 'rb') as _file:
		while True:
			data = _file.read(CHUNK)
			if not data:
				break
			yield data

upload_response = requests.post(
	upload_endpoint,
	headers=headers_auth_only, data=read_file(save_location)
)

audio_url = upload_response.json()['upload_url']
print('Uploaded to', audio_url)



## Start transcription job of audio file
data = {
	'audio_url': audio_url,
	'sentiment_analysis': 'True',
}

transcript_response = requests.post(transcript_endpoint, json=data, headers=headers)
print(transcript_response)

transcript_id = transcript_response.json()['id']
polling_endpoint = transcript_endpoint + "/" + transcript_id

print("Transcribing at", polling_endpoint)



## Waiting for transcription to be done
status = 'submitted'
while status != 'completed':
	print('not ready yet')
	sleep(1)
	polling_response = requests.get(polling_endpoint, headers=headers)
	transcript = polling_response.json()['text']
	status = polling_response.json()['status']
	

# Display transcript
print('creating transcript')
st.sidebar.header('Transcript of the 30s speech')
st.sidebar.markdown(transcript)


print(json.dumps(polling_response.json(), indent=4, sort_keys=True))



## Sentiment analysis response	
sar = polling_response.json()['sentiment_analysis_results']

## Save to a dataframe for ease of visualization
sen_df = pd.DataFrame(sar)
print(sen_df.head())





## Visualizations
st.markdown("### Number of sentences: " + str(sen_df.shape[0]))


grouped = pd.DataFrame(sen_df['sentiment'].value_counts()).reset_index()
grouped.columns = ['sentiment','count']
print(grouped)


col1, col2 = st.columns(2)


# Display number of positive, negative and neutral sentiments
fig = px.bar(grouped, x='sentiment', y='count', color='sentiment', color_discrete_map={"NEGATIVE":"firebrick","NEUTRAL":"navajowhite","POSITIVE":"darkgreen"})

fig.update_layout(
	showlegend=False,
    autosize=False,
    width=400,
    height=500,
    margin=dict(
        l=50,
        r=50,
        b=50,
        t=50,
        pad=4
    )
)

col1.plotly_chart(fig)


## Display sentiment score
neu_perc = 0
pos_perc = 0
neg_perc = 0
if not grouped[grouped['sentiment']=='POSITIVE'].empty:
    pos_perc = grouped[grouped['sentiment']=='POSITIVE']['count'].iloc[0]*100/sen_df.shape[0]
if not grouped[grouped['sentiment']=='NEGATIVE'].empty:    
    neg_perc = grouped[grouped['sentiment']=='NEGATIVE']['count'].iloc[0]*100/sen_df.shape[0]
if not grouped[grouped['sentiment']=='NEUTRAL'].empty:
    neu_perc = grouped[grouped['sentiment']=='NEUTRAL']['count'].iloc[0]*100/sen_df.shape[0]

sentiment_score = neu_perc+pos_perc-neg_perc

fig = go.Figure()

fig.add_trace(go.Indicator(
    mode = "delta",
    value = sentiment_score,
    domain = {'row': 1, 'column': 1}))

fig.update_layout(
	template = {'data' : {'indicator': [{
        'title': {'text': "Sentiment score"},
        'mode' : "number+delta+gauge",
        'delta' : {'reference': 50}}]
                         }},
    autosize=False,
    width=400,
    height=500,
    margin=dict(
        l=20,
        r=50,
        b=50,
        pad=4
    )
)

col2.plotly_chart(fig)

## Display negative sentence locations
fig = px.scatter(sar, y='sentiment', color='sentiment', size='confidence', hover_data=['text'], color_discrete_map={"NEGATIVE":"firebrick","NEUTRAL":"navajowhite","POSITIVE":"darkgreen"})


fig.update_layout(
	showlegend=False,
    autosize=False,
    width=800,
    height=300,
    margin=dict(
        l=50,
        r=50,
        b=50,
        t=50,
        pad=4
    )
)

st.plotly_chart(fig)
os.remove(WAVE_OUTPUT_FILENAME)