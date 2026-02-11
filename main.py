import modal
import torch.nn as nn
import torch
import torchaudio.transforms as T
import numpy as np
import io
import base64
import librosa
from model import Audio_Model
import soundfile as sf
from pydantic import BaseModel
import requests


app  = modal.App("Sound_classification")

image = (modal.Image.debian_slim()
         .pip_install_from_requirements("requirements.txt")
         .pip_install("fastapi[standard]","librosa","soundfile","numpy")
         .apt_install(["libsndfile1"])
         .add_local_python_source("model"))


model_volume = modal.Volume.from_name("esc-model")


class AudioProcessor:
    def __init__(self):

        self.transform = nn.Sequential(
        T.MelSpectrogram(
            sample_rate=44100,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            f_min=0,
            f_max= 11025 # 44100/2
        ),
        T.AmplitudeToDB()
    )   
        

    def process_raw_audio(self,audio):
        # convert data to tensor
        waveform = torch.from_numpy(audio)

        # add a dimension

        waveform = waveform.unsqueeze(dim=0)


        spectogram = self.transform(waveform)

        return spectogram.unsqueeze(dim =0)
    

class InferenceRequest(BaseModel):
    audio_data: str
    


@app.cls(image=image,gpu ="A10G",volumes= {"/models":model_volume},scaledown_window=15)
class AudioClassifier:
    @modal.enter()

    def load_model(self):
        print("Loading Model")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint = torch.load(f = "/models/best_model.pth",
                                map_location= self.device)
        
        self.classes = checkpoint["classes"]

        self.model = Audio_Model(num_classes=len(self.classes)).to(self.device)

        self.model.load_state_dict(state_dict=checkpoint["model_state_dict"])

        self.model.eval()


        self.audio_processor = AudioProcessor()

        print("Model loaded on enter")


        # model endpoint
    @modal.fastapi_endpoint(method="POST")
    def inference(self, request: InferenceRequest):
        audio_bytes = base64.b64decode(request.audio_data)

        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype ="float32")

        if audio_data.ndim > 1 :
                audio_data = np.mean(audio_data,axis=1)

        if sample_rate != 44100:

            audio_data = librosa.resample(y = audio_data,orig_sr=sample_rate, target_sr=44100)


        spectogram = self.audio_processor.process_raw_audio(audio_data)

        spectogram = spectogram.to(self.device)

        with torch.inference_mode():

            output,feature_map = self.model(spectogram,return_feature_maps = True)

            output = torch.nan_to_num(output)

            pred_prob = torch.softmax(output,dim=1)

            top3_pred_value, top3_pred_index = torch.topk(pred_prob[0],k=3)

            prediction = [{"class": self.classes[index.item()], "confidence/accuracy": float(prob.item())} for prob,index in zip(top3_pred_value,top3_pred_index)]


        visualize_data = {}

        for names ,tensor in feature_map.items():
            #Checks how many dimensions (axes) the tensor has dim() == 4 means the tensor shape looks like: (batch_size, channels, height, width)
            if tensor.dim() == 4:
                # mean of the channel dim
                mean_tensor = torch.mean(tensor,dim=1)
                # Remove batch size channe
                squeeze_data = mean_tensor.squeeze(0)
                array_data = squeeze_data.cpu().numpy()
                clean_data = np.nan_to_num(array_data)

                visualize_data[names] = {
                    "shape": list(clean_data.shape),
                    "value": clean_data.tolist()
                }

        spectogram_data =  spectogram.squeeze(0).squeeze(0).cpu().numpy()
        clean_spectogram = np.nan_to_num(spectogram_data)


        max_sample =8000
        waveform_sample_rate = 44100

        if len(audio_data)> max_sample:
            steps = len(audio_data) // max_sample

            waveform_data = audio_data[::steps]
        else:
            waveform_data = audio_data

        response = {
            "predictions": prediction,
            "visualization": visualize_data,
            "input_spectogram": {
                "shape": list(clean_spectogram.shape),
                "value": clean_spectogram.tolist()},
            "waveform": {
                "values": waveform_data.tolist(),
                "sample_rate": 44100,
                "duration": len(audio_data) / 44100

            }

            }

        return response


@app.local_entrypoint()
def main():
    audio_data,sample_rate = sf.read("1-17565-A-12.wav")

    buffer =io.BytesIO()

    sf.write(buffer, audio_data,sample_rate,format ="wav")
    audio_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    payload = {"audio_data": audio_b64}

    server = AudioClassifier()

    url =  server.inference.get_web_url()
    response = requests.post(url,json=payload)
    response.raise_for_status()

    result = response.json()

    waveform_info = result.get("waveform",{})

    if waveform_info:
        values = waveform_info.get("values", {})

        print(f" First 10 values: {[round(i, 4) for i in values[:10]]}")
        print(f" Duration: {waveform_info.get("duration",0)}")



    print("Prediction")

    for pred in result.get("predictions",[]):

        print(f"  {pred["class"]} : {pred["confidence/accuracy"]:.2%}")









        