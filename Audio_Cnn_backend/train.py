import modal
from model import Audio_Model
import numpy as np
import torch.nn as nn
from pathlib import Path
import torchaudio
import torch
from torch.utils.data import Dataset, DataLoader

import torchaudio.transforms as T

import pandas as pd
from model import Audio_Model
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

app = modal.App("audio_cnn")

image = (modal.Image.debian_slim()
         .pip_install_from_requirements("requirements.txt")
         .apt_install(["wget", "unzip", "ffmpeg", "libsndfile1"])
         .run_commands([
             "cd /tmp && wget https://github.com/karolpiczak/ESC-50/archive/master.zip -O esc50.zip",
             "cd /tmp && unzip esc50.zip",
             "mkdir -p /opt/esc50-data",
             "cp -r /tmp/ESC-50-master/* /opt/esc50-data/",
             "rm -rf /tmp/esc50.zip /tmp/ESC-50-master"
         ])
         .add_local_python_source("model"))

volume = modal.Volume.from_name("esc50-data",create_if_missing=True)
model_volume = modal.Volume.from_name("esc-model",create_if_missing=True)



class ESC50(Dataset):
    def __init__(self,data_dir,metadata_file,split,transform=None):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.metadata = pd.read_csv(metadata_file)
        self.split = split
        self.transform = transform

        if split == "train":
            self.metadata = self.metadata[self.metadata["fold"] != 5]
        else:
            self.metadata = self.metadata[self.metadata["fold"] == 5]

        self.classes = sorted(self.metadata["category"].unique())

        self.class_to_idx = {cls: idx for idx,cls in enumerate(self.classes)}
        # create a new label containing only the idx
        self.metadata["label"] = self.metadata["category"].map(self.class_to_idx)

    def __len__(self):
        return len(self.metadata)
    

    def __getitem__(self,idx):
        # get the row which gives the filename and the label
        row = self.metadata.iloc[idx]

        audio_path = self.data_dir/ "audio" / row["filename"]

        waveform,sample_rate = torchaudio.load(audio_path)
         # convert stero to mono
        if waveform.shape[0] > 1 :
            waveform = torch.mean(input = waveform,dim = 0 ,keepdim =True)
        else:
            waveform = waveform
        
        if self.transform :
            spectrogram = self.transform(waveform)
        else:
            spectrogram = waveform

        return spectrogram, row["label"]
    
def mixed_dataset(x,y):
    lam = np.random.beta(0.2,0.2)

    batch_size = x.size(0)


    index = torch.randperm(batch_size).to(x.device)

    mixed_data = lam * x + (1-lam) * x[index,:]

    y_original, y_mixed = y, y[index]
    return mixed_data,y_original,y_mixed,lam

def mixed_criterion(loss_fn,pred,y_original,y_mixed,lam):
    return lam * loss_fn(pred ,y_original) + (1-lam) *loss_fn(pred, y_mixed)
        

@app.function(image=image,gpu="A10G", volumes={"/data": volume,"/models": model_volume}, timeout=60 * 60 *3)
def train():

    from datetime import datetime
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S%")
    log_dir = f"/models/tensorboard_logs/run_{timestamp}"
    writer = SummaryWriter(log_dir=log_dir)
    # create the train transform and validation transform

    train_transform = nn.Sequential(
        T.MelSpectrogram(
            sample_rate= 44100,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            f_min=0,
            f_max= 22050 # 44100/2
        ),
        T.AmplitudeToDB(),
        T.FrequencyMasking(freq_mask_param=30), # to avoid overfitting and reduce noise
        T.TimeMasking(time_mask_param=80)

    )

    val_transform = nn.Sequential(
        T.MelSpectrogram(
            sample_rate=44100,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            f_min=0,
            f_max= 22050 # 44100/2
        ),
        T.AmplitudeToDB()
    )   

    data_path = Path("/opt/esc50-data")
    
    train_data = ESC50(data_dir = data_path,metadata_file = data_path /"meta"/ "esc50.csv" ,split = "train",transform= train_transform)
   
    val_data = ESC50(data_dir = data_path,metadata_file = data_path /"meta"/ "esc50.csv" ,split = "val",transform= val_transform)
    print(f"training {len(train_data)}")
    print(f"training {len(val_data)}")

    # data loader 

    train_dataloader = DataLoader(dataset= train_data,
                                  batch_size= 32,
                                  shuffle = True)
    test_dataloader = DataLoader(dataset= val_data,
                                 batch_size= 32,
                                 shuffle = False)
    
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = Audio_Model(num_classes= len(train_data.classes))
    model.to(device)


    num_epochs = 150
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(params=model.parameters(),
                                  lr=0.0001,
                                  weight_decay= 0.01)
    
    scheduler_lr = OneCycleLR(
        optimizer=optimizer,
        max_lr= 0.0035,
        epochs = num_epochs,
        steps_per_epoch=len(train_dataloader),
        pct_start= 0.1,
        final_div_factor=1000 # keeps LR higher
    )

    best_accuracy = 0.0
    

    for epoch in range(num_epochs):

        epoch_loss = 0.0

        model.train()

        epoch_loss = 0.0

        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')

        for data,target in progress_bar:
            data,target =data.to(device), target.to(device)

            # creating mixed data to help the model build a robust training

            if np.random.random() > 0.7:
                mixed_data,original_label,mixed_label,lam = mixed_dataset(x=data,y=target)
                pred_output =  model(mixed_data)
                loss =  mixed_criterion(loss_fn=criterion,pred =pred_output,y_original=original_label,y_mixed=mixed_label,lam =lam)
            else:
                pred_output = model(data)
                loss = criterion(pred_output,target)

            
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            scheduler_lr.step()

            epoch_loss += loss.item()

            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

        avg_epoch_loss = epoch_loss/ len(train_dataloader)

        writer.add_scalar("Loss/Train",
                          avg_epoch_loss,
                          epoch)
        writer.add_scalar("Learning_rate", 
                          optimizer.param_groups[0]["lr"],
                          epoch)


        model.eval()
        correct = 0
        total = 0
        val_loss = 0

        with torch.inference_mode():
            for data,target in test_dataloader:
                data,target = data.to(device),target.to(device)

                test_pred = model(data)

                test_loss = criterion(test_pred,target)


                val_loss += test_loss.item()

                
                predicted = torch.argmax(test_pred,dim =1) # _, predicted = torch.max(test_pred.data, 1)
                total += target.size(0) # batch size
                correct += (predicted == target).sum().item()
        

        avg_val_loss = val_loss/ len(test_dataloader)

        accuracy = 100 * correct / total # len(test_pred)

        writer.add_scalar("Loss/Validation",
                          avg_val_loss,
                          epoch)
        writer.add_scalar("Accuracy/Validation", 
                          accuracy,
                          epoch)

        print(f" Epoch: {epoch+1} | Loss: {avg_epoch_loss:.4f} | Val_loss: {avg_val_loss:.4f} Accuracy: {accuracy:.4f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy

            torch.save({
                "model_state_dict": model.state_dict(),
                "accuracy": best_accuracy,
                "epoch": epoch,
                "classes": train_data.classes
            },f = "/models/best_model.pth")

            print(f" The Best Accuracy {accuracy: .2f}")
    writer.close()
    print(f" Training Completed, Best accuracy : {best_accuracy:.2f}")


@app.local_entrypoint()
def main():
    train.remote()