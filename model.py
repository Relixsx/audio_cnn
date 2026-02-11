import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self,input,hidden_unit,stride =1):
        super().__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=input,
                      out_channels= hidden_unit,
                      kernel_size= 3,
                      stride = stride,
                      padding = 1,
                      bias =False),
                      nn.BatchNorm2d(hidden_unit),
                      nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_unit,
                      out_channels=hidden_unit,
                       kernel_size=3,
                         padding=1,
                          bias = False ),
                          nn.BatchNorm2d(hidden_unit)
        )

        self.shortcut = nn.Sequential()
        self.use_shortcut = stride != 1 or input != hidden_unit
        if self.use_shortcut:
            self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels=input,
                      out_channels= hidden_unit,
                      kernel_size= 1,
                      stride = stride,
                      bias =False),
                      nn.BatchNorm2d(hidden_unit)
        )

    def forward(self,x,fmap_dict= None,prefix=""): #fmap_dict is a dictionary meant to store intermediate feature maps
        #print(f"the first shape {x.shape}")
        x_out = self.layer1(x)
        #print(f"the second shape {x_out.shape}")
        x_out =  self.layer2(x_out)
        #print(f"the third shape {x_out.shape}")
        shortcut_x =  self.shortcut(x) if self.use_shortcut else x
        #print(shortcut_x.shape)
        x_add = x_out + shortcut_x

        if fmap_dict is not None:
            fmap_dict[f"{prefix}.conv"] = x_add
        x_out = torch.relu(x_add)

        if fmap_dict is not None:
            fmap_dict[f"{prefix}.relu"] = x_out
        return x_out
    

class Audio_Model(nn.Module):
    def __init__(self,num_classes):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels= 1,
                      out_channels= 64,
                      kernel_size= 7,
                      stride = 2,
                      padding =3,
                      bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace= True))

        self.layer2 = nn.ModuleList([ResidualBlock(64,64) for i in range(3)])
        self.layer3 = nn.ModuleList([ResidualBlock(64 if i== 0 else 128, 128, stride= 2 if i== 0 else 1) for i in range(4)])
        self.layer4 = nn.ModuleList([ResidualBlock(128 if i== 0 else 256,256, stride= 2 if i== 0 else 1) for i in range(6)])
        self.layer5 = nn.ModuleList([ResidualBlock(256 if i== 0 else 512,512, stride= 2 if i== 0 else 1) for i in range(3)])



        self.classifier = nn.Sequential(

            nn.AdaptiveAvgPool2d(output_size= (1,1)),
            nn.Flatten(start_dim=1),
            nn.Dropout(0.2),
            nn.Linear(in_features= 512,out_features=num_classes )
        )
      

    def forward(self,x, return_feature_maps = False):

        if not return_feature_maps:
            x = self.layer1(x)
            # Go throught the list using a for loop
            for block in self.layer2:
                x = block(x)
            for block in self.layer3:
                x = block(x)
            for block in self.layer4:
                x = block(x)
            for block in self.layer5:
                x = block(x)
            
            x = self.classifier(x)

            return x
        
        else:
            feature_maps = {}
            x = self.layer1(x)
            feature_maps["Convlayer1"] = x
            # Go throught the list using a for loop
            for i ,block in enumerate(self.layer2):
                x = block(x,feature_maps,prefix = f"layer2.block{i}")
            feature_maps["layer2"] = x
            for i,block in enumerate(self.layer3):
                x = block(x,feature_maps,prefix = f"layer2.block{i}")
            feature_maps["layer3"] = x
            for i,block in enumerate(self.layer4):
                x = block(x,feature_maps,prefix = f"layer2.block{i}")
            feature_maps["layer4"] = x
            for i,block in enumerate(self.layer5):
                x = block(x,feature_maps,prefix = f"layer2.block{i}")
            feature_maps["layer5"] = x
            x = self.classifier(x)
            
            return x, feature_maps

