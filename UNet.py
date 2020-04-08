from torch import nn


class UNet(nn.Module):
    def __init__(self, num_classes, input_filters=3, start_features = 64):
        super().__init__()
        
        self.num_classes = num_classes
        self.input_filters = input_filters
        self.start_features = start_features

        self.down_layers1 = nn.Sequential(
            nn.Conv2d(self.input_filters, self.start_features, 3, padding=1),
            nn.BatchNorm2d(self.start_features),
            nn.ReLU(),
            nn.Conv2d(self.start_features, self.start_features, 3, padding=1),
            nn.BatchNorm2d(self.start_features),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.down_layers2 = nn.Sequential(
            nn.Conv2d(self.start_features, self.start_features*2, 3, padding=1),
            nn.BatchNorm2d(self.start_features*2),
            nn.ReLU(),
            nn.Conv2d(self.start_features*2, self.start_features*2, 3, padding=1),
            nn.BatchNorm2d(self.start_features*2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.down_layers3 = nn.Sequential(
            nn.Conv2d(self.start_features*2, self.start_features*4, 3, padding=1),
            nn.BatchNorm2d(self.start_features*4),
            nn.ReLU(),
            nn.Conv2d(self.start_features*4, self.start_features*4, 3, padding=1),
            nn.BatchNorm2d(self.start_features*4),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.down_layers4 = nn.Sequential(
            nn.Conv2d(self.start_features*4, self.start_features*8, 3, padding=1),
            nn.BatchNorm2d(self.start_features*8),
            nn.ReLU(),
            nn.Conv2d(self.start_features*8, self.start_features*8, 3, padding=1),
            nn.BatchNorm2d(self.start_features*8),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.base_layer = nn.Sequential(
            nn.Conv2d(self.start_features*8, self.start_features*16, 3, padding=1),
            nn.BatchNorm2d(self.start_features*16),
            nn.ReLU(),
            nn.Conv2d(self.start_features*16, self.start_features*16, 3, padding=1),
            nn.BatchNorm2d(self.start_features*16),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)
        )
        self.up_layers1 = nn.Sequential(
            nn.Conv2d(self.start_features*16,self.start_features*8,3, padding=1),
            nn.BatchNorm2d(self.start_features*8),
            nn.ReLU(),
            nn.Conv2d(self.start_features*8, self.start_features*8, 3, padding=1),
            nn.BatchNorm2d(self.start_features*8),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)            
        )
        self.up_layers2 = nn.Sequential(
            nn.Conv2d(self.start_features*8,self.start_features*4,3, padding=1),
            nn.BatchNorm2d(self.start_features*4),
            nn.ReLU(),
            nn.Conv2d(self.start_features*4, self.start_features*4, 3, padding=1),
            nn.BatchNorm2d(self.start_features*4),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)
        )
        self.up_layers3 = nn.Sequential(
            nn.Conv2d(self.start_features*4,self.start_features*2,3, padding=1),
            nn.BatchNorm2d(self.start_features*2),
            nn.ReLU(),
            nn.Conv2d(self.start_features*2, self.start_features*2, 3, padding=1),
            nn.BatchNorm2d(self.start_features*2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)
        )
        self.up_layers4 = nn.Sequential(
            nn.Conv2d(self.start_features*2,self.start_features,3, padding=1),
            nn.BatchNorm2d(self.start_features),
            nn.ReLU(),
            nn.Conv2d(self.start_features, self.start_features, 3, padding=1),
            nn.BatchNorm2d(self.start_features),
            nn.ReLU(),
            nn.Conv2d(self.start_features,num_classes,1)
        )
        

    def forward(self, x):
            
        x = self.down_layers1(x)
        x = self.down_layers2(x)
        x = self.down_layers3(x)
        x = self.down_layers4(x)
        x = self.base_layer(x)
        y = self.up_layers1(x)
        y = self.up_layers2(y)
        y = self.up_layers3(y)
        y = self.up_layers4(y)
        
        return y
       