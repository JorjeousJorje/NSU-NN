import torch.nn as nn


class Flattener(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        batch_size, *_ = x.shape
        return x.view(batch_size, -1)
    

class NetModule(nn.Module):
    
    def __add_activation_layer(self, activation):
        if activation == 'relu':
            self.modules.append(nn.ReLU(inplace=True))
        else:
            self.modules.append(nn.Sigmoid())
    
    def __add_helper_layers(self, hidden_size, activation, batch_norm, batch_norm_after_activation):
        if batch_norm and batch_norm_after_activation:
            self.__add_activation_layer(activation)
            self.modules.append(nn.BatchNorm1d(hidden_size))
        elif batch_norm and not batch_norm_after_activation:
            self.modules.append(nn.BatchNorm1d(hidden_size))
            self.__add_activation_layer(activation)
        else:
            self.__add_activation_layer(activation)
    
    def __add_end_layer(self, hidden_size, n_output):
        n_classes = n_output
        self.modules.append(nn.Linear(in_features=hidden_size, out_features=n_classes))
            
    def __init__(self, in_features, hidden_size=32, start_point=False, end_point=False,
                 n_output=0, activation='relu', batch_norm=False, batch_norm_after_activation=False):
        super().__init__()
        
        
        self.modules = [nn.Linear(in_features=in_features, out_features=hidden_size)]
        
        if start_point:
            self.modules.insert(0, Flattener())
        
        self.__add_helper_layers(hidden_size, activation, batch_norm, batch_norm_after_activation)
        
        if end_point:
            self.__add_end_layer(hidden_size, n_output)

        
        
class FCNet(nn.Module):
    def __init_network(self, hidden_sizes, n_output, activation, batch_norm, batch_norm_after_activation):
        
        end_point = False
        current_output = 0
        for i in range(1, len(hidden_sizes)):
            
            if i == len(hidden_sizes) - 1:
                end_point = True
                current_output = n_output
                
            module = NetModule(in_features=hidden_sizes[i - 1],
                               hidden_size=hidden_sizes[i],
                               end_point=end_point,
                               n_output=current_output,
                               activation=activation, 
                               batch_norm=batch_norm,
                               batch_norm_after_activation=batch_norm_after_activation)
            
            for layer in module.modules:
                self.net_layers.append(layer)
    
    
    def __init__(self, in_features, n_output, hidden_sizes=[32], 
                 activation='relu', batch_norm=False, batch_norm_after_activation=False):
        super().__init__()
        assert isinstance(hidden_sizes, list), "hidden_sizes is not a list"
        
        end_point = True if len(hidden_sizes) == 1 else False
        
        start_layer = NetModule(in_features, 
                                hidden_sizes[0], 
                                start_point=True,
                                end_point=end_point,
                                activation=activation, 
                                batch_norm=batch_norm,
                                batch_norm_after_activation=batch_norm_after_activation,
                                n_output=n_output)
        
        self.net_layers = [*start_layer.modules]
        
        
        if len(hidden_sizes) > 1:
            self.__init_network(hidden_sizes, n_output, activation, batch_norm, batch_norm_after_activation)
        
        
        self.features = nn.Sequential(*self.net_layers)

    
    def forward(self, x):
        features = self.features(x)
        return features
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
