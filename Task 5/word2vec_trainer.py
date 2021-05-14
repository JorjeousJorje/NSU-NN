from numpy.lib.shape_base import _replace_zero_by_x_arrays
from sklearn.utils.validation import check_X_y
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import matplotlib.pyplot as plt

class TrainProcessPlotter:
    
    @staticmethod
    def show_results_ipython(train_loss,
                             train_metric):
        from IPython.display import clear_output
        clear_output(True)

        train_dict = {'Loss': [train_loss], 'Metric': [train_metric]}
        plt.figure(figsize=(16, 4))
        
        for i, key in enumerate(train_dict):
            plt.subplot(1, 2, i + 1)
            
            item = train_dict[key]
            plt.title(key)
            plt.plot(item[0], label=f'train ({item[-1][-1]:.5})')
            plt.xlabel('Epoch #')
            plt.ylabel(key)
            plt.legend()
            plt.grid(ls='--')
        
        plt.show()


class Trainer:
    
    def __init__(self, model, train_loader, dataset, batchsize, neg_samp=False):
        self.model = model
        self.train_loader = train_loader
        self.dataset = dataset
        self.batchsize = batchsize
        self.neg_samp = neg_samp
    
    @staticmethod
    def accuracy(predictions, labels):
        indices = torch.argmax(predictions, dim=1)
        correct_samples = torch.sum(indices == labels)
        total_samples = len(labels)
        
        return float(correct_samples) / total_samples
    
    @staticmethod
    def binary_accuracy(predictions, labels):
        indices = torch.argmax(predictions, dim=1)
        correct_samples = torch.sum(indices == 0)
        total_samples = len(labels)
        
        return float(correct_samples) / total_samples
    
    def train(self, opt, loss_fn, device, metric_fn=accuracy, scheduler=None, epochs=15):
        self.model.to(device)
        train_loss, train_metric = [], []

        for epoch in range(1, epochs + 1):
            self.model.train()
            
            print(f'[ Training.. {epoch}/{epochs} ]')
            train_epoch_loss, train_epoch_metric = self.__epoch_step(opt=opt, 
                                                                     loss_fn=loss_fn, 
                                                                     metric_fn=metric_fn, 
                                                                     device=device,
                                                                     loader=self.train_loader)
            train_loss.append(train_epoch_loss)
            train_metric.append(train_epoch_metric)
            
            if scheduler is not None:
               scheduler.step()
                    
            TrainProcessPlotter.show_results_ipython(train_loss, train_metric)
            
            
    def __epoch_step(self, loss_fn, metric_fn, device, opt, loader):
        
        epoch_loss = 0
        predictions = []
        labels = []

        if not self.neg_samp:
            for i_step, (input_vector, output_index) in enumerate(loader):
                    x, y = input_vector.to(device), output_index.to(device)
                    prediction = self.model(x)
                    loss_value = loss_fn(prediction, y)
                    
                    if opt is not None:
                        opt.zero_grad()
                        loss_value.backward()
                        opt.step()
                    
                    predictions.append(prediction)
                    labels.append(y)
                    
                    epoch_loss += loss_value
        else:
            for i_step, (input_vector, output_indices, output_target) in enumerate(loader):
                        x = input_vector.to(device)
                        h = output_indices.to(device)
                        y = output_target.to(device)
                        prediction = self.model(x, h)
                        loss_value = loss_fn(prediction, y.type_as(prediction))
                        
                        if opt is not None:
                            opt.zero_grad()
                            loss_value.backward()
                            opt.step()
                        
                        predictions.append(prediction)
                        labels.append(y.type_as(prediction))
                        
                        epoch_loss += loss_value
        num_batches = i_step + 1
        
        predictions, labels = torch.cat(predictions), torch.cat(labels)
        epoch_loss /= num_batches
        epoch_metric = metric_fn(predictions, labels)

        self.dataset.generate_dataset() # Regenerate dataset every epoch
        self.train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batchsize)
        
        return epoch_loss, epoch_metric



                    
                    
                    
                    
                    
                    
                    
                    
                
            
        