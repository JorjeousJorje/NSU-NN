import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import matplotlib.pyplot as plt

class TrainProcessPlotter:
    
    @staticmethod
    def show_results_ipython(train_loss,
                             train_metric,
                             dev_loss=None,
                             dev_metric=None):
        from IPython.display import clear_output
        clear_output(True)

        train_val_dict = {'Loss': [train_loss, dev_loss], 'Metric': [train_metric, dev_metric]}
        plt.figure(figsize=(16, 4))
        
        for i, key in enumerate(train_val_dict):
            plt.subplot(1, 2, i + 1)
            
            item = train_val_dict[key]
            train_item = item[0]
            val_item = item[1]
            
            plt.title(key)
            plt.plot(train_item, label=f'train ({train_item[-1]:.3})')

            if val_item is not None:
                plt.plot(val_item, label=f'dev ({val_item[-1]:.3})')
            plt.xlabel('Epoch #')
            plt.ylabel(key)
            plt.legend()
            plt.grid(ls='--')
        
        plt.show()


class Trainer:
    
    def __init__(self, model, train_loader, val_loader=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
    def train(self, opt, loss_fn, metric_fn, device, empty_labels=False, scheduler=None, epochs=15):
        
        self.model.to(device)
        train_loss, train_metric = [], []
        val_loss, val_metric = None, None
        
        if self.val_loader is not None:
            val_loss, val_metric = [], []
        
        # plato_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2)
        for epoch in range(1, epochs + 1):
            self.model.train()
            
            print(f'[ Training.. {epoch}/{epochs} ]')
            train_epoch_loss, train_epoch_metric = self.__epoch_step(opt=opt, 
                                                                     loss_fn=loss_fn, 
                                                                     metric_fn=metric_fn, 
                                                                     device=device,
                                                                     loader=self.train_loader,
                                                                     empty_labels=empty_labels)
            
            train_loss.append(train_epoch_loss)
            train_metric.append(train_epoch_metric)
            
            if self.val_loader is not None:
                
                print(f'[ Validating.. {epoch}/{epochs} ]')
                val_epoch_loss, val_epoch_metric = self.__validation_step(loss_fn=loss_fn,
                                                                          metric_fn=metric_fn, 
                                                                          device=device,
                                                                          empty_labels=empty_labels)
                val_loss.append(val_epoch_loss)
                val_metric.append(val_epoch_metric)
            
            if scheduler is not None:
               scheduler.step()
                    
            TrainProcessPlotter.show_results_ipython(train_loss, train_metric, val_loss, val_metric)
        return val_metric[-1]
            
            
    def __epoch_step(self, loss_fn, metric_fn, device, opt, loader, empty_labels=False):
        
        epoch_loss = 0
        predictions = []
        labels = []
        if not empty_labels:
            for i_step, (x, y) in enumerate(loader):
                x, y = x.to(device), y.to(device)
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
            for i_step, x in enumerate(loader):
                x, y = x[0].to(device), x[1].to(device)
                prediction = self.model(x)
                loss_value = loss_fn(prediction, y)
                
                if opt is not None:
                    opt.zero_grad()
                    loss_value.backward()
                    opt.step()
                
                predictions.append(prediction)
                labels.append(y)
                
                epoch_loss += loss_value
        
        num_batches = i_step + 1
        
        predictions, labels = torch.cat(predictions), torch.cat(labels)
        epoch_loss /= num_batches
        epoch_metric = metric_fn(predictions, labels)
        
        return epoch_loss, epoch_metric


    def __validation_step(self, loss_fn, metric_fn, device, empty_labels):
        
        with torch.no_grad():
            self.model.eval()
            val_loss, val_metric = self.__epoch_step(loss_fn=loss_fn,
                                                     metric_fn=metric_fn,
                                                     device=device,
                                                     opt=None,
                                                     loader=self.val_loader,
                                                     empty_labels=empty_labels)
        return val_loss, val_metric

        
    def test(self, test_loader, device, metric_fn, empty_labels=False):
        with torch.no_grad():
            self.model.eval()
            predictions = []
            labels = []

            if not empty_labels:
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    prediction = self.model(x)
                    predictions.append(prediction)
                    labels.append(y)
            else:
                for x in test_loader:
                    image, label = x[0].to(device), x[1].to(device)
                    prediction = self.model(image)
                    predictions.append(prediction)
                    labels.append(label)


            predictions, labels = torch.cat(predictions), torch.cat(labels)
            metric = metric_fn(predictions, labels)
            return metric


                    
                    
                    
                    
                    
                    
                    
                    
                
            
        