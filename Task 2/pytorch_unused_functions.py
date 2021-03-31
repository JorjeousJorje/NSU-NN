def compute_accuracy(model, loader, device):
    # What is the purpose of that line
    model.eval() 
    
    with torch.no_grad():
        correct_samples = 0
        total_samples = 0
        
        for i_step, (val_x, val_y) in enumerate(loader):
            val_x = val_x.to(device)
            val_y = val_y.to(device)
            prediction = model(val_x)  
            
            indices = torch.argmax(prediction, dim=1)
            correct_samples += torch.sum(indices == val_y)
            total_samples += val_y.shape[0]
        
        val_accuracy = float(correct_samples) / total_samples
        return val_accuracy
    
def train_model(model, train_loader, val_loader, loss, optimizer, num_epochs, device):    
    loss_history = []
    train_history = []
    val_history = []
    model.to(device)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    for epoch in range(num_epochs):
        model.train() # Enter train mode
        
        loss_accum = 0
        correct_samples = 0
        total_samples = 0
        for i_step, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            prediction = model(x)
            loss_value = loss(prediction, y)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            
            indices = torch.argmax(prediction, dim=1)
            # torch.sum(indices == y) sum of 'True'
            correct_samples += torch.sum(indices == y)
            total_samples += y.shape[0]
            
            loss_accum += loss_value
        ave_loss = loss_accum / (i_step + 1)
        train_accuracy = float(correct_samples) / total_samples
        val_accuracy = compute_accuracy(model, val_loader, device)
        
        scheduler.step()
        
        loss_history.append(float(ave_loss))
        train_history.append(train_accuracy)
        val_history.append(val_accuracy)
        
        print("Average loss: %f, Train accuracy: %f, Val accuracy: %f" % (ave_loss, train_accuracy, val_accuracy))
        
    return loss_history, train_history, val_history