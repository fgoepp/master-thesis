from tqdm import tqdm

import torch

def train(model, 
          optimizer,
          trainloader,
          testloader,
          device,
          n_epoch=25,
          eps=0.05,
          verbose=True):
    ''' train a model with cross entropy loss
    '''
    model.to(device)
    ce_loss = torch.nn.CrossEntropyLoss()
    for epoch in range(n_epoch):
        # train
        model.train()
        train_loss = 0
        train_zero_one_loss = 0
        dataiter = trainloader
        if verbose:
            dataiter = tqdm(trainloader)
        for img, label in dataiter:
            img, label = img.to(device), label.to(device)
            pred = model(img)
            optimizer.zero_grad()
            loss = ce_loss(pred, label)
            loss.backward()
            train_loss += loss.item()  
            train_zero_one_loss += (pred.softmax(dim=1).argmax(dim=1) != label).sum().item()
            optimizer.step()
        train_zero_one_loss = train_zero_one_loss / len(trainloader.dataset)
        if verbose:
            print('Train Error: ', train_zero_one_loss)
        # test error
        model.eval()
        if testloader is not None:
            test_zero_one_loss = 0
            with torch.no_grad():
                for img, label in testloader:
                    img, label = img.to(device), label.to(device)
                    pred = model(img)
                    test_zero_one_loss += (pred.softmax(dim=1).argmax(dim=1) != label).sum().item()   
            test_zero_one_loss = test_zero_one_loss / len(testloader.dataset)
            if verbose:
                print('Test Error: ', test_zero_one_loss)
            if test_zero_one_loss < eps:
                break
    return train_zero_one_loss, test_zero_one_loss