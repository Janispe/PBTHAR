import torch
from torch.utils.data import DataLoader
import numpy as np
from dataloaders import data_set
from sklearn.metrics import confusion_matrix


from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt
import seaborn as sns
from dataloaders.augmentation import RandomAugment, mixup_data

import matplotlib.pyplot as plt



def _get_data(args, data, flag="train", weighted_sampler = False):
    if flag == 'train':
        shuffle_flag = True 
    else:
        shuffle_flag = False

    data  = data_set(args,data,flag)


    if flag == "train":
        if args.mixup_probability < 1 or args.random_augmentation_prob<1:
            random_aug =RandomAugment(args.random_augmentation_nr, args.random_augmentation_config, args.max_aug )
            def collate_fn(batch):                
                if (args.random_aug_first >= np.random.uniform(0,1,1)[0]):
                    batch_x1 = []
                    batch_x2 = []
                    batch_y  = []
                    for x,y,z in batch:

                        if np.random.uniform(0,1,1)[0]>=args.random_augmentation_prob:
                            batch_x1.append(random_aug(x))
                        else:
                            batch_x1.append(x)
                        batch_x2.append(y)
                        batch_y.append(z)

                    batch_x1 = torch.tensor(np.concatenate(batch_x1, axis=0))
                    batch_x2 = torch.tensor(batch_x2)
                    batch_y = torch.tensor(batch_y)
                    
                    if np.random.uniform(0,1,1)[0] >= args.mixup_probability:

                        batch_x1 , batch_y = mixup_data(batch_x1 , batch_y,   args.mixup_alpha,  argmax = args.mixup_argmax)
                        #print("Mixup",batch_x1.shape,batch_y.shape)
                    #else:
                        #print(batch_x1.shape,batch_y.shape)
                    batch_x1 = torch.unsqueeze(batch_x1,1)
                    return batch_x1,batch_x2,batch_y
                else:
                    batch_x1 = []
                    batch_x2 = []
                    batch_y  = []
                    for x,y,z in batch:
                        batch_x1.append(x)
                        batch_x2.append(y)
                        batch_y.append(z)
                    
                    batch_x1 = torch.tensor(np.concatenate(batch_x1, axis=0))
                    batch_x2 = torch.tensor(batch_x2)
                    batch_y = torch.tensor(batch_y)

                    if np.random.uniform(0,1,1)[0] >= args.mixup_probability:
                        batch_x1 , batch_y = mixup_data(batch_x1 , batch_y,   args.mixup_alpha,  argmax = args.mixup_argmax)

                    batch_x1 = batch_x1.detach().cpu().numpy()

                    batch_x1_list = []
                    for x in batch_x1:
                        x = x[np.newaxis]
                        if np.random.uniform(0,1,1)[0]>=args.random_augmentation_prob:
                            batch_x1_list.append(random_aug(x))
                        else:
                            batch_x1_list.append(x)
                    #batch_x1 = torch.tensor(np.concatenate(batch_x1, axis=0))
                    batch_x1 = torch.tensor(batch_x1)

                    batch_x1 = torch.unsqueeze(batch_x1,1)

                    return batch_x1,batch_x2,batch_y


                    
        else:
            collate_fn = None
    else:
        collate_fn = None

    if weighted_sampler and flag == 'train':

        sampler = WeightedRandomSampler(
            data.act_weights, len(data.act_weights)
        )

        data_loader = DataLoader(data, 
                                batch_size   =  args.batch_size,
                                #shuffle      =  shuffle_flag,
                                num_workers  =  0,
                                sampler=sampler,
                                drop_last    =  False,
                                collate_fn = collate_fn)
    else:
        data_loader = DataLoader(data, 
                                batch_size   =  args.batch_size,
                                shuffle      =  shuffle_flag,
                                num_workers  =  0,
                                drop_last    =  False,
                                collate_fn   = collate_fn)
    return data_loader

def validation(args, model, data_loader, criterion, device, index_of_cv=None, selected_index = None):
    model.eval()
    total_loss = []
    preds = []
    trues = []
    with torch.no_grad():
        for i, (batch_x1,batch_x2,batch_y) in enumerate(data_loader):

            if "cross" in args.model_type:
                batch_x1 = batch_x1.double().to(device)
                batch_x2 = batch_x2.double().to(device)
                batch_y = batch_y.long().to(device)
                # model prediction
                if args.output_attention:
                    outputs = model(batch_x1,batch_x2)[0]
                else:
                    outputs = model(batch_x1,batch_x2)
            else:
                if selected_index is None:
                    batch_x1 = batch_x1.double().to(device)
                else:
                    batch_x1 = batch_x1[:, selected_index.tolist(),:,:].double().to(device)
                batch_y = batch_y.long().to(device)

                # model prediction
                if args.output_attention:
                    outputs = model(batch_x1)[0]
                else:
                    outputs = model(batch_x1)


            pred = outputs.detach()#.cpu()
            true = batch_y.detach()#.cpu()

            loss = criterion(pred, true) 
            total_loss.append(loss.cpu())
            
            preds.extend(list(np.argmax(outputs.detach().cpu().numpy(),axis=1)))
            trues.extend(list(batch_y.detach().cpu().numpy()))   
            
    total_loss = np.average(total_loss)
    acc = accuracy_score(preds,trues)
    #f_1 = f1_score(trues, preds)
    f_w = f1_score(trues, preds, average='weighted')
    f_macro = f1_score(trues, preds, average='macro')
    f_micro = f1_score(trues, preds, average='micro')
    if index_of_cv:
        cf_matrix = confusion_matrix(trues, preds)
        #with open("{}.npy".format(index_of_cv), 'wb') as f:
        #    np.save(f, cf_matrix)
        plt.figure()
        sns.heatmap(cf_matrix, annot=True)
        #plt.savefig("{}.png".format(index_of_cv))
    model.train()

    return total_loss,  acc, f_w,  f_macro, f_micro#, f_1

def trainstep(model, optimizer, train_loader, criterion, device):
    model.train()
    for i, (batch_x1,batch_x2,batch_y) in enumerate(train_loader):
        batch_x1 = batch_x1.double().to(device)
        batch_y = batch_y.long().to(device)

        outputs = model(batch_x1)
        loss = criterion(outputs, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()