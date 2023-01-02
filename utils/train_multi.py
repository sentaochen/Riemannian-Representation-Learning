import torch
import torch.nn as nn

import time
import math

from utils.lr_schedule import inv_lr_scheduler, multi_step_lr_scheduler
from utils.eval import test, predict
from utils import globalvar as gl


def finetune_for_mutil_UDA_sep(args, model, optimizer, dataloaders):
    DEVICE = gl.get_value('DEVICE')
    check_path = gl.get_value('check_path')
    record_file = gl.get_value('record_file')

    criterion = nn.CrossEntropyLoss()
    start_time = time.time()

    data_iter = []
    for data_loader in dataloaders['src_pretrain']:
        data_iter.append(iter(data_loader))
    
    save_log_step = 500

    for step in range(1, args.presteep):
        epoch_time = time.time()
        model.train()
        batch_loss = 0
        for i in range(len(data_iter)):
            if step > 0 and (step - 1) % len(dataloaders['src_pretrain'][i]) == 0:
                data_iter[i] = iter(dataloaders['src_pretrain'][i])
        inputs, labels = next(data_iter[0])
        for i in range(len(data_iter)):
            if i != 0:
                inp, lab = next(data_iter[i])
                inputs = torch.cat([inputs, inp], dim=0)
                labels = torch.cat([labels, lab], dim=0)
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs,_ = model(inputs)
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        batch_loss += loss.item()
        
        if step > 0 and step % args.log_interval == 0:
            print('Step: [{:02d}/{:02d}]: loss:{:.4f}'.format(step, args.presteep, batch_loss))

        if step > 0 and step % save_log_step == 0:
            # print('Learning rate: {:.8f}'.format(current_lr))
            epoch_loss = batch_loss / len(inputs)
            print('Step: [{:02d}/{:02d}]---{}, loss: {:.6f}'.format(step, args.presteep, 'src_pretrain', epoch_loss))
            print('{:d} step train time: {:.1f}s'.format(save_log_step, time.time()-epoch_time))
            test_time = time.time()
            loss_tar,acc_tar = test(model, dataloaders['tar_test'])
            print('one test time: {:.1f}s'.format(time.time()-test_time))

            if args.save_check:
                torch.save(model.state_dict(), '{}/best(PT)_{}_{}-{}.pth'.format(check_path, args.net, args.source, step//500))


            print('record {}'.format(record_file))
            with open(record_file, 'a') as f:
                f.write('Step {} acc_tar {:.4f}\n'.format(step, acc_tar))
            
            seconds = time.time() - start_time
            print('Step {} cost time: {}h {}m {:.0f}s\n'.format(step, seconds//3600, seconds%3600//60, seconds%60))
    
    time_pass = time.time() - start_time
    print('Training complete in {}h {}m {:.0f}s\n'.format(time_pass//3600, time_pass%3600//60, time_pass%60))

    torch.save(model.state_dict(), '{}/best(PT)_{}_{}.pth'.format(check_path, args.net, args.source))
    print('Final model saved!')



def train_for_Multi(args, model, optimizer, dataloaders, domain_labels):
    DEVICE = gl.get_value('DEVICE')
    check_path = gl.get_value('check_path')
    record_file = gl.get_value('record_file')

    _,acc_tar = test(model, dataloaders['tar_test'])
    print('Initial model:  acc_tar:{:.4f}'.format(acc_tar))

    with open(record_file, 'a') as f:
        f.write('Initial model:  acc_tar:{:.4f}\n'.format(acc_tar))
    
    criterion = torch.nn.CrossEntropyLoss().to(DEVICE)

    if args.early:
        best_acc = 0
        counter = 0

    tar_data_ul = iter(dataloaders['tar_train_ul'])
    src_label_data_list = [None for i in range(len(dataloaders['src_train_l_list']))]
    for i in range(len(dataloaders['src_train_l_list'])):
        src_label_data_list[i] = iter(dataloaders['src_train_l_list'][i])

    start_time = time.time()
    avg_div_loss = 0

    for step in range(1, args.steps + 1):
        model.train()

        current_lr = inv_lr_scheduler(optimizer, step, args.lr_mult)
        lambd = 2 / (1 + math.exp(-10 * step / args.lam_step)) - 1
        
        label_x_list, label_y_list, label_l_list = [], [], []

        for src_data_l in src_label_data_list:
            inputs_l, labels_l, domain_l = next(src_data_l)
            label_x_list.append(inputs_l)
            label_y_list.append(labels_l)
            label_l_list.append(domain_l)
        
        labeled_x = torch.cat(label_x_list)
        labeled_y = torch.cat(label_y_list)
        labeled_l = torch.cat(label_l_list)

        unlabeled_x, _,  unlabeled_l = next(tar_data_ul)
    
        s_img, s_label, s_domain_label = labeled_x.to(DEVICE), labeled_y.to(DEVICE), labeled_l.to(DEVICE)
        t_img,t_domain_label = unlabeled_x.to(DEVICE), unlabeled_l.to(DEVICE)

        s_output, div_loss = model(source=s_img, target=t_img, source_label=s_label, 
                                        source_domain_label=s_domain_label, target_domain_label=t_domain_label)
        cls_loss = criterion(s_output, s_label)

        lambd = 2 / (1 + math.exp(-10 * step / args.lam_step)) - 1
        loss = cls_loss  + args.gamma * div_loss  * lambd

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_div_loss += div_loss

        if step > 0 and step % args.log_interval == 0:
            print('Learning rate: {:.8f}'.format(current_lr))
            print('Step: [{}/{}]: lambd:{}, div_loss:{:.4f}, cls_loss:{:.4f}, total_loss:{:.4f}'.format(step, args.lam_step, lambd, div_loss, cls_loss, loss))
 
            
        if step > 0 and step % args.save_interval == 0:
            print('{} step train time: {:.1f}s'.format(args.save_interval, time.time()-start_time))
            test_time = time.time()
            _,acc_tar = test(model, dataloaders['tar_test'])
            print('one test time: {:.1f}s'.format(time.time()-test_time))
            
            print('record {}'.format(record_file))
            with open(record_file, 'a') as f:
                f.write('step {} avg_div_loss {:.4f}  acc_tar {:.4f} \n'.format(step, avg_div_loss/args.save_interval, acc_tar))
                avg_div_loss = 0
            
            if step >= args.start_update_step and step % args.update_interval == 0 :
                dataloaders['src_train_l_list'][0].batch_sampler.label_sampler.set_batch_num(args.update_interval)
                for i in range(len(dataloaders['src_train_l_list'])):
                    src_label_data_list[i] = iter(dataloaders['src_train_l_list'][i])
                pseudo_labels = predict(model, dataloaders['tar_test'])
                dataloaders['tar_train_ul'].dataset.update_pseudo_labels(pseudo_labels)
                tar_data_ul = iter(dataloaders['tar_train_ul'])

            if args.early:
                if step >= 20000 and acc_tar > best_acc:
                    best_acc = acc_tar
                    counter = 0

                    torch.save(model.state_dict(), '{}/best_Multi_{}_{}_to_{}.pth'.format(check_path, args.net, args.source, args.target))
                else:
                    counter += 1
                    if counter > args.patience:
                        print('early stop! training_step:{}'.format(step))
                        break
            

            seconds = time.time() - start_time
            print('{} step cost time: {}h {}m {:.0f}s\n'.format(step, seconds//3600, seconds%3600//60, seconds%60))

    with open(record_file, 'a') as f:
        f.write('best_acc: {:.4f} \n\n'.format(best_acc))
    # complete training
    time_pass = time.time() - start_time
    print('Training {} step complete in {}h {}m {:.0f}s\n'.format(step, time_pass//3600, time_pass%3600//60, time_pass%60))
    print('Training_step:{}, best_acc_tar:{}'.format(step, best_acc))
