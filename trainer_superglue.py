import os
import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
from torch.distributions import Categorical
import random
from collections import Counter
from copy import deepcopy as cp


def simple_accuracy(preds, labels):
    return accuracy_score(preds, labels)

def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average="macro")  
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }

def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }

def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name in ["boolq", "copa", "rte", "wic", "wsc"]:
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "cb":
        return acc_and_f1(preds, labels)
    else:
        raise KeyError(task_name)

def evaluate_model(task_loader, task_name, model, num_labels, device, split='eval', model_type='teacher'):
    assert split in ['eval', 'test']
    model.eval()   

    if task_name.lower() in ['sts-b']:
        loss_fn = nn.MSELoss(reduction='mean') 
    else:
        loss_fn = nn.CrossEntropyLoss()
    
    val_dataloader = task_loader[split]['loader']   
    val_data = task_loader[split]['dataset']  

    all_labels = []
    all_preds = []

    total_val_loss = 0
    with torch.no_grad():
        for step, batch in enumerate(val_dataloader):
            if split == 'eval':
                input_ids, attention_mask, token_type_ids, labels, images = batch[0], batch[1], batch[2], batch[3], batch[4] 
                labels = labels.to(device)
            else:
                input_ids, attention_mask, token_type_ids, images = batch[0], batch[1], batch[2], batch[3] 

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            images = images.to(device)

            if model_type == 'teacher':
                out = model.forward(text="", task_name=task_name.lower(), image=images)
            else: 
                out = model.forward(src=input_ids, task_name=task_name.lower(), mask=attention_mask, token_type_ids=token_type_ids)
            
            student_out = out[0]


            if task_name.lower() not in ['sts-b']:
                student_out_ = student_out.argmax(-1)
            else:
                student_out_ = student_out[:,0].clip(0,5)

            if task_name.lower() != 'sts-b':
                all_preds += student_out_.cpu().numpy().astype(int).tolist()
                if split == 'eval':
                    all_labels += labels.cpu().numpy().astype(int).tolist()
            else:
                all_preds += student_out_.cpu().numpy().tolist()
                if split == 'eval':
                    all_labels += labels.cpu().numpy().tolist()

            if split == 'eval':
                if task_name.lower() in ['sts-b']:
                    loss = loss_fn(student_out.view(-1), labels.view(-1)) 
                else:
                    loss = loss_fn(student_out, labels) 

                total_val_loss += loss.item() * labels.shape[0]

    total_val_loss = total_val_loss/len(val_data)

    if split == 'eval':
        out = compute_metrics(task_name.lower(), all_preds, all_labels)
    else:
        out = {}

    out['task'] = task_name
    if split == 'eval':
        out['val_loss'] = total_val_loss

    return all_preds, out, all_labels

def trainer(args, teacher_model, student_model, task_loader, num_label, task_name):

    train_dataloader = task_loader['train']['loader']
    train_data = task_loader['train']['dataset']
    
    if task_name.lower() in ['sts-b']:
        loss_fn = nn.MSELoss(reduction='mean') 
    else:
        loss_fn = nn.CrossEntropyLoss()


    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1


    teacher_model.to(device)
    student_model.to(device)
    t_total = args.train_dataloader_size * args.epochs

    t_optimizer = torch.optim.AdamW(teacher_model.parameters(), lr=args.teacher_learning_rate)
    s_optimizer = torch.optim.AdamW(student_model.parameters(), lr=args.learning_rate)
    
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        teacher_model, t_optimizer = amp.initialize(teacher_model, t_optimizer, opt_level=args.fp16_opt_level)
        student_model, s_optimizer = amp.initialize(student_model, s_optimizer, opt_level=args.fp16_opt_level)

    if args.n_gpu > 1:
        teacher_model = torch.nn.DataParallel(teacher_model)
        student_model = torch.nn.DataParallel(student_model)

    if args.local_rank != -1:
        teacher_model = torch.nn.parallel.DistributedDataParallel(teacher_model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
        student_model = torch.nn.parallel.DistributedDataParallel(student_model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    
    best_score = 0
    best_score_student = 0

    ########################################################
    #############     training loop              ###########  
    ########################################################

    if args.run_type != 'base':
        teacher_model.train()
        for epoch in range(args.epochs):

            teacher_model.train()
            if epoch < args.teacher_epochs:

                total_training_task_loss = 0
                for step, batch in enumerate(train_dataloader):
                    t_optimizer.zero_grad()

                    input_ids, attention_mask, token_type_ids, labels, images = batch[0], batch[1], batch[2], batch[3], batch[4] #, batch[5]
                    input_ids = input_ids.to(device)
                    images = images.to(device)
                    labels = labels.to(device)

                    out = teacher_model.forward(text="", task_name=task_name.lower(), image=images)
                    teacher_out = out[0]
                    teacher_out2 = out[1]

                    if task_name.lower() in ['sts-b']:
                        loss = loss_fn(teacher_out.view(-1), labels.view(-1)) + args.gamma * loss_fn(teacher_out2.view(-1), labels.view(-1)) 
                    else:
                        loss = loss_fn(teacher_out, labels) + args.gamma * loss_fn(teacher_out2, labels) 
                    
                    if args.fp16:
                        with amp.scale_loss(loss, t_optimizer) as scaled_loss:
                            scaled_loss.backward()
                        torch.nn.utils.clip_grad_norm_(amp.master_params(t_optimizer), args.max_grad_norm)
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(teacher_model.parameters(), args.max_grad_norm)

                    total_training_task_loss += loss.item() * labels.shape[0]

                    t_optimizer.step()

                total_training_task_loss = total_training_task_loss/len(train_data)
                print('| task {} | epoch {:3d} | training loss {:5.3f}'.format(
                        task_name, epoch, total_training_task_loss))
                
                preds, out, _ = evaluate_model(task_loader, task_name, teacher_model, num_label, device, split='eval')
                print ("Epoch {}, Teacher Result: {}".format(epoch, out))

                if 'mcc' in out:
                    score = out['mcc']
                elif 'f1' in out:
                    score = out['f1']
                elif 'acc' in out:
                    score = out['acc']
                elif 'spearmanr' in out:
                    score = out['spearmanr']

                models_dir = "./models"
                if not os.path.exists(models_dir):
                    os.makedirs(models_dir)

                if score > best_score:
                    torch.save(teacher_model.state_dict(), "./models/teacher_{}_{}.ckpt".format(args.task, args.seed))
                    best_score = score

                if args.wandb_logging == True:
                    wandb.log({"Teacher Training Loss": total_training_task_loss})

                if args.wandb_logging == True:
                    for key in out:
                        if key != 'task':
                            wandb.log({"Teacher " + key: out[key]})

                ###########################
                ## training the student ###
                ###########################
                
                student_model.train()
                teacher_model.eval()

                total_training_loss_student = 0
                total_training_task_loss_student = 0

                for step, batch in enumerate(train_dataloader):
                    s_optimizer.zero_grad()

                    input_ids, attention_mask, token_type_ids, labels, images = batch[0], batch[1], batch[2], batch[3], batch[4] #, batch[5]
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    token_type_ids = token_type_ids.to(device)
                    labels = labels.to(device)
                    images = images.to(device)

                    out = student_model.forward(src=input_ids, task_name=task_name.lower(), mask=attention_mask, token_type_ids=token_type_ids)
                    student_out = out[0]
                    student_out2 = out[1]
                    student_proj = out[2] 

                    with torch.no_grad():
                        out = teacher_model.forward(text="", task_name=task_name.lower(), image=images)
                        teacher_out = out[0]  
                        teacher_out2 = out[1] 
                        teacher_proj = out[2] 

                    if task_name.lower() in ['sts-b']:
                        loss = loss_fn(student_out.view(-1), labels.view(-1)) + args.gamma * loss_fn(student_out2.view(-1), labels.view(-1)) 
                    else:
                        loss = loss_fn(student_out, labels) + args.gamma * loss_fn(student_out2, labels) 

                    if task_name.lower() in ['sts-b']:
                        soft_loss = F.mse_loss(teacher_out, student_out) + args.gamma * F.mse_loss(teacher_out2, student_out2)
                    else:
                        T = args.temperature
                        soft_targets = F.softmax(teacher_out / T, dim=-1)
                        probs = F.softmax(student_out / T, dim=-1)
                        soft_targets2 = F.softmax(teacher_out2 / T, dim=-1)
                        probs2 = F.softmax(student_out2 / T, dim=-1)
                        soft_loss = (F.mse_loss(soft_targets, probs) + args.gamma * F.mse_loss(soft_targets2, probs2))* T * T

                    if args.beta == 0: 
                        pkd_loss = torch.zeros_like(soft_loss)
                        
                    else:    
                        if args.loss_type in ['cosine','euclid']:
                            x = teacher_proj.mean(0)               
                            y = student_proj.mean(0)        
                            if args.loss_type == 'cosine':
                                pkd_loss = torch.dot(x,y)/(torch.norm(x,2)*torch.norm(y,2))
                                pkd_loss = 1-pkd_loss
                            else:
                                pkd_loss = F.mse_loss(x,y)
                        else:
                            pkd_loss = F.mse_loss(teacher_proj, student_proj, reduction='mean')

                    total_loss = args.alpha * soft_loss + (
                            1 - args.alpha) * loss + args.beta * pkd_loss
                    
                    if args.fp16:
                        with amp.scale_loss(total_loss, s_optimizer) as scaled_loss:
                            scaled_loss.backward()
                        torch.nn.utils.clip_grad_norm_(amp.master_params(s_optimizer), args.max_grad_norm)
                    else:
                        total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.max_grad_norm)

                    s_optimizer.step()
                    total_training_task_loss_student += loss.item() * labels.shape[0]
                    total_training_loss_student += total_loss.item() * labels.shape[0]

            total_training_loss_student = total_training_loss_student/len(train_data)
            total_training_task_loss_student = total_training_task_loss_student/len(train_data)

            print('| task {} | epoch {:3d} | training loss {:5.3f} | training task loss {:5.3f}'.format(
                        task_name, epoch, total_training_loss_student, total_training_task_loss_student))

            if args.wandb_logging == True:
                wandb.log({"Training loss": total_training_loss_student})
                wandb.log({"Training task only loss": total_training_task_loss_student})

            if args.local_rank in [-1, 0]:
                preds, out3, _ = evaluate_model(task_loader, task_name, student_model, num_label, device, split='eval', model_type='student')
                print ("Epoch {}, Student Result Step: {}".format(epoch, out3))

                if args.wandb_logging == True:
                    for key in out3:
                        if key != 'task':
                            wandb.log({"Student " + key: out3[key]})

                if 'mcc' in out3:
                    score = out3['mcc']
                elif 'f1' in out3:
                    score = out3['f1']
                elif 'f1_macro' in out3:
                    score = out3['f1_macro']
                elif 'acc' in out3:
                    score = out3['acc']
                elif 'spearmanr' in out3:
                    score = out3['spearmanr']

                if score > best_score_student:
                    torch.save(student_model.state_dict(), "./models/student_{}_{}_{}_{}_{}.ckpt".format(args.model_type, args.loss_type, args.nlayers, args.task, args.seed))
                    best_score_student = score

        ## loading the best models
        teacher_model.load_state_dict(torch.load("./models/teacher_{}_{}.ckpt".format(args.task, args.seed)))
        student_model.load_state_dict(torch.load("./models/student_{}_{}_{}_{}_{}.ckpt".format(args.model_type, args.loss_type, args.nlayers, args.task, args.seed)))

        preds, out, _ = evaluate_model(task_loader, task_name, teacher_model, num_label, device, split='eval', model_type='teacher')
        print ("Teacher Final Result: {}".format(out))

        if args.wandb_logging == True:
            for key in out:
                if key != 'task':
                    wandb.log({"Teacher Final " + key: out[key]})

        # Test prediction
        all_preds, _, _ = evaluate_model(task_loader, task_name, teacher_model, num_label, device, split='test', model_type='teacher')
        test_pred = pd.DataFrame({'index': range(len(all_preds)), 'prediction': all_preds})

        if task_name.lower() in ['rte']:
            mapping = ["entailment", "not_entailment"]
            test_pred['prediction'] = test_pred['prediction'].apply(lambda x: mapping[x])
        elif task_name.lower() in ['cb']:
            mapping = ["entailment","contradiction", "neutral"]
            test_pred['prediction'] = test_pred['prediction'].apply(lambda x: mapping[x])

        try:
            os.makedirs("test_outputs")
        except:
            pass

        test_pred.to_csv("./test_outputs/"+"teacher_" + task_name + '.tsv', sep='\t', index=False)

        preds, out, _ = evaluate_model(task_loader, task_name, student_model, num_label, device, split='eval', model_type='student')
        print ("Student Final Result: {}".format(out))

        if args.wandb_logging == True:
            for key in out:
                if key != 'task':
                    wandb.log({"Student Final " + key: out[key]})

        preds, out, all_labels = evaluate_model(task_loader, task_name, student_model, num_label, device, split='eval', model_type='student')
        if os.path.exists("./val_outputs/comparison_base_{}.csv".format(task_name)):
            val_pred1 = pd.read_csv("./val_outputs/comparison_base_{}.csv".format(task_name))
            val_pred2 = pd.DataFrame({'index': range(len(preds)), 'prediction': preds})

            val_pred = pd.merge(val_pred1, val_pred2, how='inner')
            val_pred['true label'] = all_labels

            try:
                os.makedirs("./val_outputs/")
            except:
                pass
            
            val_pred.to_csv("./val_outputs/comparison_{}.csv".format(task_name), index=False)

        # Test prediction
        all_preds, _, _ = evaluate_model(task_loader, task_name, student_model, num_label, device, split='test', model_type='student')
        test_pred = pd.DataFrame({'index': range(len(all_preds)), 'prediction': all_preds})

        if task_name.lower() in ['rte']:
            mapping = ["entailment", "not_entailment"]
            test_pred['prediction'] = test_pred['prediction'].apply(lambda x: mapping[x])
        elif task_name.lower() in ['cb']:
            mapping = ["entailment","contradiction", "neutral"]
            test_pred['prediction'] = test_pred['prediction'].apply(lambda x: mapping[x])

        test_pred.to_csv("./test_outputs/"+"student_" + task_name + '.tsv', sep='\t', index=False)
        


    ################################################
    #######  training without distillation #########
    ################################################
    
    if args.run_type == 'base':
        ## base run
        best_score = 0
        student_model.train()

        for epoch in range(args.epochs):
            total_training_loss = 0
            total_training_task_loss = 0


            for step, batch in enumerate(train_dataloader):
                if step == 0:
                    print(device)
                s_optimizer.zero_grad()

                input_ids, attention_mask, token_type_ids, labels, images = batch[0], batch[1], batch[2], batch[3], batch[4]
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                token_type_ids = token_type_ids.to(device)
                labels = labels.to(device)
                images = images.to(device)


                out = student_model.forward(src=input_ids, task_name=task_name.lower(), mask=attention_mask, token_type_ids=token_type_ids)
                student_out = out[0]   
                student_out2 = out[1]  
                student_proj = out[2]  

                if task_name.lower() in ['sts-b']:
                    loss = loss_fn(student_out.view(-1), labels.view(-1)) 
                
                else:
                    loss = loss_fn(student_out, labels)

                total_loss = loss
                
                if args.fp16:
                    with amp.scale_loss(total_loss, s_optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(s_optimizer), args.max_grad_norm)
                else:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.max_grad_norm)

                s_optimizer.step()
                total_training_task_loss += loss.item() * labels.shape[0]
                total_training_loss += total_loss.item() * labels.shape[0]

            total_training_loss = total_training_loss/len(train_data)
            total_training_task_loss = total_training_task_loss/len(train_data)

            print('| task {} | epoch {:3d} | training loss {:5.3f} | training task loss {:5.3f}'.format(
                        task_name, epoch, total_training_loss, total_training_task_loss))

            if args.wandb_logging == True:
                wandb.log({"Training loss Student base": total_training_loss})

            if args.local_rank in [-1, 0]:
                preds, out3, _ = evaluate_model(task_loader, task_name, student_model, num_label, device, split='eval', model_type='student')
                print ("Epoch {}, Student base Result : {}".format(epoch, out3))

                if args.wandb_logging == True:
                    for key in out3:
                        if key != 'task':
                            wandb.log({"Student base " + key: out3[key]})

                if 'mcc' in out3:
                    score = out3['mcc']
                elif 'f1' in out3:
                    score = out3['f1']
                elif 'acc' in out3:
                    score = out3['acc']
                elif 'spearmanr' in out3:
                    score = out3['spearmanr']


                if score > best_score:
                    torch.save(student_model.state_dict(), "./models/student_base_{}_{}_{}_{}.ckpt".format(args.model_type, args.nlayers, args.task, args.seed))
                    best_score = score

        try:
            student_model.load_state_dict(torch.load("./models/student_base_{}_{}_{}_{}.ckpt".format(args.model_type, args.nlayers, args.task, args.seed)))
        except:
            pass
        
        preds, out, _ = evaluate_model(task_loader, task_name, student_model, num_label, device, split='eval', model_type='student')
        print ("Student Base Final Result: {}".format(out))

        if args.wandb_logging == True:
            for key in out:
                if key != 'task':
                    wandb.log({"Student Base Final " + key: out[key]})

        preds, out, _ = evaluate_model(task_loader, task_name, student_model, num_label, device, split='eval', model_type='student')
        val_pred1 = pd.DataFrame({'index': range(len(preds)), 'prediction base': preds})

        try:
            os.makedirs("./val_outputs/")
        except:
            pass
        
        val_pred1.to_csv("./val_outputs/comparison_base_{}.csv".format(task_name), index=False)