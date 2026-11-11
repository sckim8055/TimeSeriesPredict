"""원본 utils/tools.py 에서 학습 루프가 쓰는 두 개(adjust_learning_rate, EarlyStopping)만 그대로 가져옴. 수정 없음."""
import numpy as np
import torch


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'dtype1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'dtype2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type-e':
        lr_adjust = {epoch: 0.2}
        if epoch <= 20:
             lr_adjust[epoch] = 0.15
        elif lr_adjust[epoch] <= 0.001:
            lr_adjust[epoch] = 0.001
        else:
            lr_adjust = {epoch: args.learning_rate * (0.95 ** (epoch // 1))}

    elif args.lradj == '1':
        lr_adjust = {epoch: args.learning_rate * (0.98 ** (epoch // 1))}
        if lr_adjust[epoch] <= 0.000001:
            lr_adjust[epoch] = 0.000001

    elif args.lradj == '2':
        lr_adjust = {
            0: 0.02, 10: 0.01, 30: 0.005, 50: 0.001, 80: 0.0005, 100: 0.0001, 130: 0.00005, 150: 0.00001, 180:0.000005, 200:0.000001
        }
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch <
                     10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch <
                     15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch <
                     25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch <
                     5 else args.learning_rate*0.1}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))
    else:
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
    return lr


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path, model_id):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, model_id)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, model_id)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, model_id):
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + model_id +'_best_checkpoint.pth')
        self.val_loss_min = val_loss


