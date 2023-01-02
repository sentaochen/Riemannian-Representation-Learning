

def lr_schedule(optimizer, epoch):
    def lr_decay(LR, n_epoch, e):
        return LR / (1 + 10 * e / n_epoch) ** 0.75

    for i in range(len(optimizer.param_groups)):
        if i < len(optimizer.param_groups) - 1:
            optimizer.param_groups[i]['lr'] = lr_decay(LEARNING_RATE, N_EPOCH, epoch)
        else:
            optimizer.param_groups[i]['lr'] = lr_decay(LEARNING_RATE, N_EPOCH, epoch) * 10

# change by iteration
def inv_lr_scheduler(optimizer, iter_num, lr_mult, init_lr=0.01, gamma=0.0001,
                     power=0.75):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (1 + gamma * iter_num) ** (- power)
    
    for idx, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr * lr_mult[idx]
        
    return lr

def multi_step_lr_scheduler(optimizer, iter_num, lr_mult, filter_steps=[1,1001,1001],init_lr=0.01, gamma=0.0001,
                     power=0.75):
    lr = init_lr
    if iter_num <  filter_steps[-1]:
        lr = init_lr *  (1 + gamma * (iter_num % (filter_steps[1] - filter_steps[0]))) ** (- power)
    else:
        lr = init_lr * (1 + gamma * (iter_num - filter_steps[-1])) ** (- power)
    
    for idx, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr * lr_mult[idx]
        
    return lr