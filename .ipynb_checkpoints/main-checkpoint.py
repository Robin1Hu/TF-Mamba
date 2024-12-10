import os
import time
from src.__init__ import *

def main():
    start_time = time.time()
    args = get_config()
    args.log_dir = './experiments/{}/'.format(args.model)
    init_seed(args.seed) # Set random seed
    device = torch.device(args.device) # Set device
    args.data_path, args.adj_path, args.node_num = get_dataset_info(args.dataset)
    args.logger.info('Adj path: ' + args.adj_path)

    # Load adjacency matrix
    args.adj_mx = load_adj_from_numpy(args.adj_path)
    if args.nor_adj:
        args.adj_mx = normalize_adj_mx(args.adj_mx, args.adj_type)
    args.supports = [torch.tensor(i).to(args.device) for i in args.adj_mx]

    # Load dataset
    args.loss_fn = masked_mae
    args.dataloader, args.scaler = load_dataset(args.data_path, args)
    
    args.model = TFMamba(
        node_num = args.node_num,
        input_dim = args.input_dim,
        out_dim = args.output_dim,
        device = args.device,
        horizon = args.horizon,
    )
    
    engine = BaseEngine(device = args.device,
                        model = args.model,
                        dataloader = args.dataloader,
                        scaler = args.scaler,
                        sampler = None,
                        loss_fn = masked_mae,
                        lrate = args.lrate,
                        optimizer = optimizer,
                        scheduler = scheduler,
                        clip_grad_value = args.clip_grad_value,
                        max_epochs = args.max_epochs,
                        patience = args.patience,
                        log_dir = args.log_dir,
                        logger = args.logger
                        seed = args.seed
                       )
    
    print_trainable_parameters(engine.model)
    train_time = time.time()
    if args.mode == 'train':
        engine.train()
    else:
        engine.evaluate(args.mode)
    
    if args.save:
        if not os.path.exists('./save'):
            os.makedirs('./save')
        torch.save(engine.model.state_dict(), './save/'+args.save)

    print(args.model,args.mode," finished!! thank you!!")
    end_time = time.time()
    print("total run time: {} s".format(end_time - start_time))
    print("total train time: {} s".format(end_time - train_time))
    
if __name__ == "__main__":
    main()