import os
import yaml
# import argparse

def parse_configure():
    # parser = argparse.ArgumentParser(description='SSLRec')
    # parser.add_argument('--model', type=str, help='Model name')
    # parser.add_argument('--dataset', type=str, default=None, help='Dataset name')
    # parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
    # parser.add_argument('--cuda', type=str, default='0', help='Device number')
    # args = parser.parse_args()

    device = 'cpu'
    model = 'dcrec_seq'
    dataset = 'sports'

    # if args.device == 'cuda':
    #     os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    if model == None:
        raise Exception("Please provide the model name through --model.")
    model_name = model.lower()
    if not os.path.exists('./config/modelconf/{}.yml'.format(model_name)):
        raise Exception("Please create the yaml file for your model first.")

    with open('./config/modelconf/{}.yml'.format(model_name), encoding='utf-8') as f:
        config_data = f.read()
        configs = yaml.safe_load(config_data)

        # model name
        configs['model']['name'] = configs['model']['name'].lower()

        # grid search
        if 'tune' not in configs:
            configs['tune'] = {'enable': False}

        # gpu device
        configs['device'] = device

        # dataset
        if dataset is not None:
            configs['data']['name'] = dataset

        # log
        if 'log_loss' not in configs['train']:
            configs['train']['log_loss'] = True

        # early stop
        if 'patience' in configs['train']:
            if configs['train']['patience'] <= 0:
                raise Exception("'patience' should be greater than 0.")
            else:
                configs['train']['early_stop'] = True
        else:
            configs['train']['early_stop'] = False



        return configs

configs = parse_configure()
