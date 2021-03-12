import argparse
import numpy as np

class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # self.device = 'cuda'

        self.lr = 1e-3
        self.max_split_iters = 3
        self.verbose = True
        self.epsilon = 1e-2
        self.granularity = 5
        self.init_args()

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def init_args(self):
        #################### Experiment Settings ####################
        self.parser.add_argument('--data', default='cifar10', help='[mnist/cifar10]')
        self.parser.add_argument('--method', default='none', help='[none/exact/fast/firefly/random]')
        self.parser.add_argument('--model', default='mobile', help='[vgg19/mobile]')
        self.parser.add_argument('--batch_size', default=128, type=int, help='batch size')
        self.parser.add_argument('--granularity', default=3, type=int, help='granularity')
        self.parser.add_argument('--seed', default=0, type=int, help='seed')
        self.parser.add_argument('--dim_hidden', default=16, type=int, help='initial hidden dimension')
        self.parser.add_argument('--grow_every', default=10, type=int, help='grow per # epoch')
        self.parser.add_argument('--warmup', default=50, type=int, help='warmup # epoch')
        self.parser.add_argument('--opt', default='SGD', help='which optimizer to use')
        self.parser.add_argument('--beta1', default=0.9, type=float, help='beta1 in Adam')
        self.parser.add_argument('--beta2', default=0.999, type=float, help='beta2 in Adam')
        self.parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
        self.parser.add_argument('--weight_decay', default=1e-4, type=float, help='l2 regularization')
        self.parser.add_argument('--momentum', default=0.9, type=float, help='sgd momentum')
        self.parser.add_argument('--n_rounds', default=10, type=int, help='number of growing round')
        self.parser.add_argument('--n_elites', default=5, type=int, help='number of added neurons per split')
        self.parser.add_argument('--n_epochs', default=160, type=int, help='number of training epochs in each round before split')
        self.parser.add_argument('--grow_ratio', default=0.35, type=float, help='number of added neurons per split')
        self.parser.add_argument('--alpha', default=0.3, type=float, help='alpha')
        self.parser.add_argument('--load_round', default=0, type=int, help='round to load')
        self.parser.add_argument('--gpu', default='0', type=str, help="cuda devices")
        self.parser.add_argument('--save', default='default', type=str, help="save folder path")

        # #################### Experiment Summary ####################
        # self.parser.add_argument('--summary', default=False, dest='summary', action='store_true')
        # self.parser.add_argument('--load', default="default", type=str, help="load models folder path")
        # self.parser.add_argument('--expname', default="exp", type=str, help="load experiment names")

        args = self.parser.parse_args()

        self.dataset = args.data
        self.method = args.method
        self.model = args.model
        self.grow_every = args.grow_every
        self.granularity = args.granularity
        self.warmup = args.warmup
        self.optimizer = args.opt
        self.batch_size = args.batch_size
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.alpha = args.alpha
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.n_rounds = args.n_rounds
        self.n_elites = args.n_elites
        self.n_epochs = args.n_epochs
        self.dim_hidden = args.dim_hidden
        self.momentum = args.momentum
        self.grow_ratio = args.grow_ratio
        self.seed = args.seed
        self.load_round = args.load_round
        self.resume = False if args.load_round == 0 else True
        self.device = f"cuda:{args.gpu}"
        self.save = args.save

        # self.summary = args.summary
        # self.load = args.load
        # self.expname = args.expname
        
        if self.dataset == 'mnist':
            self.dim_input = 784
            self.dim_output = 10
        elif self.dataset == 'cifar10':
            self.dim_input = (3, 32, 32)
            self.dim_output = 10
        elif self.dataset == 'cifar100':
            self.dim_input = (3, 32, 32)
            self.dim_output = 100

        # self.exp_name = "exp_%s_%s_initdim%d_seed%d_grow%.3f_gra%d_alpha3_new" % (
        #     self.dataset,
        #     self.method,
        #     self.dim_hidden,
        #     self.seed,
        #     self.grow_ratio,
        #     self.granularity)

        #########################################################
        self.logpath = "checkpoint/summary_%s.log" %self.save
        # create file handler which logs even debug messages
        import logging
        log = logging.getLogger()
        log.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        fh = logging.FileHandler(self.logpath)

        formatter = logging.Formatter('%(asctime)s - %(message)s')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        log.addHandler(fh)
        log.addHandler(ch)
        print("logging file created.")
        self.log = log
        #########################################################

        self.verbose = True
        if self.verbose:
            log.info("="*80)
            log.info("[INFO] -- Experiment Configs --")
            log.info("       1. data & split method")
            log.info("          dataset: %s" % self.dataset)
            log.info("          split method: %s" % self.method)
            log.info("       2. training")
            log.info("          lr: %10.4f" % self.lr)
            log.info("          optimizer: %s" % str(self.optimizer))
            log.info("          rounds: %d" % self.n_rounds)
            log.info("          epochs: %d" % self.n_epochs)
            log.info("          grow ratio: %10.4f" % self.grow_ratio)
            log.info("       3. model")
            log.info("          dim_input: %s" % str(self.dim_input))
            log.info("          dim_hidden: %d" % self.dim_hidden)
            log.info("          dim_output: %d" % self.dim_output)
            log.info("          model: %s" % self.model)
            log.info("       4. device")
            log.info("          %s" % str(self.device))
            
            log.info("="*80)
