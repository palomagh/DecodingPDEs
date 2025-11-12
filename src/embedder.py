import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from timeit import default_timer
from functools import partial
from transformers import AutoModel, AutoConfig, SwinForImageClassification, SwinForMaskedImageModeling, RobertaForTokenClassification
from otdd.pytorch.distance import DatasetDistance, FeatureCost

from task_configs import get_data, get_optimizer_scheduler
from utils import conv_init, embedder_init, embedder_placeholder, adaptive_pooler, to_2tuple, set_grad_state, create_position_ids_from_inputs_embeds, l2, MMD_loss


def otdd(feats, ys=None, src_train_dataset=None, exact=True):
    ys = torch.zeros(len(feats)) if ys is None else ys

    if not torch.is_tensor(feats):
        feats = torch.from_numpy(feats).to('cpu')
        ys = torch.from_numpy(ys).long().to('cpu')

    dataset = torch.utils.data.TensorDataset(feats, ys)

    dist = DatasetDistance(src_train_dataset, dataset,
                                    inner_ot_method = 'exact' if exact else 'gaussian_approx',
                                    debiased_loss = True, inner_ot_debiased=True,
                                    p = 2, inner_ot_p=2, entreg = 1e-1, ignore_target_labels = False,
                                    device=feats.device, load_prev_dyy1=None)
                
    d = dist.distance(maxsamples = len(src_train_dataset))
    return d

class wrapper1D(torch.nn.Module):
    def __init__(self, input_shape, output_shape, use_embedder=True, weight='roberta', train_epoch=0, activation=None, target_seq_len=512, drop_out=None, from_scratch=False, random=False, double=False):
        super().__init__()

        self.dense = False
        self.output_raw = True
        self.weight = weight
        self.output_shape = output_shape
        self.double = double

        if isinstance(output_shape, tuple):
            self.dense = True

        if weight == 'gpt2':
            modelname = 'gpt2'
            configuration = AutoConfig.from_pretrained(modelname)
            if drop_out is not None:
                configuration.hidden_dropout_prob = drop_out
                configuration.attention_probs_dropout_prob = drop_out
            self.model = AutoModel.from_pretrained(modelname, config = configuration) if not from_scratch else AutoModel.from_config(configuration)
        
        elif weight == 'gpt2-medium':
            modelname = 'gpt2-medium'
            configuration = AutoConfig.from_pretrained(modelname)
            if drop_out is not None:
                configuration.hidden_dropout_prob = drop_out
                configuration.attention_probs_dropout_prob = drop_out
            self.model = AutoModel.from_pretrained(modelname, config = configuration) if not from_scratch else AutoModel.from_config(configuration)
        
        elif weight == 'gpt2-large':
            modelname = 'gpt2-large'
            configuration = AutoConfig.from_pretrained(modelname)
            if drop_out is not None:
                configuration.hidden_dropout_prob = drop_out
                configuration.attention_probs_dropout_prob = drop_out
            self.model = AutoModel.from_pretrained(modelname, config = configuration) if not from_scratch else AutoModel.from_config(configuration)
        
        elif weight == 'gpt2-xl':
            modelname = 'gpt2-xl'
            configuration = AutoConfig.from_pretrained(modelname)
            if drop_out is not None:
                configuration.hidden_dropout_prob = drop_out
                configuration.attention_probs_dropout_prob = drop_out
            self.model = AutoModel.from_pretrained(modelname, config = configuration) if not from_scratch else AutoModel.from_config(configuration)
        
        elif weight == 'pythia-14m':
            modelname = 'EleutherAI/pythia-14m'
            configuration = AutoConfig.from_pretrained(modelname)
            if drop_out is not None:
                configuration.hidden_dropout_prob = drop_out
                configuration.attention_probs_dropout_prob = drop_out
            self.model = AutoModel.from_pretrained(modelname, config = configuration) if not from_scratch else AutoModel.from_config(configuration)
        
        elif weight == 'pythia-70m':
            modelname = 'EleutherAI/pythia-70m'
            configuration = AutoConfig.from_pretrained(modelname)
            if drop_out is not None:
                configuration.hidden_dropout_prob = drop_out
                configuration.attention_probs_dropout_prob = drop_out
            self.model = AutoModel.from_pretrained(modelname, config = configuration) if not from_scratch else AutoModel.from_config(configuration)
        
        elif weight == 'pythia-160m':
            modelname = 'EleutherAI/pythia-160m'
            configuration = AutoConfig.from_pretrained(modelname)
            if drop_out is not None:
                configuration.hidden_dropout_prob = drop_out
                configuration.attention_probs_dropout_prob = drop_out
            self.model = AutoModel.from_pretrained(modelname, config = configuration) if not from_scratch else AutoModel.from_config(configuration)
            
        elif weight == 'pythia-410m':
            modelname = 'EleutherAI/pythia-410m'
            configuration = AutoConfig.from_pretrained(modelname)
            if drop_out is not None:
                configuration.hidden_dropout_prob = drop_out
                configuration.attention_probs_dropout_prob = drop_out
            self.model = AutoModel.from_pretrained(modelname, config = configuration) if not from_scratch else AutoModel.from_config(configuration)
        
        elif weight == 'pythia-1b':
            modelname = 'EleutherAI/pythia-1b'
            configuration = AutoConfig.from_pretrained(modelname)
            if drop_out is not None:
                configuration.hidden_dropout_prob = drop_out
                configuration.attention_probs_dropout_prob = drop_out
            self.model = AutoModel.from_pretrained(modelname, config = configuration) if not from_scratch else AutoModel.from_config(configuration)
        
        elif weight == 'pythia-1,4b':
            modelname = 'EleutherAI/pythia-1.4b'
            configuration = AutoConfig.from_pretrained(modelname)
            if drop_out is not None:
                configuration.hidden_dropout_prob = drop_out
                configuration.attention_probs_dropout_prob = drop_out
            self.model = AutoModel.from_pretrained(modelname, config = configuration) if not from_scratch else AutoModel.from_config(configuration)
        
        else:
            modelname = 'roberta-base' if weight[:7] == 'roberta' else 'bert-base-uncased'
            configuration = AutoConfig.from_pretrained(modelname)
            if drop_out is not None:
                configuration.hidden_dropout_prob = drop_out
                configuration.attention_probs_dropout_prob = drop_out
            self.model = AutoModel.from_pretrained(modelname, config = configuration) if not from_scratch else AutoModel.from_config(configuration)

        if random: 
            for module_ in self.model.named_modules():
                if isinstance(module_[1],(torch.nn.Linear, torch.nn.Embedding)):
                    module_[1].weight.data.normal_(mean=0.0, std=self.model.config.init_std)
                elif isinstance(module_[1], torch.nn.LayerNorm):
                    module_[1].bias.data.zero_()
                    module_[1].weight.data.fill_(1.0)
                if isinstance(module_[1], torch.nn.Linear) and module_[1].bias is not None:
                    module_[1].bias.data.zero_()
        if use_embedder:
            if weight == 'gpt2-medium' or weight == 'gpt2-medium' or weight == 'pythia-410m':
                self.embedder = Embeddings1D(input_shape, embed_dim=1024, target_seq_len=target_seq_len, config=self.model.config, dense=self.dense)
                embedder_init(self.model, self.embedder, train_embedder=train_epoch > 0)
            elif weight == 'gpt2-large':
                self.embedder = Embeddings1D(input_shape, embed_dim=1280, target_seq_len=target_seq_len, config=self.model.config, dense=self.dense)
                embedder_init(self.model, self.embedder, train_embedder=train_epoch > 0)
            elif weight == 'gpt2-xl':
                self.embedder = Embeddings1D(input_shape, embed_dim=1600, target_seq_len=target_seq_len, config=self.model.config, dense=self.dense)
                embedder_init(self.model, self.embedder, train_embedder=train_epoch > 0)
            elif weight == 'pythia-14m':
                self.embedder = Embeddings1D(input_shape, embed_dim=128, target_seq_len=target_seq_len, config=self.model.config, dense=self.dense)
                embedder_init(self.model, self.embedder, train_embedder=train_epoch > 0)
            elif weight == 'pythia-70m':
                self.embedder = Embeddings1D(input_shape, embed_dim=512, target_seq_len=target_seq_len, config=self.model.config, dense=self.dense)
                embedder_init(self.model, self.embedder, train_embedder=train_epoch > 0)
            elif weight == 'pythia-1b' or weight == 'pythia-1,4b':
                self.embedder = Embeddings1D(input_shape, embed_dim=2048, target_seq_len=target_seq_len, config=self.model.config, dense=self.dense)
                embedder_init(self.model, self.embedder, train_embedder=train_epoch > 0)
            elif weight == 'gpt2' or weight == 'pythia-160m':
                self.embedder = Embeddings1D(input_shape, embed_dim=768, target_seq_len=target_seq_len, config=self.model.config, dense=self.dense)
                embedder_init(self.model, self.embedder, train_embedder=train_epoch > 0)
            else:
                self.embedder = Embeddings1D(input_shape, config=self.model.config, embed_dim= 768, target_seq_len=target_seq_len, dense=self.dense)
                embedder_init(self.model.embeddings, self.embedder, train_embedder=train_epoch > 0)
                set_grad_state(self.embedder, True)    
        else:
            self.embedder = nn.Identity()

        self.model.embeddings = embedder_placeholder()

        self.model.pooler = nn.Identity()
        self.predictor = adaptive_pooler(out_channel = output_shape[-2] * self.embedder.stack_num, output_shape=output_shape, dense=True)


        if activation == 'sigmoid':
            self.predictor = nn.Sequential(self.predictor, nn.Sigmoid())  
            
        set_grad_state(self.model, False)
        set_grad_state(self.predictor, False)


    def forward(self, x):

        if self.output_raw:
            return self.embedder(x) 

        x = self.embedder(x)

        x = self.model(inputs_embeds=x)['last_hidden_state']
        x = self.predictor(x)


        if x.shape[1] == 1 and len(x.shape) == 2:
            x = x.squeeze(1)

        if self.double:
            x = x[:,:,x.shape[-1]//2:]
        
        return x


class Embeddings1D(nn.Module):
    def __init__(self, input_shape, embed_dim=768, target_seq_len=64, config=None, dense=False, double=False):
        super().__init__()
        self.dense = dense
        self.double = double
        self.embed_dim = embed_dim
        self.stack_num = self.get_stack_num(input_shape[-1], target_seq_len)
        self.patched_dimensions = (int(np.sqrt(input_shape[-1] // self.stack_num)), int(np.sqrt(input_shape[-1] // self.stack_num)))
        self.norm = nn.LayerNorm(embed_dim)
        self.padding_idx = 1
        
        if self.double:
            self.position_embeddings = nn.Embedding(target_seq_len * 2, embed_dim, padding_idx=self.padding_idx)
        else: 
            self.position_embeddings = nn.Embedding(target_seq_len, embed_dim, padding_idx=self.padding_idx)

        self.projection = nn.Conv1d(input_shape[1], embed_dim, kernel_size=self.stack_num, stride=self.stack_num)
        conv_init(self.projection)


    def get_stack_num(self, input_len, target_seq_len):
        if self.embed_dim in [128, 512, 768, 1024, 1280, 1600, 2048]:
            for i in range(1, input_len + 1):
                if input_len % i == 0 and input_len // i <= target_seq_len:
                    break
            return i
        else:
            for i in range(1, input_len + 1):
                root = np.sqrt(input_len // i)
                if input_len % i == 0 and input_len // i <= target_seq_len and int(root + 0.5) ** 2 == (input_len // i):
                    break
            return i


    def forward(self, x=None, inputs_embeds=None, *args, **kwargs):
        if x is None:
            x = inputs_embeds
        b, c, l = x.shape

        x = self.projection(x).transpose(1, 2)
        x = self.norm(x)
            
        position_ids = create_position_ids_from_inputs_embeds(x, self.padding_idx)
        self.ps = self.position_embeddings(position_ids)
        x = x + self.ps

        if self.embed_dim in [128, 512, 768, 1024, 1280, 1600, 2048]:
            return x
        else:
            return x, self.patched_dimensions



####################################################

def get_tgt_model(args, root, sample_shape, num_classes, loss, add_loss=False, use_determined=False, context=None, opid=0):
    
    src_train_loader, _, _, _, _, _, _ = get_data(root, args.embedder_dataset, args.batch_size, False, maxsize=5000,flip=args.flip, dobule=args.double)

    src_feats, src_ys = src_train_loader.dataset.tensors[0].mean(1), src_train_loader.dataset.tensors[1]
    src_train_dataset = torch.utils.data.TensorDataset(src_feats, src_ys)
        
    tgt_train_loader, _, _, n_train, _, _, data_kwargs = get_data(root, args.dataset, args.batch_size, False, get_shape=True, flip=args.flip, double=args.double)
    transform = data_kwargs['transform'] if data_kwargs is not None and 'transform' in data_kwargs else None
        
    if args.infer_label:
        tgt_train_loader, num_classes_new = infer_labels(tgt_train_loader)
    else:
        num_classes_new = num_classes

    print("src feat shape", src_feats.shape, src_ys.shape, "num classes", num_classes_new) 

    tgt_train_loaders, tgt_class_weights = load_by_class(tgt_train_loader, num_classes_new)

    wrapper_func = wrapper1D
    tgt_model = wrapper_func(sample_shape, num_classes, weight=args.weight, train_epoch=args.embedder_epochs, activation=args.activation, target_seq_len=args.target_seq_len, drop_out=args.drop_out)
    tgt_model = tgt_model.to(args.device).train()

    args, tgt_model, tgt_model_optimizer, tgt_model_scheduler = get_optimizer_scheduler(args, tgt_model, module='embedder')
    tgt_model_optimizer.zero_grad()

    if args.objective == 'otdd-exact':
        score_func = partial(otdd, src_train_dataset=src_train_dataset, exact=True)
    elif args.objective == 'otdd-gaussian':
        score_func = partial(otdd, src_train_dataset=src_train_dataset, exact=False)
    elif args.objective == 'l2':
        score_func = partial(l2, src_train_dataset=src_train_dataset)
    else:
        score_func = MMD_loss(src_data=src_feats, maxsamples=args.maxsamples)
    
    score = 0
    total_losses, times, embedder_stats = [], [], []
    
    for ep in range(args.embedder_epochs):   

        total_loss = 0    
        time_start = default_timer()

        for i in np.random.permutation(num_classes_new):
            feats = []
            datanum = 0

            for j, data in enumerate(tgt_train_loaders[i]):
                
                if transform is not None:
                    x, y, z = data
                else:
                    x, y = data 
                
                x = x.to(args.device)
                out = tgt_model(x)
                feats.append(out)
                datanum += x.shape[0]
                
                if datanum > args.maxsamples: break

            feats = torch.cat(feats, 0).mean(1)
            if feats.shape[0] > 1:
                loss = tgt_class_weights[i] * score_func(feats)
                loss.backward()
                total_loss += loss.item()

        time_end = default_timer()  
        times.append(time_end - time_start) 

        total_losses.append(total_loss)
        embedder_stats.append([total_losses[-1], times[-1]])
        print("[train embedder", ep, "%.6f" % tgt_model_optimizer.param_groups[0]['lr'], "] time elapsed:", "%.4f" % (times[-1]), "\totdd loss:", "%.4f" % total_losses[-1])

        tgt_model_optimizer.step()
        tgt_model_scheduler.step()
        tgt_model_optimizer.zero_grad()

    del tgt_train_loader, tgt_train_loaders
    torch.cuda.empty_cache()

    tgt_model.output_raw = False

    return tgt_model, embedder_stats


def infer_labels(loader, k = 10):
    from sklearn.cluster import k_means, MiniBatchKMeans
    
    if hasattr(loader.dataset, 'tensors'):
        X, Y = loader.dataset.tensors[0].cpu(), loader.dataset.tensors[1].cpu().numpy()
        try:
            Z = loader.dataset.tensors[2].cpu()
        except:
            Z = None
    else:
        X, Y, Z = get_tensors(loader.dataset)

    Y = Y.reshape(len(Y), -1)

    if len(Y) <= 10000:
        labeling_fun = lambda Y: torch.LongTensor(k_means(Y, k)[1])
        Y = labeling_fun(Y).unsqueeze(1)
    else:
        kmeans = MiniBatchKMeans(n_clusters=k, batch_size=10000).fit(Y)
        Y = torch.LongTensor(kmeans.predict(Y)).unsqueeze(1)

    if Z is None:
        return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, Y), batch_size=loader.batch_size, shuffle=True, num_workers=4, pin_memory=True), k
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, Y, Z), batch_size=loader.batch_size, shuffle=True, num_workers=4, pin_memory=True), k


def load_by_class(loader, num_classes):
    train_set = loader.dataset
    subsets = {}

    if len(train_set.__getitem__(0)) == 3:
        try:
            subsets = {target: torch.utils.data.Subset(train_set, [i for i, (x, y, _) in enumerate(train_set) if y == target]) for target in range(num_classes)}
        except:
            subsets = {target: torch.utils.data.Subset(train_set, [i for i, (x, y, _) in enumerate(train_set) if y.item() == target]) for target in range(num_classes)}
    else:
        try:
            subsets = {target: torch.utils.data.Subset(train_set, [i for i, (x, y) in enumerate(train_set) if y == target]) for target in range(num_classes)}
        except:
            subsets = {target: torch.utils.data.Subset(train_set, [i for i, (x, y) in enumerate(train_set) if y.item() == target]) for target in range(num_classes)}
    loaders = {target: torch.utils.data.DataLoader(subset, batch_size=loader.batch_size, shuffle=True, num_workers=4, pin_memory=True) for target, subset in subsets.items()}
    class_weights = {target: len(subset)/len(train_set) for target, subset in subsets.items()}
    
    print("class weights")
    for target, subset in subsets.items():
        print(target, len(subset), len(train_set), len(subset)/len(train_set))

    return loaders, class_weights



def get_tensors(dataset):
    xs, ys, zs = [], [], []
    for i in range(dataset.__len__()):
        data = dataset.__getitem__(i)
        xs.append(np.expand_dims(data[0], 0))
        ys.append(np.expand_dims(data[1], 0))
        if len(data) == 3:
            zs.append(np.expand_dims(data[2], 0))

    xs = torch.from_numpy(np.array(xs)).squeeze(1)
    ys = torch.from_numpy(np.array(ys)).squeeze(1)

    if len(zs) > 0:
        zs = torch.from_numpy(np.array(zs)).squeeze(1)
    else:
        zs = None

    return xs, ys, zs