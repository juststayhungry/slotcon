import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torchvision

class DINOHead(nn.Module):
    ''' b  k*d  256--4096---256
    （linear+bn+gelu）*3 +Linear(bottleneck)  non-linear predictor。将stu的输出进一步变换。predictor 避免训练崩塌的重要元素之一
    '''
    def __init__(self, in_dim, use_bn=True, nlayers=3, hidden_dim=4096, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)#根据输入参数来判断此时是实例化还是调用该函数的call调用的forward
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        return x

class DINOHead2d(nn.Module):
    '''
    输入：encoder输出的feature resnet: bdhw(d = 512/2048)  ViT
    投射projector
    输出：feature map，输出通道数c：256。b c h w
    conv2d+bn+2*(conv+gelu)+conv2d(bottleneck)
    卷积核的大小均为1*1projector降维
    '''
    def __init__(self, in_dim, use_bn=True, nlayers=3, hidden_dim=4096, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Conv2d(in_dim, bottleneck_dim, 1)
        else:
            layers = [nn.Conv2d(in_dim, hidden_dim, 1)]
            if use_bn:
                layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Conv2d(hidden_dim, hidden_dim, 1))
                if use_bn:
                    layers.append(nn.BatchNorm2d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Conv2d(hidden_dim, bottleneck_dim, 1))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        return x
    
class SemanticGrouping(nn.Module):
    '''
    实例化的输入：聚类数量，维度
    实例对象的输入：projector的输出feature
    输出：dots  slots
    功能：语义聚类
    唯一可学习参数：聚类中心slot，nn.embedding
    '''
    def __init__(self, num_slots, dim_slot, temp=0.07, eps=1e-6):#num_prototypes, self.dim_out, self.teacher_temp
        '''
        prototypes即solts
        '''
        super().__init__()
        self.num_slots = num_slots
        self.dim_slot = dim_slot
        self.temp = temp
        self.eps = eps

        self.slot_embed = nn.Embedding(num_slots, dim_slot)#可学习的slot即一种slot_embedding，降维，将feature---更低维度的slot_vecter
        '''
        num_slots词典大小尺寸
        embedding_dim表示嵌入向量的维度，即slots的维度
        '''
    def forward(self, x):
        x_prev = x
        torch.matmul()
        slots = self.slot_embed(torch.arange(0, self.num_slots, device=x.device)).unsqueeze(0).repeat(x.size(0), 1, 1)#b个slots，每个slots是一个词汇表(num*dim) b k dim
        dots = torch.einsum('bkd,bdhw->bkhw', F.normalize(slots, dim=2), F.normalize(x, dim=1))#normalize归一化后的zθl 输入的feature map:x与聚类中心slot做点乘操作
        '''将输入的feature与slot做注意力操作  kd@dhw---khw
        dots是k:input，q:slots做注意力得到的初始化未经过归一化前的权重系数矩阵，做softmax后才是attn。将attn@input后得到的就是更新后的slot
        矩阵运算是核心
         PyTorch 的函数 einsum，它实现了张量的乘法和求和操作。
        具体来说，该函数的第一个参数是一个字符串，指定了输入张量之间的运算规则。其中，'bkd' 表示第一个输入张量slots的形状为 (b,k,d)batch/k聚类中心数/d特征维度，
        这意味着函数将两个张量进行了乘法运算，并沿着 d维度进行求和，得到了一个形状为 (b, k, h, w) 的输出张量：聚类中心响应dots:map-----dots bkhw。
        '''
        attn = (dots / self.temp).softmax(dim=1) + self.eps#Aθl(attn)=softmax(  dots / self.temp )/温度系数并在dim=1下归一化sum=1#softmax分布结果(dino的部分)
        slots = torch.einsum('bdhw,bkhw->bkd', x_prev, attn / attn.sum(dim=(2, 3), keepdim=True))#Sθ(slot)=平均化(x点乘attn)
        '''
        通过slot attention得到slots  bkd
        '''
        return slots, dots

class SlotCon(nn.Module):
    def __init__(self, encoder, args):
        super().__init__()

        self.dim_hidden = args.dim_hidden
        self.dim_out = args.dim_out
        self.teacher_momentum = args.teacher_momentum

        if args.arch in ('resnet18', 'resnet34'):
            self.num_channels = 512
        elif args.arch in ('vit_tiny'):
            self.num_channels = 192
        elif args.arch in ('vit_small'):
            self.num_channels = 384
        elif args.arch in ('vit_base'):
            self.num_channels = 768
        else:
            self.num_channels = 2048

        # self.num_channels = 512 if args.arch in ('resnet18', 'resnet34') else 2048#resnet50输出通道数是2048
        
        if encoder is None:
            self.encoder_q = encoder(head_type='early_return')
            self.encoder_k = encoder(head_type='early_return')

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            '''
            k的encoder模型参数初始化为q
            '''
            param_k.requires_grad = False  # not update by gradient

        nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder_q)
        '''
        convert all :attr:`BatchNorm*D` layers in the model to class:`torch.nn.SyncBatchNorm` layers.
        用于将普通的 Batch Normalization 层（nn.BatchNorm2d 或 nn.BatchNorm1d）转换为支持分布式训练的 Sync Batch Normalization 层。
        该函数通过复制每个 BatchNorm 层的均值和标准差，并在多个进程间同步统计信息来实现跨进程通信，可以提高在分布式环境下的模型训练效率。
        具体来说，在使用 PyTorch 进行分布式训练的时候，由于数据被划分为多个 batch 并分配给多个 GPU，每个 GPU 只能看到本地 GlobalBatchSize / NumGpu 大小的数据。
        这会导致 Batch Normalization 在不同进程之间需要同步均值和方差信息。而 Sync Batch Normalization 通过将每个进程上的均值和方差同步，从而避免了精度下降的问题。
        因此，nn.SyncBatchNorm.convert_sync_batchnorm 的主要作用是将普通的 Batch Normalization 层转换为支持分布式训练的 Sync Batch Normalization 层，
        以防止分布式训练中出现 batch 统计信息不准确的问题，从而提高训练效率。
        '''
        nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder_k)

        self.group_loss_weight = args.group_loss_weight
        self.student_temp = args.student_temp
        self.teacher_temp = args.teacher_temp
            
        self.projector_q = DINOHead2d(self.num_channels, hidden_dim=self.dim_hidden, bottleneck_dim=self.dim_out)
        self.projector_k = DINOHead2d(self.num_channels, hidden_dim=self.dim_hidden, bottleneck_dim=self.dim_out)

        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            '''
            k的projector模型参数初始化为q
            '''
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
            
        nn.SyncBatchNorm.convert_sync_batchnorm(self.projector_q)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.projector_k)

        self.num_prototypes = args.num_prototypes#聚类中心数量的初始化
        self.center_momentum = args.center_momentum#中心化的动量参数的初始化
        self.register_buffer("center", torch.zeros(1, self.num_prototypes))
        '''
        在模型中register_buffer注册一个缓存(buffer)变量，并将其命名为 "center"。
        该缓存变量是一个形状为 (1, self.num_prototypes) 的张量(tensor)，元素全部初始化为 0。
        其中，self 是指该代码所在的类的实例对象。
        '''
        self.grouping_q = SemanticGrouping(self.num_prototypes, self.dim_out, self.teacher_temp)#实例化
        self.grouping_k = SemanticGrouping(self.num_prototypes, self.dim_out, self.teacher_temp)
        self.predictor_slot = DINOHead(self.dim_out, hidden_dim=self.dim_hidden, bottleneck_dim=self.dim_out)
        '''
        slots>----DINOHead--->slots_for_predict
        '''
        nn.SyncBatchNorm.convert_sync_batchnorm(self.predictor_slot)
            
        for param_q, param_k in zip(self.grouping_q.parameters(), self.grouping_k.parameters()):
            '''
            k的grouping模块的参数初始化为q
            '''
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.K = int(args.num_instances * 1. / args.world_size / args.batch_size * args.epochs)
        self.k = int(args.num_instances * 1. / args.world_size / args.batch_size * (args.start_epoch - 1))

    def re_init(self, args):
        self.k = int(args.num_instances * 1. / args.world_size / args.batch_size * (args.start_epoch - 1))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder(教师模型的encoder的更新是动量形式更新的——即MoCo那样的动量编码器)
        动量编码器是自监督模型MoCo中使用的一种编码器，在训练过程中引入了动量(momentum)的概念。
        在MoCo模型中，动量编码器的作用是通过更新Key编码器的参数，以一种合适的方式引入Query编码器的状态，从而使得Query和Key之间的相似性得到保持。
        具体来说，通过使用动量的方式更新Key编码器的参数，可以获得一个更加稳定的网络状态，同时也可以避免模型在训练过程中陷入局部最优点。
        此外，研究表明，使用合适大小的动量可以保持Query和Key之间的一致性，并有助于Key编码器的训练。
        """
        momentum = 1. - (1. - self.teacher_momentum) * (math.cos(math.pi * self.k / self.K) + 1.) * 0.5
        self.k += 1
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)
                    # ema update   原来的数值*m+更新(shudent的数值)*(1-m)
        # self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)#将teacher的输出求和，求平均，ema形式的更新center
        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)
        for param_q, param_k in zip(self.grouping_q.parameters(), self.grouping_k.parameters()):
            param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)  

    def invaug(self, x, coords, flags):
        '''
        逆增强
        '''
        N, C, H, W = x.shape
        batch_idxs = torch.arange(N, device=coords.device).view(N, 1)
        coords_rescaled = coords.clone()
        coords_rescaled[:, 0] = coords_rescaled[:, 0] * W  # x1
        coords_rescaled[:, 2] = coords_rescaled[:, 2] * W  # x2
        coords_rescaled[:, 1] = coords_rescaled[:, 1] * H  # y1
        coords_rescaled[:, 3] = coords_rescaled[:, 3] * H  # y2
        coords_with_idxs = torch.cat([batch_idxs, coords_rescaled], dim=1)
        
        x_aligned = torchvision.ops.roi_align(x, coords_with_idxs, (H, W), aligned=True)
        x_flipped = torch.stack([feat.flip(-1) if flag else feat for feat, flag in zip(x_aligned, flags)])
        return x_flipped

    def self_distill(self, q, k): #DINO伪代码里面的def H(t,s)，输入两个view的分布，输出对应的自蒸馏损失函数(即两view量化的差异)
        '''
        输入对齐后的dots以后再做softmax归一化，计算交叉熵损失CE_loss=-q*logp
        自蒸馏q1_aligned.permute(0, 2, 3, 1).flatten(0, 2)  ------  k2_aligned.permute(0, 2, 3, 1).flatten(0, 2)
        q1_aligned, q2_aligned = self.invaug(score_q1, coords[0], flags[0]), self.invaug(score_q2, coords[1], flags[1])
        #对聚类中心响应后的map做inverse augument对齐align
        '''
        q = F.log_softmax(q / self.student_temp, dim=-1)

        k = F.softmax((k - self.center) / self.teacher_temp, dim=-1)
        '''
        center平滑
        temp_t锐化
        '''
        return torch.sum(-k * q, dim=-1).mean()

    def ctr_loss_filtered(self, q, k, score_q, score_k, tau=0.2):
        '''
        q2, k1, score_q2, score_k1
        q,k是slots，score是dots
        对比学习的损失函数:InfoNCE
        '''
        q = q.flatten(0, 1)  #q:bkd---b*k  d   q待会得输入predictor，对其输出做归一化
        k = F.normalize(k.flatten(0, 1), dim=1)#k先展平至b*k  d后归一化
        '''
        由于初始的slot由整个数据集共享，因此在一个特定的视图view中可能缺少相应的语义，从而产生冗余的slot。
        因此计算以下二进制指示器1l(sumsum>0)来掩盖mask无法占据主导像素的插槽
        '''
        '''#scatter_(dim,index,src)，在维度dim上，将索引号为index最大分数的位置，替换为src:1    
        dots:bkhw
        b个khw，scatter_(1, score_q.argmax(1, keepdim=True), 1)n个样本的k张h*w大小的slot_i（语义类别i）(i=0~k-1)的响应图
        sum sum>0   mask无法占据主导像素的插槽(k类的响应小于0的mask掉)  ---b k，且全部都>0/=0(mask)   ---mask_q:b k    a1...ak  b1...bk
        '''
        mask_q = (torch.zeros_like(score_q).scatter_(1, score_q.argmax(1, keepdim=True), 1).sum(-1).sum(-1) > 0).long().detach()
        mask_k = (torch.zeros_like(score_k).scatter_(1, score_k.argmax(1, keepdim=True), 1).sum(-1).sum(-1) > 0).long().detach()#b k    a1...ak  b1...bk
        mask_intersection = (mask_q * mask_k).view(-1)#展平，b k---b*k               a1...akb1...bk
        
        idxs_q = mask_intersection.nonzero().squeeze(-1)#含语义的slot索引---idxs_q

        mask_k = concat_all_gather(mask_k.view(-1))#展平，b k---b*k   b*k--gather---n b*k      n  b*k  mask_k:n条  a1...akb1...bk
        idxs_k = mask_k.nonzero().squeeze(-1)#除去没有物体的，得到含有物体的idxs
        '''qk相似度矩阵，q是正样本，k是包含正样本与负样本，label：'''
        N = k.shape[0] #k: b*k  d    N=b*k       
        #q：b*k  d_in ---经过predictor_slot---b*k  d_out   (d_in=d_out=agr.dim-out = 256)
        # b*k  d_out,b*k_gather  d_out-> b*k  b*k_gather(两view的交互响应矩阵)
        logits = torch.einsum('nc,mc->nm', [F.normalize(self.predictor_slot(q[idxs_q]), dim=1), concat_all_gather(k)[idxs_k]]) / tau
        labels = mask_k.cumsum(0)[idxs_q + N * torch.distributed.get_rank()] - 1#标签的取值范围应该在[0，类别数-1]之间
        '''
        nn.CrossEntropyLoss()函数不需要对logits进行softmax操作，因为它包含了softmax计算的过程。
        同时，labels的形状应该与logits的形状相同，即每个样本对应一个标签，标签的取值范围应该在[0，类别数-1]之间。
        使用交叉熵损失函数计算InfoNCE损失。其中，labels表示Key的分类标签，也就是按照相似度排序后每个Key所在的类别编号。
        具体来说，通过mask_k.cumsum(0)计算得到Key的类别累计和，再根据idxs_q和进程编号获得对应的类别编号，从而得到labels
        '''
        return F.cross_entropy(logits, labels) * (2 * tau)#labels怎么得到的，因为是自监督，将k的输出作为label
#单步调试 GDB FPGD
    def forward(self, input):
        crops, coords, flags = input
        x1, x2 = self.projector_q(self.encoder_q(crops[0])), self.projector_q(self.encoder_q(crops[1]))#v->f(v)->p(v)
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            y1, y2 = self.projector_k(self.encoder_k(crops[0])), self.projector_k(self.encoder_k(crops[1]))
            
        (q1, score_q1), (q2, score_q2) = self.grouping_q(x1), self.grouping_q(x2)#对投影输出的query做语义分组得到return slots, dots
        q1_aligned, q2_aligned = self.invaug(score_q1, coords[0], flags[0]), self.invaug(score_q2, coords[1], flags[1])#对聚类中心响应后的map做inverse augument对齐align
        with torch.no_grad():
            (k1, score_k1), (k2, score_k2) = self.grouping_k(y1), self.grouping_k(y2)#对投影输出的key做语义分组得到slots-----k or q 与dots------score(响应后的map)
            k1_aligned, k2_aligned = self.invaug(score_k1, coords[0], flags[0]), self.invaug(score_k2, coords[1], flags[1])#对聚类中心响应后的map做inverse augument对齐align

        loss = self.group_loss_weight * self.self_distill(q1_aligned.permute(0, 2, 3, 1).flatten(0, 2), k2_aligned.permute(0, 2, 3, 1).flatten(0, 2)) \
             + self.group_loss_weight * self.self_distill(q2_aligned.permute(0, 2, 3, 1).flatten(0, 2), k1_aligned.permute(0, 2, 3, 1).flatten(0, 2))#\换行

        self.update_center(torch.cat([score_k1, score_k2]).permute(0, 2, 3, 1).flatten(0, 2))#bkhw  bhwk---flatten(s=0,end=2)  b*h*w  k
        '''
        loss=group_loss_weight*自蒸馏的损失(两个dots对齐后的损失)+(1-group_loss_weight)*slots：q1,q2对比学习的infoNCE损失
        '''
        loss += (1. - self.group_loss_weight) * self.ctr_loss_filtered(q1, k2, score_q1, score_k2) \
              + (1. - self.group_loss_weight) * self.ctr_loss_filtered(q2, k1, score_q2, score_k1)
        '''model(input data)返回的就是forward的输出，即loss或者y'''
        return loss
    
    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        对teacher输出b*h*w  k做center操作，中心化操作，需要将
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)#将teacher的输出求和，求平均，ema形式的更新center

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

class SlotConEval(nn.Module):
    def __init__(self, encoder, args):
        super().__init__()

        self.dim_hidden = args.dim_hidden
        self.dim_out = args.dim_out

        # self.num_channels = 512 if args.arch in ('resnet18', 'resnet34') else 2048
        if args.arch in ('resnet18', 'resnet34'):
            self.num_channels = 512
        elif args.arch in ('vit_tiny'):
            self.num_channels = 192
        elif args.arch in ('vit_small'):
            self.num_channels = 384
        elif args.arch in ('vit_base'):
            self.num_channels = 768
        else:
            self.num_channels = 2048
            
        self.encoder_k = encoder(head_type='early_return')
        for param_k in self.encoder_k.parameters():
            param_k.requires_grad = False  # not update by gradient

        self.projector_k = DINOHead2d(self.num_channels, hidden_dim=self.dim_hidden, bottleneck_dim=self.dim_out)
        for param_k in self.projector_k.parameters():
            param_k.requires_grad = False  # not update by gradient

        self.num_prototypes = args.num_prototypes
        self.grouping_k = SemanticGrouping(self.num_prototypes, self.dim_out)
        for param_k in self.grouping_k.parameters():
            param_k.requires_grad = False  # not update by gradient

    def forward(self, x):
        with torch.no_grad():
            slots, probs = self.grouping_k(self.projector_k(self.encoder_k(x)))#整体流程
            return probs #dots
        
#SwAV的训练
'''
def train(train_loader, model, optimizer, epoch, lr_schedule, queue):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    use_the_queue = False

    end = time.time()
    for it, inputs in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # update learning rate
        iteration = epoch * len(train_loader) + it
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_schedule[iteration]

        # normalize the prototypes
        with torch.no_grad():
            w = model.module.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            model.module.prototypes.weight.copy_(w)

        # ============ multi-res forward passes ... ============
        embedding, output = model(inputs)
        embedding = embedding.detach()
        bs = inputs[0].size(0)

        # ============ swav loss ... ============
        loss = 0
        for i, crop_id in enumerate(args.crops_for_assign):
            with torch.no_grad():
                out = output[bs * crop_id: bs * (crop_id + 1)].detach()

                # time to use the queue
                if queue is not None:
                    if use_the_queue or not torch.all(queue[i, -1, :] == 0):
                        use_the_queue = True
                        out = torch.cat((torch.mm(
                            queue[i],
                            model.module.prototypes.weight.t()
                        ), out))
                    # fill the queue
                    queue[i, bs:] = queue[i, :-bs].clone()
                    queue[i, :bs] = embedding[crop_id * bs: (crop_id + 1) * bs]

                # get assignments
                q = distributed_sinkhorn(out)[-bs:]#最优运输/分配问题？   将样本点与聚类中心相匹配

            # cluster assignment prediction
            subloss = 0
            for v in np.delete(np.arange(np.sum(args.nmb_crops)), crop_id):
                x = output[bs * v: bs * (v + 1)] / args.temperature
                subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))
            loss += subloss / (np.sum(args.nmb_crops) - 1)
        loss /= len(args.crops_for_assign)

        # ============ backward and optim step ... ============
        optimizer.zero_grad()
        if args.use_fp16:
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        # cancel gradients for the prototypes
        if iteration < args.freeze_prototypes_niters:
            for name, p in model.named_parameters():
                if "prototypes" in name:
                    p.grad = None
        optimizer.step()

        # ============ misc ... ============
        losses.update(loss.item(), inputs[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if args.ran ==k0 and it % 50 == 0:
            logger.info(
                "Epoch: [{0}][{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Lr: {lr:.4f}".format(
                    epoch,
                    it,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    lr=optimizer.optim.param_groups[0]["lr"],
                )
            )
    return (epoch, losses.avg), queue
'''
'''DINO的loss'''
'''
class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss
'''


'''SlotAttention'''
''' 通过将input作为k-v对，slots作为q，将slots参数初始化为高斯分布，随后对kqv分别做projection，
将kq做dots再softamx得到attn（注意力分数），将attn与v做dots得到注意力机制的最终输出update，将该输出与slot_pre输入GRU，进行GRU的更新
gru(updates, [slots_prev])，将GRU输出的slots做LN归一化后得到slot attention一次迭代更新得到的slots，循环n次(相对于n层transformer)'''
'''
# class SlotAttention(layers.Layer):
#   """Slot Attention module.带GRU的transformer
#   用一个slot去找到一个对象，object-level"""

#   def __init__(self, num_iterations, num_slots, slot_size, mlp_hidden_size,
#                epsilon=1e-8):
#     """Builds the Slot Attention module.
#     Args:
#       num_iterations: Number of iterations.
#       num_slots: Number of slots.
#       slot_size: Dimensionality of slot feature vectors.
#       mlp_hidden_size: Hidden layer size of MLP.
#       epsilon: Offset for attention coefficients before normalization.
#     """
#     super().__init__()
#     self.num_iterations = num_iterations#迭代次数
#     self.num_slots = num_slots
#     self.slot_size = slot_size
#     self.mlp_hidden_size = mlp_hidden_size
#     self.epsilon = epsilon

#     self.norm_inputs = layers.LayerNormalization()
#     self.norm_slots = layers.LayerNormalization()
#     self.norm_mlp = layers.LayerNormalization()

#     # Parameters for Gaussian init (shared by all slots).
#     self.slots_mu = self.add_weight(
#         initializer="glorot_uniform",
#         shape=[1, 1, self.slot_size],
#         dtype=tf.float32,
#         name="slots_mu")
#     self.slots_log_sigma = self.add_weight(
#         initializer="glorot_uniform",
#         shape=[1, 1, self.slot_size],
#         dtype=tf.float32,
#         name="slots_log_sigma")

#     # Linear maps for the attention module.
#     self.project_q = layers.Dense(self.slot_size, use_bias=False, name="q")
#     self.project_k = layers.Dense(self.slot_size, use_bias=False, name="k")
#     self.project_v = layers.Dense(self.slot_size, use_bias=False, name="v")

#     # Slot update functions.
#     self.gru = layers.GRUCell(self.slot_size)#用于迭代更新slot
#     self.mlp = tf.keras.Sequential([
#         layers.Dense(self.mlp_hidden_size, activation="relu"),
#         layers.Dense(self.slot_size)
#     ], name="mlp")

#   def call(self, inputs):
#     # `inputs` has shape [batch_size, num_inputs, inputs_size].
#     inputs = self.norm_inputs(inputs)  # Apply layer norm to the input.
#     k = self.project_k(inputs)  # Shape: [batch_size, num_inputs, slot_size].
#     v = self.project_v(inputs)  # Shape: [batch_size, num_inputs, slot_size].

#     # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
#     slots = self.slots_mu + tf.exp(self.slots_log_sigma) * tf.random.normal(
#         [tf.shape(inputs)[0], self.num_slots, self.slot_size])

#     # Multiple rounds of attention.
#     for _ in range(self.num_iterations):
#       slots_prev = slots
#       slots = self.norm_slots(slots)

#       # Attention.
#       q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].
#       q *= self.slot_size ** -0.5  # Normalization.
#       attn_logits = tf.keras.backend.batch_dot(k, q, axes=-1)
#       attn = tf.nn.softmax(attn_logits, axis=-1)
#       # `attn` has shape: [batch_size, num_inputs, num_slots].

#       # Weigted mean.# aggregate
#       attn += self.epsilon
#       attn /= tf.reduce_sum(attn, axis=-2, keepdims=True)
#       updates = tf.keras.backend.batch_dot(attn, v, axes=-2)
#       # `updates` has shape: [batch_size, num_slots, slot_size].

#       # Slot update.
#       slots, _ = self.gru(updates, [slots_prev])#迭代更新slot
#       slots += self.mlp(self.norm_mlp(slots))
#     return slots
'''