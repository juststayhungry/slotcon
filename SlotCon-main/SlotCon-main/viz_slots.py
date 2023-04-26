import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from data.datasets import ImageFolder
from models import resnet
from models.slotcon import SlotConEval

def denorm(img):
    mean, val = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])
    img = (img * val[:, None, None] + mean[:, None, None]) * torch.tensor([255, 255, 255])[:, None, None]
    return img.permute(1, 2, 0).cpu().type(torch.uint8)

def get_model(args):
    encoder = resnet.__dict__[args.arch]
    model = SlotConEval(encoder, args)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    weights = {k.replace('module.', ''):v for k, v in checkpoint['model'].items()}
    model.load_state_dict(weights, strict=False)
    model = model.eval()
    return model

def get_features(model, dataset, bs):
    memory_loader = torch.utils.data.DataLoader(
        dataset, batch_size=bs, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    bank = []
    for data in tqdm(memory_loader, desc='Feature extracting', leave=False, disable=False):
        feature = model.projector_k(model.encoder_k(data))#.mean(dim=(-2, -1)) #data(bhw)-encoder-->(bchw)---projector(降维)--bdhw
        feature = F.normalize(feature, dim=1)#bdhw
        bank.append(feature)
    bank = torch.cat(bank, dim=0)# n(n=b*i) dhw
    return bank

def prepare_knn(model, dataset, args):
    prototypes = F.normalize(model.grouping_k.slot_embed.weight, dim=1) # kd
    memory_bank = get_features(model, dataset, args.batch_size) # ndhw
    dots = torch.einsum('kd,ndhw->nkhw', [prototypes, memory_bank]) # nkhw
    masks = torch.zeros_like(dots).scatter_(1, dots.argmax(1, keepdim=True), 1)#nkhw n个样本的k张h*w大小的slot_i(i=0~k-1)的响应图
    masks_adder = masks + 1.e-6
    scores = (dots * masks_adder).sum(-1).sum(-1) / masks_adder.sum(-1).sum(-1) # nk   n k 求出每个slot的响应数值
    _, idxs = scores.t().topk(dim=1, k=args.topk)#选择batch中每类slot中响应最强的K个样本的索引输出
    return dots, idxs

def viz_slots(dataset, dots, idxs, slot_idxs, args):   
    color = np.array([255, 0, 0]).reshape(1, 1, 3)
    fig, ax = plt.subplots(args.topk, len(slot_idxs), figsize=(len(slot_idxs)*2, args.topk*2), squeeze=False, dpi=args.dpi)
    '''
    dots, idxs = prepare_knn(model, dataset, args)
    slot_idxs----range(args.num_prototypes)#0-k-1
    '''
    for i, slot_idx in enumerate(tqdm(slot_idxs, desc='KNN retreiving', leave=False, disable=False)):
        # ax[0, i].set_title(i)
        for j in range(args.topk):
            idx = idxs[slot_idx, j]
            image = denorm(dataset[idx]).numpy()#去除归一化
            pred = transforms.functional.resize(dots[idx],image.shape[:2], TF.InterpolationMode.BILINEAR)#将dots还原回原图大小
            mask = torch.zeros_like(pred).scatter_(0, pred.argmax(0, keepdim=True), 1)
            mask = mask[slot_idx].unsqueeze(-1).cpu().numpy()
            image = np.int32((args.alpha * (image * mask) + (1 - args.alpha) * color * mask) + (image * (1 - mask)))
            '''
            mask掩膜对应的才是object
            '''
            ax[j, i].imshow(image)
            ax[j, i].axis('off')
    fig.tight_layout()
    fig.savefig(args.save_path, bbox_inches='tight')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # viz-related
    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--dpi', type=int, default=100)
    parser.add_argument('--sampling', type=int, default=1)
    parser.add_argument('--idxs', type=list, default=[])
    parser.add_argument('--save_path', type=str, default='viz_slots_4_21.jpg')
    # dataset
    parser.add_argument('--dataset', type=str, default='CLEVRtest', help='dataset type')
    parser.add_argument('--data_dir', type=str, default='D:/research/代码复现', help='dataset director')
    parser.add_argument('--batch_size', type=int, default=64)
    # Model.
    parser.add_argument('--model_path', type=str, default='SlotCon-main\output\slotcon_coco_r50_800ep.pth')
    parser.add_argument('--dim_hidden', type=int, default=4096)
    parser.add_argument('--dim_out', type=int, default=256)
    parser.add_argument('--arch', type=str, default='resnet50')
    parser.add_argument('--num_prototypes', type=int, default=26)#聚类中心数量
    args = parser.parse_args()

    mean_vals, std_vals = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    # transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=mean_vals, std=std_vals)])
    transform = transforms.Compose([transforms.Resize(256), transforms.ToTensor(), transforms.Normalize(mean=mean_vals, std=std_vals)])

    dataset = ImageFolder(args.dataset, args.data_dir, transform)
    print(dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args).to(device)

    dots, idxs = prepare_knn(model, dataset, args)
    if args.sampling > 0:
        slot_idxs = np.random.randint(0, args.num_prototypes, args.sampling)#输出0-K之间的一个随机整数，随机选一个聚类中心，共选sampling个，即向量长度为samping，元素数值介于0-k-1
    elif len(args.idxs) > 0:
        slot_idxs = args.idxs
    else:
        slot_idxs = range(args.num_prototypes)#0-k-1
    viz_slots(dataset, dots, idxs, slot_idxs, args)
