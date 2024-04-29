from __future__ import print_function
import argparse
import time
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, LLCMData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb, eval_llcm
from model_sle import embed_net as basline_pcb
from utils import *
from loss import OriTripletLoss, KLDivLoss
from tensorboardX import SummaryWriter
from ChannelAug import ChannelAdapGray, ChannelRandomErasing

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline:resnet18 or resnet50')
parser.add_argument('--resume', '-r', default=1, type=int,
                    help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='save_model/', type=str,
                    help='model save path')
parser.add_argument('--save_epoch', default=20, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str,
                    help='log save path')
parser.add_argument('--vis_log_path', default='log/vis_log/', type=str,
                    help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=8, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--method', default='adp', type=str,
                    metavar='m', help='method type: base or agw, adp')
parser.add_argument('--margin', default=0.3, type=float,
                    metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=4, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--gpu', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')
parser.add_argument('--augc', default=1, type=int,
                    metavar='aug', help='use channel aug or not')
parser.add_argument('--rande', default=0.5, type=float,
                    metavar='ra', help='use random erasing or not and the probability')
parser.add_argument('--kl', default=0, type=float,
                    metavar='kl', help='use kl loss and the weight')
parser.add_argument('--alpha', default=1, type=int,
                    metavar='alpha', help='magnification for the hard mining')
parser.add_argument('--gamma', default=1, type=int,
                    metavar='gamma', help='gamma for the hard mining')
parser.add_argument('--square', default=1, type=int,
                    metavar='square', help='gamma for the hard mining')
parser.add_argument('--otri', default=1, type=int,
                    metavar='otri', help='otri for the hard mining')
parser.add_argument('--pl', default=1, type=int, metavar='pl', help='pl for the model')
parser.add_argument('--stage', default=23, type=int, metavar='stage', help='stage for the model')
parser.add_argument('--depth', default=2, type=int, metavar='depth', help='depth for the model')
parser.add_argument('--head', default=4, type=int, metavar='head', help='head for the model')
parser.add_argument('--goballoss', default=1.0, type=float, metavar='gid', help='use global loss and the weight')
parser.add_argument('--localloss', default=1.0, type=float, metavar='lid', help='use local loss and the weight')
parser.add_argument('--gobaltri', default=1.0, type=float, metavar='gtri', help='use global tri loss and the weight')
parser.add_argument('--localtri', default=0.0, type=float, metavar='ltri', help='use local tri loss and the weight')
parser.add_argument('--pha', default=1.0, type=float, metavar='pha', help='use pha')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
set_seed(args.seed)
dataset = args.dataset

base_path = "./sle_ckpt/"

if dataset == 'sysu':
    # path to your sysu-mm01 dataset
    data_path = '/SSD_dataset/CMReID/SYSU-MM01/ori_data/'
    log_path = base_path + dataset + "/" + args.model_path + '/' + args.log_path + 'sysu_log/'
    test_mode = [1, 2]  # thermal to visible
elif dataset == 'regdb':
    # path to your regdb dataset
    data_path = '/SSD_dataset/CMReID/RegDB/'
    log_path = base_path + dataset + "/" + args.model_path + '/' + args.log_path + 'regdb_log/'
    test_mode = [2, 1]  # visible to thermal
elif dataset == 'llcm':
    # path to your llcm dataset
    data_path = '/SSD_dataset/CMReID/LLCM/LLCM/'
    log_path = base_path + dataset + "/" + args.model_path + '/' + args.log_path + 'llcm_log/'
    test_mode = [2, 1]  # [2, 1]: VIS to IR; [1, 2]: IR to VIS

checkpoint_path = base_path + dataset + "/" + args.model_path + '/'

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.isdir(args.vis_log_path):
    os.makedirs(args.vis_log_path)

suffix = dataset
if args.method == 'adp':
    suffix = suffix + '_{}_joint_co_nog_ch_nog_sq{}'.format(args.method, args.square)
else:
    suffix = suffix + '_{}'.format(args.method)
# suffix = suffix + '_KL_{}'.format(args.kl)
if args.augc == 1:
    suffix = suffix + '_aug_G'
if args.rande > 0:
    suffix = suffix + '_erase_{}'.format(args.rande)

suffix = suffix + '_p{}_n{}_lr_{}_seed_{}_localtri_{}_otri_{}_stage_{}_depth_{}_head_{}_pha_{}'.format(
    args.num_pos, args.batch_size,
    args.lr, args.seed, args.localtri, args.otri, args.stage, args.depth, args.head, args.pha)

if not args.optim == 'sgd':
    suffix = suffix + '_' + args.optim

if dataset == 'regdb':
    suffix = suffix + '_trial_{}'.format(args.trial)

sys.stdout = Logger(log_path + suffix + '_os.txt')

vis_log_dir = args.vis_log_path + suffix + '/'

if not os.path.isdir(vis_log_dir):
    os.makedirs(vis_log_dir)
writer = SummaryWriter(vis_log_dir)
print("==========\nArgs:{}\n==========".format(args))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train_list = [
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize]

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize])

if args.rande > 0:
    transform_train_list = transform_train_list + [ChannelRandomErasing(probability=args.rande)]

if args.augc == 1:
    # transform_train_list = transform_train_list +  [ChannelAdap(probability =0.5)]
    transform_train_list = transform_train_list + [ChannelAdapGray(probability=0.5)]

transform_train = transforms.Compose(transform_train_list)

end = time.time()
if dataset == 'sysu':
    # training set
    trainset = SYSUData(data_path, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

elif dataset == 'regdb':
    # training set
    trainset = RegDBData(data_path, args.trial, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='visible')
    gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')

elif dataset == 'llcm':
    # training set
    trainset = LLCMData(data_path, args.trial, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label, query_cam = process_query_llcm(data_path, mode=test_mode[1])
    gall_img, gall_label, gall_cam = process_gallery_llcm(data_path, mode=test_mode[0], trial=0)

gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))

# testing data loader
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

n_class = len(np.unique(trainset.train_color_label))
nquery = len(query_label)
ngall = len(gall_label)

print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))
print('  thermal  | {:5d} | {:8d}'.format(n_class, len(trainset.train_thermal_label)))
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

print('==> Building model..')

net = basline_pcb(n_class, no_local='on', gm_pool='on', arch=args.arch, dataset=dataset, plearn=args.pl,
                  stage=args.stage, depth=args.depth, head=args.head)

net.to(device)
cudnn.benchmark = True



if dataset == 'sysu':
    model_path = "./baseline/sysu_adp_joint_co_nog_ch_nog_sq1_aug_G_erase_0.5_p4_n8_lr_0.1_seed_0_best.t"
if args.dataset == 'regdb':
    model_path = './baseline/regdb_'.format(args.trial)
if dataset == 'llcm':
    model_path = './baseline/llcm_'
if os.path.isfile(model_path):
    print('==> loading checkpoint {}'.format(model_path))
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['net'], strict=False)
    print('==> loaded checkpoint {}'
          .format(model_path))

assert os.path.isfile(model_path)

# define loss function
criterion_id = nn.CrossEntropyLoss()
if args.method == 'agw':
    pass
elif args.method == 'adp':
    # criterion_tri = TripletLoss_ADP(alpha=args.alpha, gamma=args.gamma, square=args.square)
    # criterion_tri_l = TripletLoss_ADP(alpha=args.alpha, gamma=args.gamma, square=args.square)
    if args.otri == 1:
        loader_batch = args.batch_size * args.num_pos
        criterion_tri = OriTripletLoss(batch_size=loader_batch, margin=args.margin)
        criterion_tri_l = OriTripletLoss(batch_size=loader_batch, margin=args.margin)

else:
    pass
criterion_kl = KLDivLoss()
criterion_id.to(device)
criterion_tri_l.to(device)
criterion_tri.to(device)
criterion_kl.to(device)

if args.optim == 'sgd':
    ignored_params = list(map(id, net.bottleneck.parameters())) \
                     + list(map(id, net.vit.parameters())) \
                     + list(map(id, net.classifier.parameters())) \
                     + list(map(id, net.local_conv_list.parameters())) \
                     + list(map(id, net.fc_list.parameters()))

    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

    optimizer = optim.SGD([
        {'params': base_params, 'lr': 0.1 * args.lr},
        {'params': net.bottleneck.parameters(), 'lr': args.lr},
        {'params': net.vit.parameters(), 'lr': args.lr},
        {'params': net.classifier.parameters(), 'lr': args.lr},
        {'params': net.local_conv_list.parameters(), 'lr': args.lr},
        {'params': net.fc_list.parameters(), 'lr': args.lr}
    ],
        weight_decay=5e-4, momentum=0.9, nesterov=True)


# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
    elif epoch >= 10 and epoch < 20:
        lr = args.lr
    elif epoch >= 20 and epoch < 50:
        lr = args.lr * 0.1
    elif epoch >= 50:
        lr = args.lr * 0.01

    optimizer.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]['lr'] = lr

    return lr


def train(epoch):
    current_lr = adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    id_loss_g = AverageMeter()
    id_loss_l = AverageMeter()
    tri_loss_g = AverageMeter()
    tri_loss_l = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0

    # switch to train mode
    net.train()
    end = time.time()

    for batch_idx, (input10, input11, input2, label1, label2) in enumerate(trainloader):

        labels = torch.cat((label1, label1, label2), 0)
        labels_vit = torch.cat((label1, label2), 0)
        input2 = Variable(input2.cuda())
        input10 = Variable(input10.cuda())
        input11 = Variable(input11.cuda())
        labels = Variable(labels.cuda())
        labels_vit = Variable(labels_vit.cuda())
        input1 = torch.cat((input10, input11,), 0)
        input2 = Variable(input2.cuda())
        data_time.update(time.time() - end)

        featG, outG, feat, out0, feat_all = net(input1, input2)
        b_x = featG.shape[0]
        b_x = b_x // 5

        loss_id = args.pha * criterion_id(out0[0][0:b_x * 3], labels) + (2.0 - args.pha) * criterion_id(
            out0[0][b_x * 3:], labels_vit)
        for i in range(len(feat) - 1):
            loss_id_temp = args.pha * criterion_id(out0[i + 1][0:b_x * 3], labels) + (2.0 - args.pha) * criterion_id(
                out0[i + 1][b_x * 3:], labels_vit)
            loss_id += loss_id_temp

        loss_tri_l = args.pha * criterion_tri_l(feat_all[0:b_x * 3], labels)[0] + (2.0 - args.pha) * \
                     criterion_tri_l(feat_all[b_x * 3:], labels_vit)[0]

        loss_id_G = args.pha * criterion_id(outG[0:b_x * 3], labels) + (2.0 - args.pha) * criterion_id(outG[b_x * 3:],
                                                                                                       labels_vit)
        loss_tri, batch_acc = criterion_tri(featG[0:b_x * 3], labels)
        loss_tri = args.pha * loss_tri
        loss_tri += (2.0 - args.pha) * criterion_tri(featG[b_x * 3:], labels_vit)[0]
        correct += (batch_acc / 2)
        _, predicted = outG[0:b_x * 3].max(1)
        correct += (predicted.eq(labels).sum().item() / 2)

        correct += batch_acc
        loss_id = args.localloss * loss_id
        loss_id_G = args.goballoss * loss_id_G
        loss_tri = args.gobaltri * loss_tri
        loss_tri_l = args.localtri * loss_tri_l

        loss = loss_id + loss_id_G + loss_tri + loss_tri_l

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update P
        train_loss.update(loss.item(), 2 * input1.size(0))

        id_loss_l.update(loss_id.item(), 2 * input1.size(0))
        id_loss_g.update(loss_id_G.item(), 2 * input1.size(0))

        tri_loss_g.update(loss_tri.item(), 2 * input1.size(0))
        tri_loss_l.update(loss_tri_l.item(), 2 * input1.size(0))

        total += labels.size(0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 50 == 0:
            print('Epoch: [{}][{}/{}] '
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'lr:{:.5f} '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                  'G-Loss: {id_loss_g.val:.4f} ({id_loss_g.avg:.4f}) '
                  'L-Loss: {id_loss_l.val:.4f} ({id_loss_l.avg:.4f}) '
                  'G-TLoss: {tri_loss_g.val:.4f} ({tri_loss_g.avg:.4f}) '
                  'L-TLoss: {tri_loss_l.val:.4f} ({tri_loss_l.avg:.4f}) '
                  'Accu: {:.2f}'.format(
                epoch, batch_idx, len(trainloader),
                current_lr,
                100. * correct / total, batch_time=batch_time,
                train_loss=train_loss,
                id_loss_g=id_loss_g,
                id_loss_l=id_loss_l,
                tri_loss_g=tri_loss_g,
                tri_loss_l=tri_loss_l))

    writer.add_scalar('total_loss', train_loss.avg, epoch)
    writer.add_scalar('id_loss_l', id_loss_l.avg, epoch)
    writer.add_scalar('id_loss_g', id_loss_g.avg, epoch)
    writer.add_scalar('tri_loss_g', tri_loss_g.avg, epoch)
    writer.add_scalar('tri_loss_l', tri_loss_l.avg, epoch)
    writer.add_scalar('lr', current_lr, epoch)


def test(epoch):
    # switch to evaluation mode
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall, 3584))
    gall_feat_att = np.zeros((ngall, 3584))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat, feat_att = net(input, input, test_mode[0])
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            gall_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    # switch to evaluation
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, 3584))
    query_feat_att = np.zeros((nquery, 3584))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat, feat_att = net(input, input, test_mode[1])
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            query_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()
    # compute the similarity
    distmat = np.matmul(query_feat, np.transpose(gall_feat))
    distmat_att = np.matmul(query_feat_att, np.transpose(gall_feat_att))

    # evaluation
    if dataset == 'regdb':
        cmc, mAP, mINP = eval_regdb(-distmat, query_label, gall_label)
        cmc_att, mAP_att, mINP_att = eval_regdb(-distmat_att, query_label, gall_label)
    elif dataset == 'sysu':
        cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
        cmc_att, mAP_att, mINP_att = eval_sysu(-distmat_att, query_label, gall_label, query_cam, gall_cam)
    elif dataset == 'llcm':
        cmc, mAP, mINP = eval_llcm(-distmat, query_label, gall_label, query_cam, gall_cam)
        cmc_att, mAP_att, mINP_att = eval_llcm(-distmat_att, query_label, gall_label, query_cam, gall_cam)

    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))

    writer.add_scalar('rank1', cmc[0], epoch)
    writer.add_scalar('mAP', mAP, epoch)
    writer.add_scalar('mINP', mINP, epoch)
    writer.add_scalar('rank1_att', cmc_att[0], epoch)
    writer.add_scalar('mAP_att', mAP_att, epoch)
    writer.add_scalar('mINP_att', mINP_att, epoch)
    return cmc, mAP, mINP, cmc_att, mAP_att, mINP_att


# training
print('==> Start Training...')
for epoch in range(start_epoch, 60):

    print('==> Preparing Data Loader...')
    # identity sampler
    sampler = IdentitySampler(trainset.train_color_label, \
                              trainset.train_thermal_label, color_pos, thermal_pos, args.num_pos, args.batch_size,
                              epoch)

    trainset.cIndex = sampler.index1  # color index
    trainset.tIndex = sampler.index2  # thermal index
    print(epoch)
    print(trainset.cIndex)
    print(trainset.tIndex)

    loader_batch = args.batch_size * args.num_pos

    trainloader = data.DataLoader(trainset, batch_size=loader_batch, \
                                  sampler=sampler, num_workers=args.workers, drop_last=True)

    # training
    train(epoch)

    if epoch >= 0:
        print('Test Epoch: {}'.format(epoch))

        # testing
        cmc, mAP, mINP, cmc_att, mAP_att, mINP_att = test(epoch)
        # save model
        if cmc_att[0] + mAP_att > best_acc:  # not the real best for sysu-mm01
            best_acc = cmc_att[0] + mAP_att
            best_epoch = epoch
            state = {
                'net': net.state_dict(),
                'cmc': cmc_att,
                'mAP': mAP_att,
                'mINP': mINP_att,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + '_best.t')

        print(
            'POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print(
            'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))
        print('Best Epoch [{}]'.format(best_epoch))