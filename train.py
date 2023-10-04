import torch
import torch.nn as nn
import os
import argparse
from data.comp_dataset import CompDataset
from models import networks, GenBoxNet
from utils.image_show import batch_coord2mask
from utils.calculate_loss import f_lossMSE, backward_D_basic


parser = argparse.ArgumentParser(description='')
parser.add_argument('--root', dest='root', type=str, default='', help='')
parser.add_argument('--phase', dest='phase', type=str, default='train', help='')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='')
parser.add_argument('--cup_num', type=int, default=1)
parser.add_argument('--datashuffle', dest='datashuffle', type=bool, default=True, help='')
parser.add_argument('--num_threads', default=1, type=int, help='')
parser.add_argument('--gpu', dest='gpu', type=int, default=0, help='gpu id')
parser.add_argument('--epoch', dest='epoch', type=int, default=3001, help='')
parser.add_argument('--lr', dest='lr', type=float, default=0.000005, help='')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='', help='models are saved here')
parser.add_argument('--loss', dest='loss', type=str, default='L1loss', help='')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='')
parser.add_argument('--resize', dest='resize', type=int, default=256, help='')
parser.add_argument('--lmd_m', type=float, default=1, help='')
parser.add_argument('--lmd_b', type=float, default=0.1, help='')
parser.add_argument('--lmd_g', type=float, default=0.1, help='')
parser.add_argument('--test_start', type=int, default=0)

if __name__ == '__main__':
    args = parser.parse_args()
    selfname = os.path.basename(__file__)
    print('Run file '+os.getcwd()+'/'+selfname)
    add_path = selfname.split('.')[0] + '_{}'.format(args.loss.split('loss')[0])
    add_path = add_path + '_m{}_b{}_g{}'.format(args.lmd_m, args.lmd_b, args.lmd_g)
    dir_cup = 'cup{}'.format(args.cup_num)
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, dir_cup, add_path)

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    dataset = CompDataset(args, phase='train', lbl='h_lbl')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                             num_workers=int(args.num_threads))
    dataset_size = len(dataloader)
    print('#training images = %d' % dataset_size)

    genbox_net = GenBoxNet(10).to(device)
    gencup_net = networks.define_G(8, 4, 64, 'resnet_atn', 'instance', False, 'normal', 0.02, [args.gpu]).to(device)
    D_box = networks.define_D(3, 64, 'basic', 3, 'instance', False, 'normal', 0.02, [args.gpu], padw=8).to(device)
    D_glb = networks.define_D(3, 64, 'basic', 3, 'instance', False, 'normal', 0.02, [args.gpu]).to(device)

    loss_MSE = nn.MSELoss()
    loss_L1 = nn.L1Loss()
    if args.loss == 'MSEloss':
        G_loss = loss_MSE
    else:
        G_loss = loss_L1
    opt_G_box = torch.optim.Adam(genbox_net.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    opt_G_cup = torch.optim.Adam(gencup_net.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    opt_D_box = torch.optim.Adam(D_box.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    opt_D_glb = torch.optim.Adam(D_glb.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    log_loss_G_cup, log_loss_G_box, log_loss_G_h, log_loss_G_db, log_loss_G_dg, log_loss_D_box, log_loss_D_glb = 0, 0, 0, 0, 0, 0, 0
    for epoch0 in range(args.epoch):
        if args.load_num <= 0:
            epoch = epoch0
        else:
            epoch = epoch0 + args.load_num

        genbox_net.train()
        gencup_net.train()
        D_box.train()
        D_glb.train()

        for i, data in enumerate(dataloader):
            img_c = data['img_c'].to(device)
            img_n = data['img_n'].to(device)
            path_n = data['path_n']
            img_h = data['img_h'].to(device)
            img_s = data['img_s'].to(device)
            coord_norm = data['box_xy_norm'].to(device)
            coord = data['box_xy']
            box_c = img_h[:, :, coord[0, 2]:coord[0, 3], coord[0, 0]:coord[0, 1]].to(device).detach()

            genbox_net.zero_grad()
            gencup_net.zero_grad()
            D_box.zero_grad()
            D_glb.zero_grad()

            gbox_net_in = torch.cat((img_c, img_n, img_s), 1)
            gbox_net_out = genbox_net(gbox_net_in)
            mask_fake = batch_coord2mask(gbox_net_out, path_n, resize=args.resize, rtn=True)

            net_in = torch.cat((img_c, mask_fake, img_n), 1)
            net_out = gencup_net(net_in)
            out_h = net_out[:, 0:3]
            alpha = (0.5*net_out[:, -1:]+0.5).repeat(1, 3, 1, 1)
            fake_h = out_h*alpha + (1-alpha)*img_n

            xys = coord_norm[0] * args.resize/2 + args.resize/2
            a0, a1, b0, b1 = xys.to(dtype=torch.int)
            box_fake = out_h[:, :, b0:b1, a0:a1].to(device)

            loss_G_box = args.lmd_m * loss_MSE(gbox_net_out, coord_norm)
            loss_G_db = args.lmd_b*f_lossMSE(loss_MSE, D_box(box_fake), 1.0, device)
            loss_G_dg = args.lmd_g*f_lossMSE(loss_MSE, D_glb(fake_h), 1.0, device)
            loss_G_h = G_loss(fake_h, img_h)
            loss_G_cup = loss_G_h + loss_G_db + loss_G_dg
            loss_G = loss_G_cup + loss_G_box

            loss_D_box = backward_D_basic(loss_MSE, D_box, box_c, box_fake, device)
            loss_D_glb = backward_D_basic(loss_MSE, D_glb, img_h, fake_h, device)

            loss_G.backward(retain_graph=True)
            loss_D_box.backward()
            loss_D_glb.backward()

            opt_G_box.step()
            opt_G_cup.step()
            opt_D_box.step()
            opt_D_glb.step()

            log_loss_G_box = loss_G_box.data.cpu().item()
            log_loss_G_cup = loss_G_cup.data.cpu().item()
            log_loss_G_h = loss_G_h.data.cpu().item()
            log_loss_G_db = loss_G_db.data.cpu().item()
            log_loss_G_dg = loss_G_dg.data.cpu().item()
            log_loss_D_box = loss_D_box.data.cpu().item()
            log_loss_D_glb = loss_D_glb.data.cpu().item()

            if i % int(dataset_size) == 0:
                print('epoch:{:4d}_{:3d};\t G_box:{:9f};\t G_cup:{:9f};\t G_h:{:9f};\t G_db:{:9f};\t G_dg:{:9f};\t D_box:{:9f};\t D_glb:{:9f};\t'.format(
                    epoch, i, log_loss_G_box, log_loss_G_cup, log_loss_G_h, log_loss_G_db, log_loss_G_dg, log_loss_D_box, log_loss_D_glb))

        if epoch % 10 == 0:
            torch.save(genbox_net.state_dict(), os.path.join(args.checkpoint_dir, 'gen_box_{}.pkl'.format(epoch)))
            torch.save(gencup_net.state_dict(), os.path.join(args.checkpoint_dir, 'gen_cup_{}.pkl'.format(epoch)))
            torch.save(D_box.state_dict(), os.path.join(args.checkpoint_dir, 'D_box_{}.pkl'.format(epoch)))
            torch.save(D_glb.state_dict(), os.path.join(args.checkpoint_dir, 'D_glb_{}.pkl'.format(epoch)))

