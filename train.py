import os
import time
from tqdm import trange
import random
import torch
from torch.optim import Adam
import utils
from args_fusion import args
import pytorch_loss
from model import DDcGAN
from datetime import datetime


def main():
    start_time = datetime.now()

    image_path = utils.list_images(args.base_images_ir)
    random.shuffle(image_path)

    batch_size = args.batch_size

    SwinFuse_model = DDcGAN(if_train=True)
    if args.cuda:
        SwinFuse_model.cuda()
    print(SwinFuse_model)

    loss_con_ssim = pytorch_loss.final_ssim
    loss_con_int = pytorch_loss.L_con_int
    loss_con_grad = pytorch_loss.L_con_grad
    loss_con = pytorch_loss.L_con
    loss_g_adv = pytorch_loss.L_adv_G
    loss_g = pytorch_loss.L_G
    loss_d_ir = pytorch_loss.L_D_ir
    loss_d_vis = pytorch_loss.L_D_vis

    tbar = trange(args.epochs)
    print('Start training.....')

    all_Loss_con_ssim_list = []
    all_Loss_con_int_list = []
    all_Loss_con_grad_list = []
    all_Loss_con_list = []
    all_Loss_g_adv_list = []
    all_Loss_g_list = []
    all_Loss_d_ir_list = []
    all_Loss_d_vis_list = []

    count_loss = 0
    all_Loss_con_ssim = 0.
    all_Loss_con_int = 0.
    all_Loss_con_grad = 0.
    all_Loss_con = 0.
    all_Loss_g_adv = 0.
    all_Loss_g = 0.
    all_Loss_d_ir = 0.
    all_Loss_d_vis = 0.
    epoch_num = 0
    for e in tbar:
        epoch_num = epoch_num + 1
        print('Epoch %d.....' % e)

        random.shuffle(image_path)

        base_images_ir_list = []
        detail_images_ir_list = []
        base_images_vis_list = []
        detail_images_vis_list = []
        images_ir_list = []
        images_vis_list = []

        for i in image_path:
            original_base_images_ir_path = os.path.join(args.base_images_ir, i)
            original_detail_images_ir_path = os.path.join(args.detail_images_ir, i)
            original_base_images_vis_path = os.path.join(args.base_images_vis, i)
            original_detail_images_vis_path = os.path.join(args.detail_images_vis, i)
            original_images_ir_path = os.path.join(args.images_ir, i)
            original_images_vis_path = os.path.join(args.images_vis, i)
            base_images_ir_list.append(original_base_images_ir_path)
            detail_images_ir_list.append(original_detail_images_ir_path)
            base_images_vis_list.append(original_base_images_vis_path)
            detail_images_vis_list.append(original_detail_images_vis_path)
            images_ir_list.append(original_images_ir_path)
            images_vis_list.append(original_images_vis_path)

        base_images_ir = utils.get_train_images_auto(base_images_ir_list)
        detail_images_ir = utils.get_train_images_auto(detail_images_ir_list)
        base_images_vis = utils.get_train_images_auto(base_images_vis_list)
        detail_images_vis = utils.get_train_images_auto(detail_images_vis_list)
        original_images_ir = utils.get_train_images_auto(images_ir_list)
        original_images_vis = utils.get_train_images_auto(images_vis_list)

        base_images_ir, batches = utils.load_dataset(base_images_ir, batch_size)
        detail_images_ir, batches = utils.load_dataset(detail_images_ir, batch_size)
        base_images_vis, batches = utils.load_dataset(base_images_vis, batch_size)
        detail_images_vis, batches = utils.load_dataset(detail_images_vis, batch_size)
        original_images_ir, batches = utils.load_dataset(original_images_ir, batch_size)
        original_images_vis, batches = utils.load_dataset(original_images_vis, batch_size)

        SwinFuse_model.train()
        count = 0
        print('begin batch training')

        for batch in range(batches):
            base_images_ir_input = base_images_ir[batch * batch_size:(batch * batch_size + batch_size)]
            detail_images_ir_input = detail_images_ir[batch * batch_size:(batch * batch_size + batch_size)]
            base_images_vis_input = base_images_vis[batch * batch_size:(batch * batch_size + batch_size)]
            detail_images_vis_input = detail_images_vis[batch * batch_size:(batch * batch_size + batch_size)]
            original_images_ir_input = original_images_ir[batch * batch_size:(batch * batch_size + batch_size)]
            original_images_vis_input = original_images_vis[batch * batch_size:(batch * batch_size + batch_size)]

            count += 1

            if args.cuda:
                base_images_ir_input = base_images_ir_input.cuda()
                detail_images_ir_input = detail_images_ir_input.cuda()
                base_images_vis_input = base_images_vis_input.cuda()
                detail_images_vis_input = detail_images_vis_input.cuda()
                original_images_ir_input = original_images_ir_input.cuda()
                original_images_vis_input = original_images_vis_input.cuda()

            for i in SwinFuse_model.G.parameters():
                i.requires_grad = False
            for i in SwinFuse_model.D_ir.parameters():
                i.requires_grad = False
            for i in SwinFuse_model.D_vis.parameters():
                i.requires_grad = True
            optimizer_vis = Adam(SwinFuse_model.parameters(), args.lr)
            for _ in range(0, args.max_epoch + 1):
                score_vis, _, score_g_vis, _, _ = SwinFuse_model(base_images_ir_input,
                                                                                      detail_images_ir_input,
                                                                                      base_images_vis_input,
                                                                                      detail_images_vis_input,
                                                                                      original_images_ir_input,
                                                                                      original_images_vis_input)
                Loss_d_vis = loss_d_vis(score_vis, score_g_vis)
                if Loss_d_vis <= args.l_max:
                    break
                optimizer_vis.zero_grad()
                Loss_d_vis.backward()
                optimizer_vis.step()

            for i in SwinFuse_model.G.parameters():
                i.requires_grad = False
            for i in SwinFuse_model.D_ir.parameters():
                i.requires_grad = True
            for i in SwinFuse_model.D_vis.parameters():
                i.requires_grad = False
            optimizer_ir = Adam(SwinFuse_model.parameters(), args.lr)
            for _ in range(0, args.max_epoch + 1):
                _, score_ir, _, score_g_ir, _ = SwinFuse_model(base_images_ir_input,
                                                                                      detail_images_ir_input,
                                                                                      base_images_vis_input,
                                                                                      detail_images_vis_input,
                                                                                      original_images_ir_input,
                                                                                      original_images_vis_input)
                Loss_d_ir = loss_d_ir(score_ir, score_g_ir)
                if Loss_d_ir <= args.l_max:
                    break
                optimizer_ir.zero_grad()
                Loss_d_ir.backward()
                optimizer_ir.step()

            score_vis, score_ir, score_g_vis, score_g_ir, fusion = SwinFuse_model(base_images_ir_input,
                                                                                      detail_images_ir_input,
                                                                                      base_images_vis_input,
                                                                                      detail_images_vis_input,
                                                                                      original_images_ir_input,
                                                                                      original_images_vis_input)
            Loss_g = loss_g(original_images_ir_input, original_images_vis_input, fusion, args.a, args.b, args.alpha,
                                args.beta, args.gamma, score_g_ir, score_g_vis, args.lamda)

            L_G_max = 0.8 * Loss_g

            for i in SwinFuse_model.G.parameters():
                i.requires_grad = True
            for i in SwinFuse_model.D_ir.parameters():
                i.requires_grad = False
            for i in SwinFuse_model.D_vis.parameters():
                i.requires_grad = False
            optimizer_g = Adam(SwinFuse_model.parameters(), args.lr)
            for _ in range(0, args.max_epoch + 1):
                score_vis, score_ir, score_g_vis, score_g_ir, fusion = SwinFuse_model(base_images_ir_input,
                                                                                      detail_images_ir_input,
                                                                                      base_images_vis_input,
                                                                                      detail_images_vis_input,
                                                                                      original_images_ir_input,
                                                                                      original_images_vis_input)
                Loss_g = loss_g(original_images_ir_input, original_images_vis_input, fusion, args.a, args.b,
                                args.alpha,
                                args.beta, args.gamma, score_g_ir, score_g_vis, args.lamda)

                optimizer_g.zero_grad()
                Loss_g.backward()
                optimizer_g.step()
                if Loss_g <= L_G_max:
                    break

            Loss_con_ssim = loss_con_ssim(original_images_ir_input, original_images_vis_input, fusion)
            Loss_con_int = loss_con_int(original_images_ir_input, original_images_vis_input, fusion, args.a)
            Loss_con_grad = loss_con_grad(original_images_ir_input, original_images_vis_input, fusion, args.b)
            Loss_con = loss_con(original_images_ir_input, original_images_vis_input, fusion, args.a, args.b, args.alpha,
                                args.beta, args.gamma)
            Loss_g_adv = loss_g_adv(score_g_ir, score_g_vis)
            Loss_g = loss_g(original_images_ir_input, original_images_vis_input, fusion, args.a, args.b, args.alpha,
                            args.beta, args.gamma, score_g_ir, score_g_vis, args.lamda)
            Loss_d_ir = loss_d_ir(score_ir, score_g_ir)
            Loss_d_vis = loss_d_vis(score_vis, score_g_vis)

            all_Loss_con_ssim += Loss_con_ssim.item()
            all_Loss_con_int += Loss_con_int.item()
            all_Loss_con_grad += Loss_con_grad.item()
            all_Loss_con += Loss_con.item()
            all_Loss_g_adv += Loss_g_adv.item()
            all_Loss_g += Loss_g.item()
            all_Loss_d_ir += Loss_d_ir.item()
            all_Loss_d_vis += Loss_d_vis.item()

            if (batch + 1) % args.log_interval == 0:
                elapsed_time = datetime.now() - start_time
                print("lr: %s, elapsed_time: %s\n" % (args.lr, elapsed_time))
                mesg = "{}\tEpoch {}:\t[{}/{}]\t Loss_con_ssim: {:.6f}\t Loss_con_int: {:.6f}\t " \
                       "Loss_con_grad: {:.6f}\t Loss_con: {:.6f}\t Loss_g_adv: {:.6f}\t all_Loss_g: {:.6f}\t " \
                       "all_Loss_d_ir: {:.6f}\t all_Loss_d_vis: {:.6f}".format(
                                  time.ctime(), e + 1, count, batches,
                                  all_Loss_con_ssim / args.log_interval,
                                  all_Loss_con_int / args.log_interval,
                                  all_Loss_con_grad / args.log_interval,
                                  all_Loss_con / args.log_interval,
                                  all_Loss_g_adv / args.log_interval,
                                  all_Loss_g / args.log_interval,
                                  all_Loss_d_ir / args.log_interval,
                                  all_Loss_d_vis / args.log_interval)
                tbar.set_description(mesg)
                all_Loss_con_ssim_list.append(all_Loss_con_ssim / args.log_interval)
                all_Loss_con_int_list.append(all_Loss_con_int / args.log_interval)
                all_Loss_con_grad_list.append(all_Loss_con_grad / args.log_interval)
                all_Loss_con_list.append(all_Loss_con / args.log_interval)
                all_Loss_g_adv_list.append(all_Loss_g_adv / args.log_interval)
                all_Loss_g_list.append(all_Loss_g / args.log_interval)
                all_Loss_d_ir_list.append(all_Loss_d_ir / args.log_interval)
                all_Loss_d_vis_list.append(all_Loss_d_vis / args.log_interval)

                count_loss = count_loss + 1

                all_Loss_con_ssim = 0.
                all_Loss_con_int = 0.
                all_Loss_con_grad = 0.
                all_Loss_con = 0.
                all_Loss_g_adv = 0.
                all_Loss_g = 0.
                all_Loss_d_ir = 0.
                all_Loss_d_vis = 0.

            if (batch + 1) % args.log_save_model_interval == 0:
                save_model_filename = "epoch" + str(epoch_num) + "_" + "batch" + str(batch+1) + "_" + \
                                      str(time.ctime()).replace(' ', '_').replace(':', '_') + ".model"
                save_model_path = os.path.join(args.save_model_dir, save_model_filename)
                torch.save(SwinFuse_model.state_dict(), save_model_path)
                print("\nDone, trained model saved at", save_model_path)

    SwinFuse_model.eval()
    SwinFuse_model.cpu()


if __name__ == "__main__":
    main()
