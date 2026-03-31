import argparse
import yaml
import os
import os.path as osp
import pprint

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

import time
import numpy as np
import pandas as pd
import torch.nn.functional as F

from augseg.utils.dist_helper import setup_distributed
from augseg.utils.utils import set_random_seed, setup_default_logging
from augseg.models.model_helper import ModelBuilder
from augseg.utils.loss_helper import get_criterion
from augseg.dataset.builder import get_loader
from augseg.utils.lr_helper import get_optimizer, get_scheduler
from augseg.utils.utils import load_state, AverageMeter, intersectionAndUnion
from augseg.dataset.augs_ALIA import cut_mix_label_adaptive
from augseg.utils.loss_helper import compute_unsupervised_loss_by_threshold

# ---------------- Aa op mapping + strength normalize (0..1) ----------------
AA_OPS = {
    1: "identity",
    2: "autocontrast",
    3: "equalize",
    4: "blur",
    5: "contrast",
    6: "brightness",
    7: "color",
    8: "sharpness",
    9: "posterize",
    10: "solarize",
    11: "hue",
}

def aa_strength(k_id: int, t: float) -> float:
    """
    Chuẩn hoá intensity về [0,1] (0=nhẹ, 1=mạnh).
    Giải quyết vấn đề: thang đo khác nhau + đảo chiều (solarize/posterize).
    """
    import math
    if t is None or (isinstance(t, float) and math.isnan(t)):
        return float("nan")

    # blur sigma: càng lớn càng mạnh
    if k_id == 4:
        return float((t - 0.1) / (2.0 - 0.1))

    # contrast/brightness/color/sharpness: factor (<=1) càng nhỏ càng mạnh
    if k_id in (5, 6, 7, 8):
        vmin, vmax = 0.05, 0.95
        v = max(vmin, min(vmax, float(t)))
        return float((1.0 - v) / (1.0 - vmin))

    # posterize bits: bits càng thấp càng mạnh
    if k_id == 9:
        b = int(round(float(t)))
        b = max(1, min(8, b))
        return float((8 - b) / 7.0)

    # solarize threshold: threshold càng thấp càng mạnh
    if k_id == 10:
        thr = int(round(float(t)))
        thr = max(1, min(256, thr))
        return float((256 - thr) / 255.0)

    # hue: |hue| càng lớn càng mạnh
    if k_id == 11:
        v = max(-0.5, min(0.5, float(t)))
        return float(abs(v) / 0.5)

    return float("nan")

def read_run_id_from_ckpt(ckpt_path, fallback_run_id):
    """
    Đọc run_id từ checkpoint.
    Nếu checkpoint chưa có run_id thì dùng fallback_run_id.
    """
    print(f"[DEBUG] loading ckpt: {ckpt_path}", flush=True)

    if not os.path.exists(ckpt_path):
        return fallback_run_id

    t0 = time.time()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"[DEBUG] torch.load finished in {time.time() - t0:.2f}s", flush=True)

    return ckpt.get("run_id", fallback_run_id)
    
def trim_iter_csv_for_resume(train_iter_csv, last_epoch, logger = None):
    """
    Xóa các dòng iter log của epoch chưa hoàn tất để tránh duplicate khi resume.

    Quy ước:
    - nếu last_epoch = 7
    - nghĩa là checkpoint hiện tại tương ứng trạng thái sau epoch 6
    - vậy chỉ giữ meta/epoch < 7
    """
    if train_iter_csv is None:
        return
    
    if not os.path.exists(train_iter_csv):
        return

    try: 
        df = pd.read_csv(train_iter_csv)

        if "meta/epoch" not in df.columns:
            if logger is not None:
                logger.info(
                    f"[resume-trim] bỏ qua trim vì file không có cột meta/epoch: {train_iter_csv}"
                )
            return
            
        old_len = len(df)

        df = df[df["meta/epoch"] < last_epoch].copy()

        new_len = len(df)
        removed = old_len - new_len
        
        df.to_csv(train_iter_csv, index = False)

        if logger is not None:
            logger.info(
                f"[resume-trim] file={train_iter_csv}, last_epoch={last_epoch}, "
                f"removed_rows={removed}, remaining_rows={new_len}"
            )
    except Exception as e:
        if logger is not None:
            logger.info(f"[resume-trim] lỗi khi trim iter csv: {e}")

def build_iter_log_columns():
    cols = [
        # meta
        "meta/log_type",
        "meta/epoch",
        "meta/iter_in_epoch",
        "meta/global_iter",

        # core training
        "iter/sup_loss",
        "iter/uns_loss",
        "iter/pseudo_high_ratio",
        "iter/lr",

        # validation
        "val/model",
        "val/class_id",
        "val/loss",
        "val/miou",
        "val/class_iou",
        "val/best_miou",

        # Ar
        "ar/triggered",
        "ar/applied",
        "ar/area_ratio_est",

        # U
        "u/entropy_mean",
        "u/maxprob_mean",
        "u/maxprob_p10",
        "u/maxprob_p50",
        "u/maxprob_p90",
        "u/pseudo_ratio_mean",

        # Aa global
        "aa/ops_per_image",
    ]

    for kid in range(1, 12):
        op_name = AA_OPS.get(kid, f"op{kid}")
        cols.extend([
            f"aa/op_rate/{op_name}",
            f"aa/count/{op_name}",
            f"aa/pct_mean/{op_name}",
            f"aa/pct_max/{op_name}",
            f"aa/t_mean/{op_name}",
            f"aa/t_std/{op_name}",
        ])

    return cols


ITER_LOG_COLUMNS = build_iter_log_columns()


def make_default_iter_log_dict(
    sup_loss,
    uns_loss,
    pseudo_high_ratio,
    lr,
    epoch,
    step,
    global_iter,
    ar_triggered,
    ar_applied,
    ar_area_ratio_est,
    u_entropy_mean,
    u_maxprob_mean,
    u_maxprob_p10,
    u_maxprob_p50,
    u_maxprob_p90,
    u_pseudo_ratio_mean,
):
    log_dict = {
        # meta
        "meta/log_type": "train_iter",
        "meta/epoch": int(epoch),
        "meta/iter_in_epoch": int(step),
        "meta/global_iter": int(global_iter),

        # core training
        "iter/sup_loss": float(sup_loss),
        "iter/uns_loss": float(uns_loss),
        "iter/pseudo_high_ratio": float(pseudo_high_ratio),
        "iter/lr": float(lr),

        # validation default
        "val/model": "",
        "val/class_id": -1,
        "val/loss": float("nan"),
        "val/miou": float("nan"),
        "val/class_iou": float("nan"),
        "val/best_miou": float("nan"),

        # Ar
        "ar/triggered": int(ar_triggered),
        "ar/applied": int(ar_applied),
        "ar/area_ratio_est": float(ar_area_ratio_est),

        # U
        "u/entropy_mean": float(u_entropy_mean),
        "u/maxprob_mean": float(u_maxprob_mean),
        "u/maxprob_p10": float(u_maxprob_p10),
        "u/maxprob_p50": float(u_maxprob_p50),
        "u/maxprob_p90": float(u_maxprob_p90),
        "u/pseudo_ratio_mean": float(u_pseudo_ratio_mean),

        # Aa global default
        "aa/ops_per_image": 0.0,
    }

    for kid in range(1, 12):
        op_name = AA_OPS.get(kid, f"op{kid}")
        log_dict[f"aa/op_rate/{op_name}"] = 0.0
        log_dict[f"aa/count/{op_name}"] = 0
        log_dict[f"aa/pct_mean/{op_name}"] = -1.0
        log_dict[f"aa/pct_max/{op_name}"] = -1.0
        log_dict[f"aa/t_mean/{op_name}"] = -1.0
        log_dict[f"aa/t_std/{op_name}"] = -1.0

    return log_dict

def main(in_args):
    args = in_args
    if args.seed is not None:
        # print("set random seed to", args.seed)
        set_random_seed(args.seed, deterministic=True)
        # set_random_seed(args.seed)
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    rank, word_size = setup_distributed(port=args.port)

    # ✅ đặt ở đây
    if rank != 0:
        os.environ["WANDB_MODE"] = "disabled"
    
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    torch.cuda.set_device(local_rank)

    ###########################
    # 1. output settings
    ###########################
    cfg["exp_path"] = osp.dirname(args.config)
    cfg["save_path"] = osp.join(cfg["exp_path"], cfg["saver"]["snapshot_dir"])
    cfg["log_path"] = osp.join(cfg["exp_path"], "log")
    flag_use_tb = cfg["saver"]["use_tb"]
    
    if not os.path.exists(cfg["log_path"]) and rank == 0:
        os.makedirs(cfg["log_path"])
    if not osp.exists(cfg["save_path"]) and rank == 0:
        os.makedirs(cfg["save_path"])

    if rank == 0:
        logger, curr_timestr = setup_default_logging("global", cfg["log_path"])
    else:
        logger, curr_timestr = None, ""
    
    # mặc định: coi như đây là một run mới
    run_id = curr_timestr

    # checkpoint latest
    resume_ckpt = os.path.join(cfg["save_path"], "ckpt.pth")
    
    # nếu bật auto_resume và checkpoint đã tồn tại,
    # thử đọc run_id cũ để nối lại đúng log cũ
    if cfg["saver"].get("auto_resume", False) and os.path.exists(resume_ckpt):
        print(f"[DEBUG] resume_ckpt = {resume_ckpt}", flush=True)
        run_id = read_run_id_from_ckpt(resume_ckpt,  curr_timestr)

    # dùng run_id để đặt tên CSV
    if rank == 0:
        csv_path = os.path.join(cfg["log_path"], f"seg_{run_id}_stat.csv")
        train_iter_csv = os.path.join(cfg["log_path"], f"train_iter_{run_id}.csv")
    else:
        csv_path = None
        train_iter_csv = None

    # tensorboard: hiện tại cứ để theo launch mới cho an toàn
    if rank == 0:
        logger.info("{}".format(pprint.pformat(cfg)))
        logger.info(f"[log] run_id = {run_id}")
        if flag_use_tb:
            tb_logger = SummaryWriter(osp.join(cfg["log_path"], "events_seg", curr_timestr))
        else:
            tb_logger = None
    else:
        tb_logger = None

    # ---------------- W&B (wandb) ----------------
    wandb_run = None
    use_wandb = cfg.get("wandb", {}).get("enable", False)

    if use_wandb:
        if rank != 0:
            # DDP: chỉ rank0 log, rank khác tắt để khỏi spam nhiều runs
            os.environ["WANDB_MODE"] = "disabled"
        else:
            import wandb
            wandb_run = wandb.init(
                project=cfg.get("wandb", {}).get("project", "AugsegResearch"),
                entity=cfg.get("wandb", {}).get("entity", None),
                name=cfg.get("wandb", {}).get("name", run_id),
                config=cfg,
                dir=cfg["log_path"],
                settings=wandb.Settings(start_method="thread"),
            )

    # make sure all folders and csv handler are correctly created on rank ==0.
    dist.barrier(device_ids=[local_rank])

    ###########################
    # 2. prepare model 1
    ###########################
    model = ModelBuilder(cfg["net"])
    modules_back = [model.encoder]
    modules_head = [model.decoder]
    if cfg["net"].get("aux_loss", False):
        modules_head.append(model.auxor)
    if cfg["net"].get("sync_bn", True):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)

    ###########################
    # 3. data
    ###########################
    sup_loss_fn = get_criterion(cfg)
    train_loader_sup, train_loader_unsup, val_loader = get_loader(cfg, seed=args.seed)

    ##############################
    # 4. optimizer & scheduler
    ##############################
    cfg_trainer = cfg["trainer"]
    cfg_optim = cfg_trainer["optimizer"]
    times = 10 if "pascal" in cfg["dataset"]["type"] else 1

    params_list = []
    for module in modules_back:
        params_list.append(
            dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"])
        )
    for module in modules_head:
        params_list.append(
            dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"] * times)
        )
    optimizer = get_optimizer(params_list, cfg_optim)

    ###########################
    # 5. prepare model more
    ###########################
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )

    # Teacher model -- freeze training
    model_teacher = ModelBuilder(cfg["net"])
    if cfg["net"].get("sync_bn", True):
        model_teacher = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_teacher)
    model_teacher.cuda(local_rank)
    model_teacher = torch.nn.parallel.DistributedDataParallel(
        model_teacher,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )
    for p in model_teacher.parameters():
        p.requires_grad = False

    # initialize teacher model -- not neccesary if using warmup
    with torch.no_grad():
        for t_params, s_params in zip(model_teacher.parameters(), model.parameters()):
            t_params.data = s_params.data
        
    ######################################
    # 6. resume
    ######################################
    last_epoch = 0
    best_prec = 0
    best_epoch = -1
    best_prec_stu = 0
    best_epoch_stu = -1
    # auto_resume > pretrain
    if cfg["saver"].get("auto_resume", False):
        lastest_model = os.path.join(cfg["save_path"], "ckpt.pth")
        if not os.path.exists(lastest_model):
            "No checkpoint found in '{}'".format(lastest_model)
        else:
            print(f"Resume model from: '{lastest_model}'")
            best_prec, last_epoch = load_state(
                lastest_model, model, optimizer=optimizer, key="model_state"
            )
            _, _ = load_state(
                lastest_model, model_teacher, optimizer=optimizer, key="teacher_state"
            )

            # rất quan trọng:
            # checkpoint resume từ đầu epoch last_epoch,
            # nên phải xóa log iter của epoch chưa hoàn tất để tránh duplicate
            if rank == 0:
                trim_iter_csv_for_resume(train_iter_csv, last_epoch, logger)

    optimizer_start = get_optimizer(params_list, cfg_optim)
    lr_scheduler = get_scheduler(
        cfg_trainer, len(train_loader_sup), optimizer_start, start_epoch=last_epoch
    )

    ######################################
    # 7. training loop
    ######################################
    if rank == 0:
        logger.info('-------------------------- start training --------------------------')
    # Start to train model
    for epoch in range(last_epoch, cfg_trainer["epochs"]):
        # Training
        res_loss_sup, res_loss_unsup = train(
            model,
            model_teacher,
            optimizer,
            lr_scheduler,
            sup_loss_fn,
            train_loader_sup,
            train_loader_unsup,
            epoch,
            tb_logger,
            logger,
            cfg,
            wandb_run,   # <-- thêm dòng này
            train_iter_csv=train_iter_csv,
        )

        # Validation and store checkpoint
        if "cityscapes" in cfg["dataset"].get("type", "pascal"):
            if epoch % 10 == 0 or epoch > (cfg_trainer["epochs"]-50):
                if cfg_trainer.get("evaluate_student", True):
                    val_stu = validate_citys(model, val_loader, epoch, logger, cfg, sup_loss_fn)
                    prec_stu = val_stu["miou"]
                else:
                    val_stu = None
                    prec_stu = -1000.0

                val_tea = validate_citys(model_teacher, val_loader, epoch, logger, cfg, sup_loss_fn)
                prec_tea = val_tea["miou"]
                prec = prec_tea
            else:
                val_stu = None
                val_tea = None
                prec_stu = -1000.0
                prec_tea = -1000.0
                prec = prec_tea
        else:
            if cfg_trainer.get("evaluate_student", True):
                val_stu = validate(model, val_loader, epoch, logger, cfg, sup_loss_fn)
                prec_stu = val_stu["miou"]
            else:
                val_stu = None
                prec_stu = -1000.0

            val_tea = validate(model_teacher, val_loader, epoch, logger, cfg, sup_loss_fn)
            prec_tea = val_tea["miou"]
            prec = prec_tea

        if rank == 0:
            state = {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "teacher_state": model_teacher.state_dict(),
                "best_miou": best_prec,
                "run_id": run_id,
            }
            if prec_stu > best_prec_stu:
                best_prec_stu = prec_stu
                best_epoch_stu = epoch

            if prec > best_prec:
                best_prec = prec
                best_epoch = epoch
                state["best_miou"] = prec
                torch.save(state, osp.join(cfg["save_path"], "ckpt_best.pth"))

            torch.save(state, osp.join(cfg["save_path"], "ckpt.pth"))
            # save statistics
            tmp_results = {
                        'loss_lb': res_loss_sup,
                        'loss_ub': res_loss_unsup,
                        'miou_stu': prec_stu,
                        'miou_tea': prec_tea,
                        "best": best_prec,
                        "best-stu":best_prec_stu}
            data_frame = pd.DataFrame(data=tmp_results, index=range(epoch, epoch+1))
            if epoch > 0 and osp.exists(csv_path):
                data_frame.to_csv(csv_path, mode='a', header=False, index_label='epoch')
            else:
                data_frame.to_csv(csv_path, index_label='epoch')

            global_step = int((epoch + 1) * len(train_loader_sup) - 1)

            # ---------------- val_epoch: student ----------------
            if (train_iter_csv is not None) and (val_stu is not None):
                row = {c: np.nan for c in ITER_LOG_COLUMNS}
                row["meta/log_type"] = "val_epoch"
                row["meta/epoch"] = int(epoch)
                row["meta/iter_in_epoch"] = -1
                row["meta/global_iter"] = int(global_step)

                row["val/model"] = "student"
                row["val/class_id"] = -1
                row["val/loss"] = float(val_stu["loss"])
                row["val/miou"] = float(val_stu["miou"])
                row["val/class_iou"] = np.nan
                row["val/best_miou"] = float(best_prec_stu)

                df_row = pd.DataFrame([row], columns=ITER_LOG_COLUMNS)
                if os.path.exists(train_iter_csv):
                    df_row.to_csv(train_iter_csv, mode="a", header=False, index=False)
                else:
                    df_row.to_csv(train_iter_csv, index=False)

            # ---------------- val_epoch: teacher ----------------
            if (train_iter_csv is not None) and (val_tea is not None):
                row = {c: np.nan for c in ITER_LOG_COLUMNS}
                row["meta/log_type"] = "val_epoch"
                row["meta/epoch"] = int(epoch)
                row["meta/iter_in_epoch"] = -1
                row["meta/global_iter"] = int(global_step)

                row["val/model"] = "teacher"
                row["val/class_id"] = -1
                row["val/loss"] = float(val_tea["loss"])
                row["val/miou"] = float(val_tea["miou"])
                row["val/class_iou"] = np.nan
                row["val/best_miou"] = float(best_prec)

                df_row = pd.DataFrame([row], columns=ITER_LOG_COLUMNS)
                if os.path.exists(train_iter_csv):
                    df_row.to_csv(train_iter_csv, mode="a", header=False, index=False)
                else:
                    df_row.to_csv(train_iter_csv, index=False)

            # ---------------- val_class: student ----------------
            if (train_iter_csv is not None) and (val_stu is not None):
                for class_id, class_iou in enumerate(val_stu["iou_class"]):
                    row = {c: np.nan for c in ITER_LOG_COLUMNS}
                    row["meta/log_type"] = "val_class"
                    row["meta/epoch"] = int(epoch)
                    row["meta/iter_in_epoch"] = -1
                    row["meta/global_iter"] = int(global_step)

                    row["val/model"] = "student"
                    row["val/class_id"] = int(class_id)
                    row["val/miou"] = float(val_stu["miou"])
                    row["val/class_iou"] = float(class_iou)
                    row["val/best_miou"] = float(best_prec_stu)

                    df_row = pd.DataFrame([row], columns=ITER_LOG_COLUMNS)
                    if os.path.exists(train_iter_csv):
                        df_row.to_csv(train_iter_csv, mode="a", header=False, index=False)
                    else:
                        df_row.to_csv(train_iter_csv, index=False)

            # ---------------- val_class: teacher ----------------
            if (train_iter_csv is not None) and (val_tea is not None):
                for class_id, class_iou in enumerate(val_tea["iou_class"]):
                    row = {c: np.nan for c in ITER_LOG_COLUMNS}
                    row["meta/log_type"] = "val_class"
                    row["meta/epoch"] = int(epoch)
                    row["meta/iter_in_epoch"] = -1
                    row["meta/global_iter"] = int(global_step)

                    row["val/model"] = "teacher"
                    row["val/class_id"] = int(class_id)
                    row["val/loss"] = np.nan
                    row["val/miou"] = float(val_tea["miou"])
                    row["val/class_iou"] = float(class_iou)
                    row["val/best_miou"] = float(best_prec)

                    df_row = pd.DataFrame([row], columns=ITER_LOG_COLUMNS)
                    if os.path.exists(train_iter_csv):
                        df_row.to_csv(train_iter_csv, mode="a", header=False, index=False)
                    else:
                        df_row.to_csv(train_iter_csv, index=False)
            
            logger.info(" <<Test>> - Epoch: {}.  MIoU: {:.2f}/{:.2f}.  \033[34mBest-STU:{:.2f}/{}  \033[31mBest-EMA: {:.2f}/{}\033[0m".format(epoch, 
                prec_stu * 100, prec_tea * 100, best_prec_stu * 100, best_epoch_stu, best_prec * 100, best_epoch))
            if tb_logger is not None:
                tb_logger.add_scalar("mIoU val", prec, epoch)

            if (rank == 0) and (wandb_run is not None):
                global_step = int((epoch + 1) * len(train_loader_sup) - 1)
                wandb_run.log({
                    "val/miou_stu": float(prec_stu),
                    "val/miou_tea": float(prec_tea),
                    "epoch/loss_sup": float(res_loss_sup),
                    "epoch/loss_unsup": float(res_loss_unsup),
                    "meta/epoch": int(epoch + 1),
                }, step=global_step)
            
    if (rank == 0) and (wandb_run is not None):
        wandb_run.finish()

    if dist.is_available() and dist.is_initialized():
        dist.barrier(device_ids=[local_rank])
        dist.destroy_process_group()





def train(
    model,
    model_teacher,
    optimizer,
    lr_scheduler,
    sup_loss_fn,
    loader_l,
    loader_u,
    epoch,
    tb_logger,
    logger,
    cfg,
    wandb_run=None,   # <-- thêm
    train_iter_csv=None
):

    local_rank = torch.cuda.current_device()
    ema_decay_origin = cfg["net"]["ema_decay"]
    rank, world_size = dist.get_rank(), dist.get_world_size()
    log_every = int(cfg.get("wandb", {}).get("log_every", 50))
    flag_extra_weak = cfg["trainer"]["unsupervised"].get("flag_extra_weak", False)
    model.train()
    
    # data loader
    loader_l.sampler.set_epoch(epoch)
    loader_u.sampler.set_epoch(epoch)
    loader_l_iter = iter(loader_l)
    loader_u_iter = iter(loader_u)
    assert len(loader_l) == len(loader_u), f"labeled data {len(loader_l)} unlabeled data {len(loader_u)}, mixmatch!"

    # metric indicators
    sup_losses = AverageMeter(20)
    uns_losses = AverageMeter(20)
    batch_times = AverageMeter(20)
    learning_rates = AverageMeter(20)
    meter_high_pseudo_ratio = AverageMeter(20)
    
    # print freq 8 times for a epoch
    print_freq = len(loader_u) // 8 # 8 for semi 4 for sup
    print_freq_lst = [i * print_freq for i in range(1,8)]
    print_freq_lst.append(len(loader_u) -1)

    # start iterations
    model.train()
    model_teacher.eval()
    for step in range(len(loader_l)):
        batch_start = time.time()
        # --------- init per-iter stats for wandb (tránh dính iter trước) ---------
        ar_triggered = 0
        ar_applied = 0
        ar_area_ratio_est = float("nan")

        u_entropy_mean = float("nan")
        u_maxprob_mean = float("nan")
        u_maxprob_p10 = float("nan")
        u_maxprob_p50 = float("nan")
        u_maxprob_p90 = float("nan")
        u_pseudo_ratio_mean = float("nan")

        i_iter = epoch * len(loader_l) + step # total iters till now
        # log schedule (đúng y hệt block W&B phía dưới)
        do_log_now = (rank == 0) and (
            (i_iter % log_every == 0) or (step == len(loader_l) - 1)
        )
        lr = lr_scheduler.get_lr()
        learning_rates.update(lr[0])
        lr_scheduler.step() # lr is updated at the iteration level

        name, image_l, label_l = next(loader_l_iter)
        image_l = image_l.cuda(local_rank, non_blocking=True)
        label_l = label_l.cuda(local_rank, non_blocking=True)

        num_classes = cfg["net"]["num_classes"]
        ignore = cfg["dataset"]["ignore_label"]

        bad = (label_l != ignore) & ((label_l < 0) | (label_l >= num_classes))
        if bad.any() and rank == 0:
            print("❌ BAD sample id:", name)
            print("unique bad values:", torch.unique(label_l[bad])[:50].tolist())
            print("label min/max:", label_l.min().item(), label_l.max().item())
            raise RuntimeError("Out-of-range labels in supervised mask")

        batch_u = next(loader_u_iter)

        # batch_u có thể là:
        # - cũ: (idx, weak, strong, label)
        # - mới: (idx, weak, strong, label, k_ids, t_vals)
        if len(batch_u) == 4:
            _, image_u_weak, image_u_aug, _ = batch_u
            k_ids, t_vals = None, None
        else:
            _, image_u_weak, image_u_aug, _, k_ids, t_vals = batch_u

        image_u_weak = image_u_weak.cuda(local_rank, non_blocking=True)
        image_u_aug  = image_u_aug.cuda(local_rank, non_blocking=True)
    
        
        # start the training
        if epoch < cfg["trainer"].get("sup_only_epoch", 0):
            # forward
            pred, aux = model(image_l)
            # supervised loss
            if "aux_loss" in cfg["net"].keys():
                sup_loss = sup_loss_fn([pred, aux], label_l)
                del aux
            else:
                sup_loss = sup_loss_fn(pred, label_l)
                del pred

            # no unlabeled data during the warmup period
            unsup_loss = torch.tensor(0.0).cuda()
            pseduo_high_ratio = torch.tensor(0.0).cuda()

        else:
            # 1. generate pseudo labels
            p_threshold = cfg["trainer"]["unsupervised"].get("threshold", 0.95)
            with torch.no_grad():
                model_teacher.eval()
                pred_u, _ = model_teacher(image_u_weak.detach())
                pred_u = F.softmax(pred_u, dim=1)
                # obtain pseudos
                logits_u_aug, label_u_aug = torch.max(pred_u, dim=1)
                
                # obtain confidence
                entropy = -torch.sum(pred_u * torch.log(pred_u + 1e-10), dim=1)
                entropy /= np.log(cfg["net"]["num_classes"])
                confidence = 1.0 - entropy
                confidence = confidence * logits_u_aug
                confidence = confidence.mean(dim=[1,2])  # 1*C
                confidence = confidence.cpu().numpy().tolist()
                # effect stats: entropy mean (teacher on weak)
                u_entropy_mean = float(entropy.detach().mean().item())

                # confidence = logits_u_aug.ge(p_threshold).float().mean(dim=[1,2]).cpu().numpy().tolist()
                del pred_u
            model.train()
            
            # 2. apply cutmix (Ar) + log flags
            use_cutmix = cfg["trainer"]["unsupervised"].get("use_cutmix", False)
            trigger_prob = cfg["trainer"]["unsupervised"].get("use_cutmix_trigger_prob", 1.0)

            rnd = np.random.uniform(0, 1)
            ar_triggered = int(rnd < trigger_prob)
            ar_applied = int(ar_triggered and use_cutmix)

            if ar_applied:
                # estimate area ratio CHỈ để log -> chỉ tính khi do_log_now
                label_u_before = None
                if do_log_now:
                    label_u_before = label_u_aug.clone()                    

                if cfg["trainer"]["unsupervised"].get("use_cutmix_adaptive", False):                                    
                    image_u_aug, label_u_aug, logits_u_aug = cut_mix_label_adaptive(
                        image_u_aug, label_u_aug, logits_u_aug,
                        image_l, label_l, confidence
                    )
                else:
                    image_u_aug, label_u_aug, logits_u_aug = cut_mix_label_adaptive(
                        image_u_aug, label_u_aug, logits_u_aug,
                        image_l, label_l, confidence
                    )

                if label_u_before is not None:
                    ar_area_ratio_est = (label_u_aug != label_u_before).float().mean().item()
                    del label_u_before


            # effect stats: maxprob quantiles + pseudo ratio (sau cutmix nếu có)
            # CHỈ tính khi thật sự log để tránh chậm (torch.quantile khá tốn)
            if do_log_now:
                mp = logits_u_aug.detach()           # [B,H,W] max prob
                mp_flat = mp.reshape(-1)

                u_maxprob_mean = float(mp_flat.mean().item())
                q = torch.quantile(
                    mp_flat,
                    torch.tensor([0.1, 0.5, 0.9], device=mp.device)
                )
                u_maxprob_p10 = float(q[0].item())
                u_maxprob_p50 = float(q[1].item())
                u_maxprob_p90 = float(q[2].item())

                u_pseudo_ratio_mean = float((mp >= p_threshold).float().mean().item())



            # 3. forward concate labeled + unlabeld into student networks
            num_labeled = len(image_l)
            if flag_extra_weak:
                pred_all, aux_all = model(torch.cat((image_l, image_u_weak, image_u_aug), dim=0))
                del image_l, image_u_weak, image_u_aug
                pred_l= pred_all[:num_labeled]
                _, pred_u_strong = pred_all[num_labeled:].chunk(2)
                del pred_all
            else:
                pred_all, aux_all = model(torch.cat((image_l, image_u_aug), dim=0))
                del image_l, image_u_weak, image_u_aug
                pred_l= pred_all[:num_labeled]
                pred_u_strong = pred_all[num_labeled:]
                del pred_all

            # 4. supervised loss
            if "aux_loss" in cfg["net"].keys():
                aux = aux_all[:num_labeled]
                sup_loss = sup_loss_fn([pred_l, aux], label_l)
                del aux_all, aux
            else:
                sup_loss = sup_loss_fn(pred_l, label_l)

            # 5. unsupervised loss
            unsup_loss, pseduo_high_ratio = compute_unsupervised_loss_by_threshold(
                        pred_u_strong, label_u_aug.detach(),
                        logits_u_aug.detach(), thresh=p_threshold)
            unsup_loss *= cfg["trainer"]["unsupervised"].get("loss_weight", 1.0)
            del pred_l, pred_u_strong, label_u_aug, logits_u_aug

        loss = sup_loss + unsup_loss

        # update student model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update teacher model with EMA
        with torch.no_grad():
            if epoch > cfg["trainer"].get("sup_only_epoch", 0):
                ema_decay = min(
                    1
                    - 1
                    / (
                        i_iter
                        - len(loader_l) * cfg["trainer"].get("sup_only_epoch", 0)
                        + 1
                    ),
                    ema_decay_origin,
                )
            else:
                ema_decay = 0.0
            # update weight
            for param_train, param_eval in zip(model.parameters(), model_teacher.parameters()):
                param_eval.data = param_eval.data * ema_decay + param_train.data * (1 - ema_decay)
            # update bn
            for buffer_train, buffer_eval in zip(model.buffers(), model_teacher.buffers()):
                buffer_eval.data = buffer_eval.data * ema_decay + buffer_train.data * (1 - ema_decay)
                # buffer_eval.data = buffer_train.data

        # gather all loss from different gpus
        reduced_sup_loss = sup_loss.clone().detach()
        dist.all_reduce(reduced_sup_loss)
        sup_losses.update(reduced_sup_loss.item() / world_size)

        reduced_uns_loss = unsup_loss.clone().detach()
        dist.all_reduce(reduced_uns_loss)
        uns_losses.update(reduced_uns_loss.item() / world_size)

        reduced_pseudo_high_ratio = pseduo_high_ratio.clone().detach()
        dist.all_reduce(reduced_pseudo_high_ratio)
        meter_high_pseudo_ratio.update(reduced_pseudo_high_ratio.item() / world_size)

        # 12. print log information
        batch_end = time.time()
        batch_times.update(batch_end - batch_start)
        # if i_iter % 10 == 0 and rank == 0:
        if step in print_freq_lst and rank == 0:
            logger.info(
                "Epoch/Iter [{}:{:3}/{:3}].  "
                "Sup:{sup_loss.val:.3f}({sup_loss.avg:.3f})  "
                "Uns:{uns_loss.val:.3f}({uns_loss.avg:.3f})  "
                "Pseudo:{high_ratio.val:.3f}({high_ratio.avg:.3f})  "
                "Time:{batch_time.avg:.2f}  "
                "LR:{lr.val:.5f}".format(
                    cfg["trainer"]["epochs"], epoch, step,
                    sup_loss=sup_losses,
                    uns_loss=uns_losses,
                    high_ratio=meter_high_pseudo_ratio,
                    batch_time=batch_times,
                    lr=learning_rates,
                )
            )

            if tb_logger is not None:
                tb_logger.add_scalar("lr", learning_rates.avg, i_iter)
                tb_logger.add_scalar("Sup Loss", sup_losses.avg, i_iter)
                tb_logger.add_scalar("Uns Loss", uns_losses.avg, i_iter)
                tb_logger.add_scalar("High ratio", meter_high_pseudo_ratio.avg, i_iter)

        # ---------- Iter log: CSV luôn ghi, W&B là tùy chọn ----------
        if rank == 0:
            do_log = (i_iter % log_every == 0) or (step == len(loader_l) - 1)
            if do_log:
                log_dict = make_default_iter_log_dict(
                    sup_loss=sup_losses.val,
                    uns_loss=uns_losses.val,
                    pseudo_high_ratio=meter_high_pseudo_ratio.val,
                    lr=learning_rates.val,
                    epoch=epoch,
                    step=step,
                    global_iter=i_iter,
                    ar_triggered=ar_triggered,
                    ar_applied=ar_applied,
                    ar_area_ratio_est=ar_area_ratio_est,
                    u_entropy_mean=u_entropy_mean,
                    u_maxprob_mean=u_maxprob_mean,
                    u_maxprob_p10=u_maxprob_p10,
                    u_maxprob_p50=u_maxprob_p50,
                    u_maxprob_p90=u_maxprob_p90,
                    u_pseudo_ratio_mean=u_pseudo_ratio_mean,
                )
                # -------- Aa (photometric) --------
                if (k_ids is not None) and (t_vals is not None):
                    k_np = k_ids.detach().cpu().numpy()
                    t_np = t_vals.detach().cpu().numpy()

                    log_dict["aa/ops_per_image"] = float((k_np > 0).sum(axis=1).mean())
                    total_ops = max(int((k_np > 0).sum()), 1)

                    AA_DEFAULT_PCT = {
                        "identity": 0.0,
                        "autocontrast": 100.0,
                        "equalize": 100.0,
                    }

                    for kid in range(1, 12):
                        op_name = AA_OPS.get(kid, f"op{kid}")
                        m = (k_np == kid)
                        cnt = int(m.sum())

                        log_dict[f"aa/op_rate/{op_name}"] = cnt / total_ops
                        log_dict[f"aa/count/{op_name}"] = cnt

                        if cnt <= 0:
                            continue

                        tv = t_np[m]
                        tv = tv[np.isfinite(tv)]

                        if tv.size > 0:
                            sv = np.array([aa_strength(kid, float(x)) for x in tv], dtype=np.float32)
                            sv = sv[np.isfinite(sv)]

                            if sv.size > 0:
                                sv_pct = sv * 100.0
                                log_dict[f"aa/pct_mean/{op_name}"] = float(sv_pct.mean())
                                log_dict[f"aa/pct_max/{op_name}"] = float(sv_pct.max())

                            log_dict[f"aa/t_mean/{op_name}"] = float(tv.mean())
                            log_dict[f"aa/t_std/{op_name}"] = float(tv.std())

                # W&B optional
                if wandb_run is not None:
                    wandb_run.log(log_dict, step=int(i_iter))

                # CSV always on if path is provided
                if train_iter_csv is not None:
                    df_row = pd.DataFrame([log_dict], columns=ITER_LOG_COLUMNS)
                    if os.path.exists(train_iter_csv):
                        df_row.to_csv(train_iter_csv, mode="a", header=False, index=False)
                    else:
                        df_row.to_csv(train_iter_csv, index=False)

    
    return sup_losses.avg, uns_losses.avg


def validate(
    model,
    data_loader,
    epoch,
    logger,
    cfg,
    sup_loss_fn,
):
    model.eval()
    data_loader.sampler.set_epoch(epoch)

    num_classes, ignore_label = (
        cfg["net"]["num_classes"],
        cfg["dataset"]["ignore_label"],
    )
    rank, world_size = dist.get_rank(), dist.get_world_size()

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    loss_meter = AverageMeter()

    for step, batch in enumerate(data_loader):
        _, images, labels = batch
        images = images.cuda()
        labels = labels.long().cuda()

        with torch.no_grad():
            pred, aux = model(images)

            if "aux_loss" in cfg["net"].keys():
                val_loss = sup_loss_fn([pred, aux], labels)
            else:
                val_loss = sup_loss_fn(pred, labels)

        reduced_val_loss = val_loss.detach().clone()
        dist.all_reduce(reduced_val_loss)
        loss_meter.update(reduced_val_loss.item() / world_size)

        output = pred.data.max(1)[1].cpu().numpy()
        target_origin = labels.cpu().numpy()

        intersection, union, target = intersectionAndUnion(
            output, target_origin, num_classes, ignore_label
        )

        reduced_intersection = torch.from_numpy(intersection).cuda()
        reduced_union = torch.from_numpy(union).cuda()
        reduced_target = torch.from_numpy(target).cuda()

        dist.all_reduce(reduced_intersection)
        dist.all_reduce(reduced_union)
        dist.all_reduce(reduced_target)

        intersection_meter.update(reduced_intersection.cpu().numpy())
        union_meter.update(reduced_union.cpu().numpy())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)

    if rank == 0:
        for i, iou in enumerate(iou_class):
            logger.info(" [Test] -  class [{}] IoU {:.2f}".format(i, iou * 100))

    return {
        "loss": float(loss_meter.avg),
        "miou": float(mIoU),
        "iou_class": iou_class.astype(np.float64),
    }

def validate_citys(
    model,
    data_loader,
    epoch,
    logger,
    cfg,
    sup_loss_fn,
):
    model.eval()
    data_loader.sampler.set_epoch(epoch)
    rank, world_size = dist.get_rank(), dist.get_world_size()

    num_classes = cfg["net"]["num_classes"]
    ignore_label = cfg["dataset"]["ignore_label"]
    if cfg["dataset"]["val"].get("crop", False):
        crop_size, _ = cfg["dataset"]["val"]["crop"].get("size", [800, 800])
    else:
        crop_size = 800

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    loss_meter = AverageMeter()

    for step, batch in enumerate(data_loader):
        _, images, labels = batch
        images = images.cuda()
        labels = labels.long()
        batch_size, h, w = labels.shape

        with torch.no_grad():
            final = torch.zeros(batch_size, num_classes, h, w).cuda()
            row = 0
            while row < h:
                col = 0
                while col < w:
                    pred, _ = model(
                        images[:, :, row:min(h, row + crop_size), col:min(w, col + crop_size)]
                    )
                    final[:, :, row:min(h, row + crop_size), col:min(w, col + crop_size)] += pred.softmax(dim=1)
                    col += int(crop_size * 2 / 3)
                row += int(crop_size * 2 / 3)

            labels_cuda = labels.cuda()
            if "aux_loss" in cfg["net"].keys():
                val_loss = sup_loss_fn([final, final], labels_cuda)
            else:
                val_loss = sup_loss_fn(final, labels_cuda)

            reduced_val_loss = val_loss.detach().clone()
            dist.all_reduce(reduced_val_loss)
            loss_meter.update(reduced_val_loss.item() / world_size)

            output = final.argmax(dim=1).cpu().numpy()
            target_origin = labels.numpy()

        intersection, union, target = intersectionAndUnion(
            output, target_origin, num_classes, ignore_label
        )

        reduced_intersection = torch.from_numpy(intersection).cuda()
        reduced_union = torch.from_numpy(union).cuda()
        reduced_target = torch.from_numpy(target).cuda()

        dist.all_reduce(reduced_intersection)
        dist.all_reduce(reduced_union)
        dist.all_reduce(reduced_target)

        intersection_meter.update(reduced_intersection.cpu().numpy())
        union_meter.update(reduced_union.cpu().numpy())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)

    if rank == 0:
        for i, iou in enumerate(iou_class):
            logger.info(" [Test] -  class [{}] IoU {:.2f}".format(i, iou * 100))

    return {
        "loss": float(loss_meter.avg),
        "miou": float(mIoU),
        "iou_class": iou_class.astype(np.float64),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semi-Supervised Semantic Segmentation")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", default=None, type=int)
    args = parser.parse_args()
    main(args)
