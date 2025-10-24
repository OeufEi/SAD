import torch
import numpy as np
import copy
import utils
import wandb
import os
import torchvision
import gc
from tqdm import tqdm

from utils import get_network, config, evaluate_synset

def build_dataset(ds, class_map, num_classes):
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]
    print("BUILDING DATASET")
    for i in tqdm(range(len(ds))):
        sample = ds[i]
        images_all.append(torch.unsqueeze(sample[0], dim=0))
        labels_all.append(class_map[torch.tensor(sample[1]).item()])
    for i, lab in tqdm(enumerate(labels_all)):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to("cpu")
    labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")

    return images_all, labels_all, indices_class


def prepare_latents_lfm(num_classes=10, im_size=(32, 32), args=None):
    with torch.no_grad():
        ''' initialize the synthetic data '''
        label_syn = torch.tensor([i*np.ones(args.ipc, dtype=np.int64) for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
        # ------------------------------
        # LFM space: optimize VAE latents
        # ------------------------------
        # Your LFM sampler/decoders expect VAE latents of shape [B, 4, H/8, W/8]
        # (see test_flow_latent*.py where x is sampled as randn(B, 4, image_size//8, image_size//8))
        # H/8, W/8 comes from the VAE downsample factor f=8.
        f = args.f  # default 8, consistent with your flow code
        h, w = im_size
        h_lat, w_lat = h // f, w // f

        # initialize synthetic VAE latents (trainable)
        latents = torch.randn(
            num_classes * args.ipc, 4, h_lat, w_lat,
            device=args.device, dtype=torch.float32, requires_grad=False
        )
        f_latents = None  # no "feature" latents for LFM branch

        print('initialize synthetic data from random noise')
        latents = latents.detach().to(args.device).requires_grad_(True)

        return latents, f_latents, label_syn


def get_optimizer_img_lfm(latents=None, lfm=None, vae=None, args=None):
    param_groups = []
    param_groups.append({'params': [latents], 'lr': args.lr_img})
    optimizer_img = torch.optim.SGD(param_groups, momentum=0.5)
    if args.learn_lfm and (lfm is not None):
            for p in lfm.parameters():
                p.requires_grad_(True)
            optimizer_img.add_param_group({'params': lfm.parameters(), 'lr': args.lr_g})
    if args.learn_vae and (vae is not None):
        for p in vae.parameters():
            p.requires_grad_(True)
        optimizer_img.add_param_group({'params': vae.parameters(), 'lr': args.lr_vae})
    
    optimizer_img.zero_grad()
    return optimizer_img

def get_eval_lrs(args):
    eval_pool_dict = {
        args.model: 0.001,
        "ResNet18": 0.001,
        "VGG11": 0.0001,
        "AlexNet": 0.001,
        "ViT": 0.001,

        "AlexNetCIFAR": 0.001,
        "ResNet18CIFAR": 0.001,
        "VGG11CIFAR": 0.0001,
        "ViTCIFAR": 0.001,
    }

    return eval_pool_dict


def lfm_eval_loop(
    latents=None, label_syn=None,
    best_acc={}, best_std={}, testloader=None, model_eval_pool=[],
    it=0, channel=3, num_classes=10, im_size=(32, 32), args=None,
    lfm=None, vae=None   # <- NEW for LFM evaluation
):
    curr_acc_dict, max_acc_dict = {}, {}
    curr_std_dict, max_std_dict = {}, {}
    eval_pool_dict = get_eval_lrs(args)
    save_this_it = False

    for model_eval in model_eval_pool:
        if model_eval != args.model and args.wait_eval and it != args.Iteration:
            continue
        print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'
              % (args.model, model_eval, it))

        accs_test, accs_train = [], []

        for it_eval in range(args.num_eval):
            net_eval = get_network(
                model_eval, channel, num_classes, im_size,
                width=args.width, depth=args.depth, dist=False
            ).to(args.device)

            # make a safe copy (no accidental in-place edits)
            image_syn_eval = copy.deepcopy(latents.detach())
            label_syn_eval = copy.deepcopy(label_syn.detach())

            if lfm is None or vae is None:
                raise ValueError("eval_loop: lfm/vae must be provided when args.space == 'lfm'")
            with torch.no_grad():
                image_syn_eval = torch.cat([
                    lfm_latent_to_im(lfm, vae, lat_split, args, y=lab_split).detach()
                    for lat_split, lab_split in zip(
                        torch.split(image_syn_eval, args.sg_batch),
                        torch.split(label_syn_eval, args.sg_batch)
                    )
                ])
            # ---------------------------------------------------------------

            args.lr_net = eval_pool_dict[model_eval]
            _, acc_train, acc_test = evaluate_synset(
                it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args=args, aug=True
            )
            del _
            del net_eval
            accs_test.append(acc_test)
            accs_train.append(acc_train)

        print(accs_test)
        accs_test = np.array(accs_test)
        accs_train = np.array(accs_train)
        acc_test_mean = np.mean(np.max(accs_test, axis=1))
        acc_test_std  = np.std(np.max(accs_test, axis=1))

        key = f"{model_eval}"
        if acc_test_mean > best_acc[key]:
            best_acc[key] = acc_test_mean
            best_std[key] = acc_test_std
            save_this_it = True

        curr_acc_dict[key] = acc_test_mean
        curr_std_dict[key] = acc_test_std
        max_acc_dict[key]  = best_acc[key]
        max_std_dict[key]  = best_std[key]

        print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'
              % (len(accs_test[:, -1]), model_eval, acc_test_mean, np.std(np.max(accs_test, axis=1))))
        wandb.log({'Accuracy/{}'.format(model_eval): acc_test_mean}, step=it)
        wandb.log({'Max_Accuracy/{}'.format(model_eval): best_acc[key]}, step=it)
        wandb.log({'Std/{}'.format(model_eval): acc_test_std}, step=it)
        wandb.log({'Max_Std/{}'.format(model_eval): best_std[key]}, step=it)

    wandb.log({
        'Accuracy/Avg_All': np.mean(np.array(list(curr_acc_dict.values()))),
        'Std/Avg_All':      np.mean(np.array(list(curr_std_dict.values()))),
        'Max_Accuracy/Avg_All': np.mean(np.array(list(max_acc_dict.values()))),
        'Max_Std/Avg_All':      np.mean(np.array(list(max_std_dict.values()))),
    }, step=it)

    curr_acc_dict.pop(f"{args.model}", None)
    curr_std_dict.pop(f"{args.model}", None)
    max_acc_dict.pop(f"{args.model}", None)
    max_std_dict.pop(f"{args.model}", None)

    wandb.log({
        'Accuracy/Avg_Cross': np.mean(np.array(list(curr_acc_dict.values()))) if curr_acc_dict else 0.0,
        'Std/Avg_Cross':      np.mean(np.array(list(curr_std_dict.values()))) if curr_std_dict else 0.0,
        'Max_Accuracy/Avg_Cross': np.mean(np.array(list(max_acc_dict.values()))) if max_acc_dict else 0.0,
        'Max_Std/Avg_Cross':      np.mean(np.array(list(max_std_dict.values()))) if max_std_dict else 0.0,
    }, step=it)

    return save_this_it

def load_lfm(res, args=None):
    import sys
    import os
    p = os.path.join("LFM")
    if p not in sys.path:
        sys.path.append(p)
    from LFM.models import create_network
    from diffusers.models import AutoencoderKL

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args is None:
        raise ValueError("load_lfm requires args with dataset and ckpt info")

    if "imagenet" in args.dataset.lower():
        pass
        # implement in the future
        #flow_ckpt = getattr(args, "lfm_ckpt", f"../checkpoints/lfm_imagenet{res}.pt")
        #vae_ckpt = getattr(args, "pretrained_autoencoder_ckpt", "../checkpoints/vae_imagenet")
    elif args.dataset.upper() == "CIFAR10":
        flow_ckpt = getattr(args, "lfm_ckpt", f"../LFM/saved_info/latent_flow/imagenet/imnet_f8_adm/model_1125.pth")
        vae_ckpt = getattr(args, "pretrained_autoencoder_ckpt", "../checkpoints/VAE_cifar10.pt")
    #elif args.dataset.upper() == "CIFAR100":
        #flow_ckpt = getattr(args, "lfm_ckpt", f"../checkpoints/lfm_cifar100_{res}.pt")
        #vae_ckpt = getattr(args, "pretrained_autoencoder_ckpt", "../checkpoints/vae_cifar100")
    else:
        # default, but unimplemented
        flow_ckpt = getattr(args, "lfm_ckpt", f"../checkpoints/lfm_generic_{res}.pt")
        vae_ckpt = getattr(args, "pretrained_autoencoder_ckpt", "../checkpoints/vae_generic")

    lfm = create_network(args).to(device)
    if os.path.exists(flow_ckpt):
        ckpt = torch.load(flow_ckpt, map_location=device)
        lfm.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    lfm.eval()

    #vae = AutoencoderKL.from_pretrained(vae_ckpt).to(device)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()

    latent_shape = (4, res // 8, res // 8)

    print(f"[load_lfm] Loaded LFM + VAE for dataset={args.dataset} at res={res}")

    return lfm, vae #, latent_shape

def sample_from_model(model, x_0, model_kwargs, args):
    """
    Integrate the latent flow ODE from t=1 -> 0.

    Args:
        model:         flow model; must implement forward(t, x, **model_kwargs)
                       and optionally forward_with_cfg(t, x, **model_kwargs) if using CFG.
        x_0:           initial latent tensor (B, C=4, H/8, W/8), requires_grad=True for training.
        model_kwargs:  dict; may include 'y' (labels), 'cfg_scale' (float), etc.
        args:          namespace with ODE/sampler settings.

    Returns:
        x_traj: tensor of shape (2, B, C, H/8, W/8) with states at t=[1.0, 0.0].
                The final latent is x_traj[-1].
                If args.compute_nfe is True, returns (x_traj, nfe).
    """
    # Infer device/dtype from x_0
    device = x_0.device
    dtype  = x_0.dtype

    # Choose solver options
    if getattr(args, "method", None) in ADAPTIVE_SOLVER:
        options = {"dtype": torch.float64}  # dopri5, bdf benefit from float64 internal state
    else:
        # fixed-step solvers (e.g., euler, midpoint, rk4) use explicit step_size; optional stochastic perturb
        options = {
            "step_size": getattr(args, "step_size", 0.1),
            "perturb":   getattr(args, "perturb", False),
        }

    # Optionally wrap model to count NFEs
    count_nfe = getattr(args, "compute_nfe", False)
    wrapped_model = model
    if count_nfe:
        wrapped_model = NFECount(model).to(device)

    # Build the time tensor on the correct device/dtype
    # Keep it float32 to match typical ODE solvers; model can internally cast as needed.
    t = torch.tensor([1.0, 0.0], device=device, dtype=torch.float32)

    # Determine CFG scale (args overrides in model_kwargs only if not provided)
    cfg_scale = model_kwargs.get("cfg_scale", getattr(args, "cfg_scale", 1.0))

    # ODE RHS
    def denoiser(t_scalar, x_state):
        if cfg_scale is not None and cfg_scale > 1.0 and hasattr(wrapped_model, "forward_with_cfg"):
            return wrapped_model.forward_with_cfg(t_scalar, x_state, **{**model_kwargs, "cfg_scale": cfg_scale})
        else:
            return wrapped_model(t_scalar, x_state, **model_kwargs)

    # Run adjoint ODE solve (keeps graph for grads w.r.t. x_0)
    x_traj = odeint(
        denoiser,
        x_0.to(device=device, dtype=dtype),
        t,
        method=getattr(args, "method", "dopri5"),
        atol=getattr(args, "atol", 1e-5),
        rtol=getattr(args, "rtol", 1e-5),
        adjoint_method=getattr(args, "method", "dopri5"),
        adjoint_atol=getattr(args, "atol", 1e-5),
        adjoint_rtol=getattr(args, "rtol", 1e-5),
        options=options,
        adjoint_params=wrapped_model.parameters(),  # required so grads can flow through the adjoint
    )

    if count_nfe:
        return x_traj, wrapped_model.nfe
    return x_traj



'''def lfm_latent_to_im(lfm, vae, latents, args, y=None):
    import sys
    import os
    p = os.path.join("LFM")
    if p not in sys.path:
        sys.path.append(p)
    from LFM.test_flow_latent import sample_from_model
    """
    Turn initial VAE latents into normalized images using the Latent Flow Matching model.

    lfm:  the flow-matching model (frozen; .eval())
    vae:  AutoencoderKL (frozen; .eval())
    latents: torch.Tensor, shape [B, 4, H/8, W/8], requires_grad=True
    y:     optional LongTensor of class ids, shape [B], or None for unconditional

    Returns:
        im: normalized images ready for the student network, shape [B, C, H, W]
    """
    # ---- Build model kwargs (CFG & labels), following your test files ----
    # If you use classifier-free guidance:
    #   - duplicate x and y, append a null y for half of the batch
    #   - pass cfg_scale in model_kwargs
    if getattr(args, "cfg_scale", 1.0) > 1.0 and (y is not None):
        # Duplicate initial latents and labels
        x_0 = torch.cat([latents, latents], dim=0)
        if "DiT" in getattr(args, "model_type", ""):
            # DiT uses an extra "null" class index = num_classes
            y_null = torch.tensor([args.num_classes] * y.shape[0], device=y.device, dtype=y.dtype)
        else:
            # Others use zeros as null
            y_null = torch.zeros_like(y)
        y_all = torch.cat([y, y_null], dim=0)
        model_kwargs = dict(y=y_all, cfg_scale=args.cfg_scale)
    else:
        x_0 = latents
        model_kwargs = {} if y is None else dict(y=y)

    # ---- Integrate ODE from t=1 -> 0 to get a VAE latent, then decode ----
    # sample_from_model returns [x_t at t=1, x_t at t=0]; we need the final latent at t=0
    fake_latent = sample_from_model(lfm, x_0, model_kwargs, args)[-1]  # differentiable ODE call :contentReference[oaicite:0]{index=0}

    # If CFG was used, throw away the null half (keep the conditioned half)
    if getattr(args, "cfg_scale", 1.0) > 1.0 and (y is not None):
        fake_latent, _ = fake_latent.chunk(2, dim=0)  # keep first half only :contentReference[oaicite:1]{index=1}

    # Decode with VAE; your flow code divides by args.scale_factor before decode
    im = vae.decode(fake_latent / args.scale_factor).sample               # :contentReference[oaicite:2]{index=2}

    # ---- Match StyleGAN path’s normalization ----
    # Your pipelines treat generator output as in [-1,1] then:
    #   (im + 1)/2 to [0,1] and per-dataset normalize by mean/std
    # (Exactly what test_flow does before saving) :contentReference[oaicite:3]{index=3}
    im = (im + 1.0) / 2.0

    # Dataset-specific normalization (same as latent_to_im)
    if "imagenet" in args.dataset:
        mean, std = config.mean, config.std
    elif args.dataset in ["CIFAR10", "CIFAR100"]:
        mean, std = config.mean, config.std
        # (Your original code optionally uses mean_1/std_1 when distributed; keep parity if needed.)

    im = (im - mean) / std
    return im'''

def lfm_latent_to_im(lfm, vae, latents, args, y=None):
    """
    Turn *initial* VAE latents (B, 4, H/8, W/8) into dataset-normalized images using LFM + VAE.
    - lfm: flow-matching model (frozen, .eval())
    - vae: AutoencoderKL (frozen, .eval())
    - latents: torch.Tensor, shape [B, 4, H/8, W/8], requires_grad=True in training loop
    - y: optional LongTensor of shape [B] with class ids if the LFM is class-conditional
    Returns
    - im: tensor of shape [B, 3, H, W], normalized by dataset mean/std (same as latent_to_im)
    """
    # ----- Build model_kwargs (CFG + labels) like in your test_flow code -----
    if (y is not None) and (getattr(args, "cfg_scale", 1.0) > 1.0):
        # duplicate for classifier-free guidance
        x_0 = torch.cat([latents, latents], dim=0)
        y_null = (torch.full_like(y, fill_value=args.num_classes)  # null idx for DiT
                  if "DiT" in getattr(args, "model_type", "")
                  else torch.zeros_like(y))
        y_all = torch.cat([y, y_null], dim=0)
        model_kwargs = dict(y=y_all, cfg_scale=args.cfg_scale)
    else:
        x_0 = latents
        model_kwargs = {} if y is None else dict(y=y)

    # ----- Integrate ODE from t=1 -> 0 and take the final latent -----
    # sample_from_model returns the whole trajectory; take the last item
    z_Tto0 = sample_from_model(lfm, x_0, model_kwargs, args)        # ODE solve (differentiable) :contentReference[oaicite:0]{index=0}
    z_0    = z_Tto0[-1]

    # If CFG was used, drop the null half (keep conditioned half)
    if (y is not None) and (getattr(args, "cfg_scale", 1.0) > 1.0):
        z_0, _ = z_0.chunk(2, dim=0)                                 # :contentReference[oaicite:1]{index=1}

    # ----- Decode with the VAE (divide by scale_factor first) -----
    im = vae.decode(z_0 / args.scale_factor).sample                  # :contentReference[oaicite:2]{index=2}

    # ----- Match latent_to_im normalization: [-1,1] -> [0,1] -> (x-mean)/std -----
    im = (im + 1.0) / 2.0
    if "imagenet" in args.dataset:
        mean, std = config.mean, config.std
    elif args.dataset in ["CIFAR10", "CIFAR100"]:
        mean, std = config.mean, config.std
    else:
        # Fallback: assume mean/std provided in args or config
        mean, std = config.mean, config.std
    im = (im - mean) / std

    return im




def lfm_image_logging(latents=None, label_syn=None, it=None, save_this_it=None, lfm=None, vae=None, args=None):
    with torch.no_grad():
        image_syn = latents.cuda()

        # LFM path: render via flow + VAE decode
        # Batch to keep memory similar to the wp code path
        if lfm is None or vae is None:
            raise ValueError("image_logging: lfm/vae must be provided for args.space == 'lfm'")

        # If your flow is class-conditional, pass per-batch labels; else y=None works too.
        if label_syn is None:
            # allow unconditional case
            image_syn = torch.cat([
                lfm_latent_to_im(lfm, vae, lat_split, args, y=None).detach()
                for lat_split in torch.split(image_syn, args.sg_batch)
            ])
        else:
            image_syn = torch.cat([
                lfm_latent_to_im(lfm, vae, lat_split, args, y=lab_split).detach()
                for lat_split, lab_split in zip(
                    torch.split(image_syn, args.sg_batch),
                    torch.split(label_syn, args.sg_batch)
                )
            ])
        
        save_dir = os.path.join(args.logdir, args.dataset, wandb.run.name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(image_syn.cpu(), os.path.join(save_dir, "images_{0:05d}.pt".format(it)))
        torch.save(label_syn.cpu(), os.path.join(save_dir, "labels_{0:05d}.pt".format(it)))

        if save_this_it:
            torch.save(image_syn.cpu(), os.path.join(save_dir, "images_best.pt".format(it)))
            torch.save(label_syn.cpu(), os.path.join(save_dir, "labels_best.pt".format(it)))

        wandb.log({"Latent_Codes": wandb.Histogram(torch.nan_to_num(latents.detach().cpu()))}, step=it)

        if args.ipc < 50 or args.force_save:

            upsampled = image_syn
            if "imagenet" not in args.dataset:
                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
            grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
            wandb.log({"Synthetic_Images": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)
            wandb.log({'Synthetic_Pixels': wandb.Histogram(torch.nan_to_num(image_syn.detach().cpu()))}, step=it)

            for clip_val in []:
                upsampled = torch.clip(image_syn, min=-clip_val, max=clip_val)
                if "imagenet" not in args.dataset:
                    upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                    upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                wandb.log({"Clipped_Synthetic_Images/raw_{}".format(clip_val): wandb.Image(
                    torch.nan_to_num(grid.detach().cpu()))}, step=it)

            for clip_val in [2.5]:
                std = torch.std(image_syn)
                mean = torch.mean(image_syn)
                upsampled = torch.clip(image_syn, min=mean - clip_val * std, max=mean + clip_val * std)
                if "imagenet" not in args.dataset:
                    upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                    upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                wandb.log({"Clipped_Synthetic_Images/std_{}".format(clip_val): wandb.Image(
                    torch.nan_to_num(grid.detach().cpu()))}, step=it)

    del upsampled, grid
