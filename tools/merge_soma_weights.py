import torch
import argparse
import copy


def load_checkpoint(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'ema_state_dict' in ckpt:
        sd = ckpt['ema_state_dict']
        is_ema = True
    elif 'state_dict' in ckpt:
        sd = ckpt['state_dict']
        is_ema = False
    else:
        sd = ckpt
        is_ema = False
    return ckpt, sd, is_ema


def add_prefix_to_keys(state_dict, is_ema):
    new_sd = {}
    for k, v in state_dict.items():
        prefix = 'module.backbone.model.' if is_ema else 'backbone.model.'
        new_key = prefix + k if not k.startswith(prefix) else k
        new_sd[new_key] = v
    return new_sd


def main():
    parser = argparse.ArgumentParser(description='Merge backbone and SoMA checkpoints into final checkpoint')
    parser.add_argument('--backbone_ckpt', type=str, required=True, help='Path to backbone checkpoint file')
    parser.add_argument('--soma_ckpt', type=str, required=True, help='Path to the SoMA checkpoint file')
    parser.add_argument('--merged_ckpt', type=str, required=True, help='Path to save the merged final checkpoint')
    args = parser.parse_args()

    _, backbone_sd, is_ema_backbone = load_checkpoint(args.backbone_ckpt)
    soma_ckpt, _, is_ema_soma = load_checkpoint(args.soma_ckpt)

    # # Decide format: use ema if either is ema
    # use_ema = is_ema_backbone or is_ema_soma

    # Prefix keys from backbone
    if is_ema_soma:
        backbone_prefixed_state = add_prefix_to_keys(backbone_sd, is_ema=False)
        backbone_prefixed_ema   = add_prefix_to_keys(backbone_sd, is_ema=True)
    else:    
        backbone_prefixed = add_prefix_to_keys(backbone_sd, is_ema=is_ema_soma)

    # Merge soma weights (overwrite)
    soma_sd = soma_ckpt.get('state_dict', {})
    soma_sd_ema   = soma_ckpt.get('ema_state_dict', None)
    
    if is_ema_soma:
        merged_sd = copy.deepcopy(backbone_prefixed_state)
    else:
        merged_sd = copy.deepcopy(backbone_prefixed)
    merged_sd.update(soma_sd)  # overwrite backbone with task-specific updates

    # Store both state_dict and ema_state_dict (safe for MMEngine)
    merged_ckpt = {
        'state_dict': merged_sd
    }
    if is_ema_soma:
        merged_sd_ema = copy.deepcopy(backbone_prefixed_ema)
        merged_sd_ema.update(soma_sd_ema)
        merged_ckpt['ema_state_dict'] = merged_sd_ema

    torch.save(merged_ckpt, args.merged_ckpt)
    print(f'Merged checkpoint saved to {args.merged_ckpt}')


if __name__ == '__main__':
    main()