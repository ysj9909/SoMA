import torch
import argparse
import copy


def load_checkpoint_dicts(ckpt_path):
    """Load both state_dict and ema_state_dict from checkpoint if available."""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_sd = ckpt.get('state_dict', {})
    ema_sd = ckpt.get('ema_state_dict', None)
    return state_sd, ema_sd


def filter_keys(sd, target_modules):
    new_sd = {}
    for k, v in sd.items():
        sk = k.replace('module.', '', 1) if k.startswith('module.') else k
        # mask_token
        if sk == 'backbone.model.mask_token':
            new_sd[k] = v
            continue
        # decode_head
        if 'decode_head.' in sk:
            new_sd[k] = v
            continue
        # LoRA adapters
        if 'lora' in sk.lower():
            new_sd[k] = v
            continue
        # specified bias terms
        if any(tm in sk and sk.endswith('bias') for tm in target_modules):
            new_sd[k] = v
            continue
    return new_sd


def main():
    parser = argparse.ArgumentParser(
        description='Extract SoMA checkpoint from final checkpoint (handles both state and ema)')
    parser.add_argument('--final_ckpt', type=str, required=True,
                        help='Path to the final checkpoint file')
    parser.add_argument('--soma_ckpt', type=str, required=True,
                        help='Path to save the SoMA checkpoint')
    parser.add_argument('--target_modules', nargs='+', default=['q','k','v','proj','fc1','fc2'],
                        help='Keywords for target module biases')
    args = parser.parse_args()

    state_sd, ema_sd = load_checkpoint_dicts(args.final_ckpt)
    new_ckpt = {}
    # extract from state_dict
    filtered_state = filter_keys(state_sd, args.target_modules)
    new_ckpt['state_dict'] = filtered_state
    # extract from ema_state_dict if exists
    if ema_sd is not None:
        filtered_ema = filter_keys(ema_sd, args.target_modules)
        new_ckpt['ema_state_dict'] = filtered_ema

    torch.save(new_ckpt, args.soma_ckpt)
    print(f'SoMA checkpoint saved to {args.soma_ckpt}')


if __name__ == '__main__':
    main()
