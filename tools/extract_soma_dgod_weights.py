import torch
import argparse
import copy


def load_checkpoint_dicts(ckpt_path):
    """Load state_dict from checkpoint (ignore ema_state_dict)."""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    return ckpt.get('state_dict', {})


def filter_keys(sd, target_modules):
    new_sd = {}
    for k, v in sd.items():
        sk = k.replace('module.', '', 1) if k.startswith('module.') else k
        # always keep the mask token
        if sk == 'backbone.model.mask_token':
            new_sd[k] = v
            continue
        # keep all keys that do NOT contain 'backbone'
        if 'backbone' not in sk:
            new_sd[k] = v
            continue
        # keep LoRA adapters even if they live under backbone
        if 'lora' in sk.lower():
            new_sd[k] = v
            continue
        # keep specified bias terms under backbone
        if any(tm in sk and sk.endswith('bias') for tm in target_modules):
            new_sd[k] = v
            continue
    return new_sd


def main():
    parser = argparse.ArgumentParser(
        description='Extract SoMA checkpoint from final checkpoint (drop ema, keep non-backbone keys)')
    parser.add_argument('--final_ckpt', type=str, required=True,
                        help='Path to the final checkpoint file')
    parser.add_argument('--soma_ckpt', type=str, required=True,
                        help='Path to save the SoMA checkpoint')
    parser.add_argument('--target_modules', nargs='+',
                        default=['q','k','v','proj','fc1','fc2'],
                        help='Keywords for target-module bias terms')
    args = parser.parse_args()

    # load just the state_dict
    state_sd = load_checkpoint_dicts(args.final_ckpt)

    # filter out everything under backbone except mask_token, lora, and bias terms;
    # also keep any key that doesn’t mention 'backbone'
    filtered_state = filter_keys(state_sd, args.target_modules)

    # build new checkpoint with only state_dict
    new_ckpt = {'state_dict': filtered_state}

    torch.save(new_ckpt, args.soma_ckpt)
    print(f'SoMA checkpoint saved to {args.soma_ckpt}')


if __name__ == '__main__':
    main()