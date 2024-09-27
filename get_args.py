import argparse



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--state', type=str, default='eval')
    parser.add_argument('--device', type=int, nargs='+', default=[0], help='List of GPU device IDs')
    parser.add_argument('--pre_ckpt', type=str, default='./Checkpoints/lle/latest_ckpt.pth')
    parser.add_argument('--image_size', type=int, default=384)

    # Diffusion
    parser.add_argument('--beta_schedule', type=str, default='linear')
    parser.add_argument('--beta_start', type=float, default=0.0001)
    parser.add_argument('--beta_end', type=float, default=0.02)
    parser.add_argument('--T', type=int, default=1000)
    parser.add_argument('--ema', type=bool, default=True)
    parser.add_argument('--ema_rate', type=float, default=0.999)

    # model
    parser.add_argument('--model_var_type', type=str, default='fixedsmall')
    parser.add_argument('--in_ch', type=int, default=6)
    parser.add_argument('--fea_in_ch', type=int, default=1)
    parser.add_argument('--un_in_ch', type=int, default=3)
    parser.add_argument('--out_ch', type=int, default=3)
    parser.add_argument('--ch', type=int, default=64)
    parser.add_argument('--num_res_blocks', type=int, default=2)
    parser.add_argument('--type', type=str, default='simple')
    parser.add_argument('--ch_mult', type=list, default=[1, 2, 4, 8])
    parser.add_argument('--attn', type=list, default=[16, ])
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--resamp_with_conv', type=bool, default=True)

    # sample
    parser.add_argument('--decom_path', type=str, default='./Checkpoints/ready/decom.pth')
    parser.add_argument('--sample_type', type=str, default='generalized')
    parser.add_argument('--skip_type', type=str, default='uniform')
    parser.add_argument('--timesteps', type=int, default=10)
    parser.add_argument('--test_folder', type=str, default='C:/Users\DELL\Desktop/fix_cl_wave/test')
    parser.add_argument('--sampled_dir', type=str, default='./SampledImg/')

    args = parser.parse_args()

    return args