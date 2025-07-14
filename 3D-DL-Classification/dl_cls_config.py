import argparse


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./3D-DL-Classification/weights')
    parser.add_argument('--writer_comment', type=str, default='3D_DL_CLS')
    parser.add_argument('--save_model', type=bool, default=True)

    # MODEL PARAMETER
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--class_num', type=int, default=3)
    parser.add_argument('--fold', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--log_step', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-6)

    parser.add_argument('--loss_function', type=str, default='CE')
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['SGD', 'Adam', 'AdamW'])
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step'])
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--warmup_decay', type=float, default=0.01)
    parser.add_argument('--min_lr', type=float, default=1e-8)
    
    parser.add_argument('--step', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.9)

    config = parser.parse_args()
    return config