from get_args import get_args
from DiffusionModel import Diffusion


if __name__ == '__main__':
    args = get_args()
    Diffusion = Diffusion(args)
    Diffusion.test_all()
