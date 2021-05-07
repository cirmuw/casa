import yaml
import argparse
import active_dynamicmemory.utils as autils
from py_jotools import slurm

def train_config(configfile, remote=False):
    with open(configfile) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    if remote:
        with open('training_configs/slurm_config.yml') as s:
            sparams = yaml.load(s, Loader=yaml.FullLoader)
        print('scheduling job to CIR cluster...')
        slurm.srun(autils.trained_model, [params['trainparams'], params['settings']], params=sparams, remote=True)
    else:
        model, logs, df_mem, exp_name = autils.trained_model(params['trainparams'], params['settings'])

    print('successfully trained model', exp_name)

if __name__ == "__main__":
    # execute only if run as a script
    parser = argparse.ArgumentParser(description='Run a training with the dynamic memory framework.')
    parser.add_argument('--config', type=str, help='path to a config file (yml) to run')
    parser.add_argument('-s',
                        '--slurm',
                        action='store_true',
                        help='run on CIR slurm cluster')

    args = parser.parse_args()

    train_config(args.config, remote=args.slurm)

