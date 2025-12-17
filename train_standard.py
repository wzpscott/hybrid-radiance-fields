from argparse import ArgumentParser
import os

m360_outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]
m360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]
tnt_scenes = ["truck", "train"]
db_scenes = ["drjohnson", "playroom"]
all_scenes = m360_outdoor_scenes + m360_indoor_scenes + db_scenes + tnt_scenes

def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--scene", "-s", type=str)
    parser.add_argument("--exp_name", "-n", type=str, default="test")
    parser.add_argument("--train_device", "-d", type=int, default=0)
    parser.add_argument("--extra_args", "-e", type=str, default="")
    return parser

def train_one_scene(scene, exp_name, train_device, extra_args):
    if scene in m360_outdoor_scenes:
        dataset, resolution = 'm360', 4
    elif scene in m360_indoor_scenes:
        dataset, resolution = 'm360', 2
    elif scene in tnt_scenes:
        dataset, resolution = 'tnt', 1
    elif scene in db_scenes:
        dataset, resolution = 'db', 1
    else:
        raise ValueError(f"Scene {scene} not in {all_scenes}")
    
    print(f"Training {scene} from {dataset} with resolution {resolution}")

    # cmd = f"CUDA_VISIBLE_DEVICES={train_device} python train.py -s ./data/{dataset}/{scene} -m ./logs/{dataset}/{scene}/{exp_name} -r {resolution} --test_iterations 7000 30000 --eval --disable_viewer"
    if resolution != 1:
        image_dir = f"images_{resolution}"
    else:
        image_dir = "images"

    cmd = f"CUDA_VISIBLE_DEVICES={train_device} python train.py -s ./data/{dataset}/{scene} -m ./logs/{dataset}/{scene}/{exp_name} -i {image_dir} --test_iterations 7000 30000 --eval --disable_viewer"


    if extra_args:
        cmd += " " + extra_args
        
    os.system(cmd)

def train_m360_indoor(exp_name, train_device, extra_args):
    for scene in m360_indoor_scenes:
        train_one_scene(scene, exp_name, train_device, extra_args)

def train_m360_outdoor(exp_name, train_device, extra_args):
    for scene in m360_outdoor_scenes:
        train_one_scene(scene, exp_name, train_device, extra_args)

def train_tnt(exp_name, train_device, extra_args):
    for scene in tnt_scenes:
        train_one_scene(scene, exp_name, train_device, extra_args)

def train_db(exp_name, train_device, extra_args):
    for scene in db_scenes:
        train_one_scene(scene, exp_name, train_device, extra_args)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    if args.scene == "all":
        train_m360_indoor(args.exp_name, args.train_device, args.extra_args)
        train_m360_outdoor(args.exp_name, args.train_device, args.extra_args)
        train_tnt(args.exp_name, args.train_device, args.extra_args)
        train_db(args.exp_name, args.train_device, args.extra_args)
    elif args.scene == "m360-in":
        train_m360_indoor(args.exp_name, args.train_device, args.extra_args)
    elif args.scene == "m360-out":
        train_m360_outdoor(args.exp_name, args.train_device, args.extra_args)
    elif args.scene == "m360":
        train_m360_indoor(args.exp_name, args.train_device, args.extra_args)
        train_m360_outdoor(args.exp_name, args.train_device, args.extra_args)
    elif args.scene == "tnt":
        train_tnt(args.exp_name, args.train_device, args.extra_args)
    elif args.scene == "db":
        train_db(args.exp_name, args.train_device, args.extra_args)
    elif args.scene in all_scenes:
        train_one_scene(args.scene, args.exp_name, args.train_device, args.extra_args)
    else:
        raise ValueError(f"Scene {args.scene} not in {all_scenes}")