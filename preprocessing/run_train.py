import argparse
from util import add_pipe_args, add_train_args, train
from functions import augment, create_darknet_data, create_yolo_cfg, resize_images, readImages, split, kfold
from Pipeline import Pipeline, PipeConfig


def main(args):
    config = PipeConfig(
        name=args.n,
        input_folder=args.i,
        output_folder=args.o,
        resized_img_size=600,
        final_img_size=416,
        number_of_augmentations=args.nbr_augment,
        color=args.c,
        transform=args.t,
        classes_txt=args.cls,
        yolo_cfg=args.yolo_cfg,
        max_batch_size=args.batch_size,
        folds=args.f,
        occlude=args.occl,
        include_no_label=args.incl_no_label,
        is_final=args.is_final
    )
    # Initialize Pipeline
    pipe = Pipeline(config=config)

    # 1. Red the images
    pipe.add(readImages)

    # 2. Resize images (if done once comment out)
    pipe.add(resize_images)

    # 3. Split the images in train & test images
    if pipe.config.folds == 1:
        pipe.add(split)
    else:
        pipe.add(kfold)

    # 4. Augment the images
    pipe.add(augment)

    # # 5. Create the darknet data file
    pipe.add(create_darknet_data)

    # # 6. Create the yolo config
    pipe.add(create_yolo_cfg)

    # Execute Pipeline
    pipe.execute()

    train(pipe.config, args.darknet)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Preprocessing & Training')
    add_pipe_args(parser)
    add_train_args(parser)
    args = parser.parse_args()

    main(args)
