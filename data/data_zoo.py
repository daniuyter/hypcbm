from torchvision import datasets
import torch
import os


def get_dataset(args, preprocess=None):
    if args.dataset == "cifar10":
        trainset = datasets.CIFAR10(root=args.out_dir, train=True,
                                    download=True, transform=preprocess)
        testset = datasets.CIFAR10(root=args.out_dir, train=False,
                                    download=True, transform=preprocess)
        classes = trainset.classes
        class_to_idx = {c: i for (i,c) in enumerate(classes)}
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.num_workers)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                          shuffle=False, num_workers=args.num_workers)
    
    
    elif args.dataset == "cifar100":
        trainset = datasets.CIFAR100(root=args.out_dir, train=True,
                                    download=True, transform=preprocess)
        testset = datasets.CIFAR100(root=args.out_dir, train=False,
                                    download=True, transform=preprocess)
        classes = trainset.classes
        class_to_idx = {c: i for (i,c) in enumerate(classes)}
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.num_workers)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                          shuffle=False, num_workers=args.num_workers)


    elif args.dataset == "cub":
        from .cub import load_cub_data
        from .constants import CUB_PROCESSED_DIR, CUB_DATA_DIR
        from torchvision import transforms
        num_classes = 200
        TRAIN_PKL = os.path.join(CUB_PROCESSED_DIR, "train.pkl")
        TEST_PKL = os.path.join(CUB_PROCESSED_DIR, "test.pkl")
        normalizer = transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
        train_loader = load_cub_data([TRAIN_PKL], use_attr=False, no_img=False, 
            batch_size=args.batch_size, uncertain_label=False, image_dir=CUB_DATA_DIR, resol=224, normalizer=normalizer,
            n_classes=num_classes, resampling=True)

        test_loader = load_cub_data([TEST_PKL], use_attr=False, no_img=False, 
                batch_size=args.batch_size, uncertain_label=False, image_dir=CUB_DATA_DIR, resol=224, normalizer=normalizer,
                n_classes=num_classes, resampling=True)

        classes = open(os.path.join(CUB_DATA_DIR, "classes.txt")).readlines()
        classes = [a.split(".")[1].strip() for a in classes]
        idx_to_class = {i: classes[i] for i in range(num_classes)}
        classes = [classes[i] for i in range(num_classes)]
        print(len(classes), "num classes for cub")
        print(len(train_loader.dataset), "training set size")
        print(len(test_loader.dataset), "test set size")
        

    elif args.dataset == "ham10000":
        from .derma_data import load_ham_data
        train_loader, test_loader, idx_to_class = load_ham_data(args, preprocess)
        class_to_idx = {v:k for k,v in idx_to_class.items()}
        classes = list(class_to_idx.keys())


    elif args.dataset == "places365":
        root = getattr(args, "places365_root", 'places365')
        if not root:
            raise ValueError("places365_root must be provided when dataset=places365")
        train_split = getattr(args, "places365_train_split", "train-standard")
        val_split = getattr(args, "places365_val_split", "val")
        use_small = bool(getattr(args, "places365_small", True))

        trainset = datasets.Places365(
            root=root,
            split=train_split,
            small=use_small,
            download=True,
            transform=preprocess,
        )
        testset = datasets.Places365(
            root=root,
            split=val_split,
            small=use_small,
            download=True,
            transform=preprocess,
        )
        classes = trainset.classes
        class_to_idx = trainset.class_to_idx
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )


    elif args.dataset == "imagenet":
        if not getattr(args, "imagenet_root", None):
            raise ValueError("imagenet_root must be provided when dataset=imagenet")

        train_dir = os.path.join(args.imagenet_root, "train")
        val_dir = os.path.join(args.imagenet_root, "val")
        if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
            raise ValueError(
                f"Expected ImageNet train/val folders under {args.imagenet_root}"
            )

        trainset = datasets.ImageFolder(train_dir, transform=preprocess)
        testset = datasets.ImageFolder(val_dir, transform=preprocess)
        classes = trainset.classes
        class_to_idx = trainset.class_to_idx
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )


    elif args.dataset == "sun397":
        root = getattr(args, "sun397_root", None)
        if root is None:
            root = 'data/sun397/SUN_dataset/data'
        root = os.path.expanduser(root)
        download = bool(getattr(args, "sun397_download", False))
        val_fraction = float(getattr(args, "sun397_val_fraction", 0.1))
        if not os.path.isdir(root) and not download:
            raise FileNotFoundError(
                f"SUN397 root '{root}' not found. Set --sun397-root to the extracted SUN_dataset/data folder or pass --sun397-download."
            )
        full_dataset = datasets.SUN397(root=root, transform=preprocess, download=download)
        classes = full_dataset.classes
        class_to_idx = full_dataset.class_to_idx
        idx_to_class = {v: k for k, v in class_to_idx.items()}

        val_size = max(1, int(len(full_dataset) * val_fraction))
        train_size = len(full_dataset) - val_size
        generator = torch.Generator().manual_seed(getattr(args, "seed", 0))
        trainset, testset = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=generator)

        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )


    else:
        raise ValueError(args.dataset)

    return train_loader, test_loader, idx_to_class, classes

