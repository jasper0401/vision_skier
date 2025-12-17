
import torchvision.transforms as transforms

def build_transforms(transform_list = None):
    transform_funcs = []
    if transform_list:
        for func in transform_list:
            transform_funcs.append(getattr(func))
    
    # append the basic transforms
    transform_funcs.append(transforms.ToTensor())
    transform_funcs.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
    return  transforms.Compose(transform_funcs)

if __name__ == "__main__":
    transform_ops = build_transforms()
    print (transform_ops)


