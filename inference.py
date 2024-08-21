from dataloaders.data_utils import image_loader
from torchvision import transforms
import torch



def infer(args, model):
    model.eval()
    rescale = args.scale_size
    random_crop = args.crop_size
    attr_group_dict = args.attr_group_dict
    workers = args.workers
    n_groups = args.n_groups
    normTransform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    scale_size = rescale
    crop_size = random_crop
    testTransform = transforms.Compose([transforms.Resize((scale_size, scale_size)),
                                        transforms.CenterCrop(crop_size),
                                        transforms.ToTensor(),
                                        normTransform])

    image = image_loader(args.image_path, testTransform)[None,:]
    mask_in = torch.tensor([[-1,-1]])

    with torch.no_grad():
        pred,int_pred,attns = model(image.cuda(),mask_in.cuda())

    pred = torch.nn.functional.sigmoid(pred)
    print("Probabilties:\n"
          f"Hood: \t\t{pred[0,0]}\n"
          f"Backdoor Left: \t{pred[0,1]}")


    # import pdb; pdb.set_trace()
