import os.path
import glob
from .listdataset import ListDataset
from .util import split2list


def make_dataset(dir, split=None, split_save_path=None):
    """Will search for triplets that go by the pattern '[name]_img1.ppm  [name]_img2.ppm    [name]_flow.flo'"""
    images = []
    for flow_map in sorted(glob.glob(os.path.join(dir, "*_flow.flo"))):
        flow_map = os.path.basename(flow_map)
        root_filename = flow_map[:-9]
        img1 = root_filename + "_img1.ppm"
        img2 = root_filename + "_img2.ppm"
        if not (
            os.path.isfile(os.path.join(dir, img1))
            and os.path.isfile(os.path.join(dir, img2))
        ):
            continue

        images.append([[img1, img2], flow_map])
    return split2list(images, split, split_save_path, default_split=0.97)

def upsampling_loader(root, path_imgs, path_flo):
    imgs = [os.path.join(root, path) for path in path_imgs]
    flo = os.path.join(root, path_flo)
    return [lanczos_upsampling(img).astype(np.float32) for img in imgs], load_flo(flo)

def lanczos_upsampling(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Image not found or invalid image path")
    original_height, original_width = image.shape[:2]
    new_height, new_width = original_height * 2, original_width * 2
    upsampled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    return upsampled_image

def flying_chairs_lr(
    root,
    transform=None,
    target_transform=None,
    co_transform=None,
    split=None,
    split_save_path=None,
):
    train_list, test_list = make_dataset(root, split, split_save_path)
    train_dataset = ListDataset(
        root, train_list, transform, target_transform, co_transform, upsampling_loader
    )
    test_dataset = ListDataset(root, test_list, transform, target_transform, upsampling_loader)

    return train_dataset, test_dataset
