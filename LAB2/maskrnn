import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

Classes = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep'
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def show(img, boxes, masks, labels, scores, thresh=0.8):
    plt.imshow(img)
    ax = plt.gca()
    
    for box, mask, label, score in zip(boxes, masks, labels, scores):
        if score < thresh or label not in Classes:
            continue
        
        color = (random.random(), random.random(), random.random())
        x1, y1, x2, y2 = box
        
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   edgecolor=color, facecolor='none', lw=2))
        ax.imshow(np.ma.masked_where(mask == 0, mask), cmap='jet', alpha=0.6)
        ax.text(x1, y1, f"{Classes[label]}: {score:.2f}", 
                color='white', backgroundcolor=color)

    plt.axis('off')
    plt.show()

def main(path):
    img = Image.open(path).convert('RGB')
    tensor_img = ToTensor()(img).to(device)

    model = maskrcnn_resnet50_fpn(weights='DEFAULT').to(device).eval()

    with torch.no_grad():
        output = model([tensor_img])[0]

    boxes = output['boxes'].cpu().numpy()
    masks = output['masks'].cpu().numpy()[:, 0] > 0.5
    labels = output['labels'].cpu().numpy()
    scores = output['scores'].cpu().numpy()

    show(np.array(img), boxes, masks, labels, scores)

if __name__ == '__main__':
    main("path/to/your/image.png")
