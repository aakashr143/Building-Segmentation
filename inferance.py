import torch
from torchvision import transforms
from models import DeepLabV3Plus, UNet, UANet, Encoder
from sateImageDataset import get_datasets
from metrics import Metrics
from edge_improvement import polymerization, blur, CRP

CHECKPOINT = "Checkpoints/UANet_ResNet50_K_1/2024-07-13 20_13_50/model_49.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "mps"

checkpoint = torch.load(CHECKPOINT, map_location=torch.device('mps'))

model = UANet().to(DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

_, _, test_dataset = get_datasets(1)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=os.cpu_count())

metrics = Metrics()

inverse_normalise = transforms.Compose([
    transforms.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
    transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
])

running_items = 0
running_metrics = {
    "background": {
        "iou": 0,
        "precision": 0,
        "recall": 0,
        "f1score": 0
    },
    "building": {
        "iou": 0,
        "precision": 0,
        "recall": 0,
        "f1score": 0
    }
}
running_metrics_edge = {
    "background": {
        "iou": 0,
        "precision": 0,
        "recall": 0,
        "f1score": 0
    },
    "building": {
        "iou": 0,
        "precision": 0,
        "recall": 0,
        "f1score": 0
    }
}

with torch.no_grad():
    for i in range(len(test_dataset)):
        if i % 100 == 0:
            print(i)

        X, Y = test_dataset.__getitem__(i)
        X, Y = X.to(DEVICE).unsqueeze(0), Y.to(DEVICE).unsqueeze(0)

        pred = model(X)
        pred_edge = polymerization(pred.cpu().detach().clone()).to(DEVICE)
        # pred_edge = blur(pred.cpu().detach().clone()).to(DEVICE)
        # pred_edge = CRP(X.cpu().detach().clone(), pred.cpu().detach().clone()).to(DEVICE)

        m = metrics(pred, Y.to(torch.long))
        m_edge = metrics(pred_edge, Y.to(torch.long))

        running_items += m["batch_size"]
        for key, value in m["background"].items():
            running_metrics["background"][key] += value

        for key, value in m["building"].items():
            running_metrics["building"][key] += value

        for key, value in m_edge["background"].items():
            running_metrics_edge["background"][key] += value

        for key, value in m_edge["building"].items():
            running_metrics_edge["building"][key] += value


for key, value in m["background"].items():
    running_metrics["background"][key] = running_metrics["background"][key] / running_items

for key, value in m["building"].items():
    running_metrics["building"][key] = running_metrics["building"][key] / running_items


for key, value in m_edge["background"].items():
    running_metrics_edge["background"][key] = running_metrics_edge["background"][key] / running_items

for key, value in m_edge["building"].items():
    running_metrics_edge["building"][key] = running_metrics_edge["building"][key] / running_items


print("NORMAL")
print(running_metrics)
print("EDGES")
print(running_metrics_edge)
