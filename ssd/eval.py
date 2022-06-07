from utils import *
from datasets import PascalVOCDataset, TestDataset
from tqdm import tqdm
from pprint import PrettyPrinter

# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()

# Parameters
data_folder = './'
keep_difficult = True  # difficult ground truth objects must always be considered in mAP calculation, because these objects DO exist!
batch_size = 64
workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = './checkpoint_ssd300.pth.tar'

# Load model checkpoint that is to be evaluated
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
model = model.to(device)

# Switch to eval mode
model.eval()

# Load test data
test_dataset = TestDataset(data_folder,
                                split='test',
                                keep_difficult=keep_difficult)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)


def evaluate(test_loader, model):
    """
    Evaluate.

    :param test_loader: DataLoader for test data
    :param model: model
    """

    # Make sure it's in eval mode
    model.eval()

    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()  # it is necessary to know which objects are 'difficult', see 'calculate_mAP' in utils.py

    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels, difficulties)  in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = images.to(device)  # (N, 3, 300, 300)

            # Forward prop.
            predicted_locs, predicted_scores = model(images)
            #print("locs",predicted_locs)
            #print(predicted_scores)
             
            #print(predicted_locs.size())
            #print(predicted_scores.size())
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                       min_score=0.01, max_overlap=0.45,
                                                                                       top_k=3)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            difficulties = [d.to(device) for d in difficulties]

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            #print(labels, type(labels))
            #print(labels.size())
            #break
            true_difficulties.extend(difficulties)

            #break

    #write file
    
    import json
    ids_path ="./../Public_Image"
    filenames = os.listdir(ids_path)
    filenames.sort(key=lambda x:int(x[-8:-4]))
    result = {}
    assert len(det_boxes) == len(filenames)
    for i in range(len(filenames)):
        result[filenames[i]]= []
        for j in range(len(det_boxes[i])):
            #print(type(true_boxes[j].cpu().numpy().tolist()))
            numbers = [1716,942, 1716, 942]
            tmp = [int(x*y) for x,y in zip(numbers,det_boxes[i][j].cpu().numpy().tolist())]
            result[filenames[i]].append( tmp+[0.99999])
            print(len(result[filenames[i]]))
        #break
    print("結果")
    print(type(result)) 
    File = open('results.json', "w")
    json.dump(result, File)
    File.close()

if __name__ == '__main__':
    evaluate(test_loader, model)
