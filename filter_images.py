import os
from tqdm import tqdm
import json
from argparse import ArgumentParser

# Dictionary mapping class IDs to descriptions
CAPTION_MAPPING = {
    1: "Accessible entrances/door",
    2: "Accessible pathway/wheelchair ramps",
    301: "Directional guide paths",
    302: "Warning guide paths",
    4: "Dropped kerbs",
    5: "Accessible lifts/elevators",
    6: "Accessible signage",
    7: "Braille and tactile floor plans/maps",
    8: "Accessible carparks",
    9: "Audible/visual signaling devices",
    10: "People with disabilities"
}

def calculate_bbox_area_percentage(bbox):
    """
    Calculate the percentage of image area occupied by a bounding box.
    
    Args:
        bbox (str or list): Bounding box in format "class_id center_x center_y width height"
                           or [class_id, center_x, center_y, width, height]
    
    Returns:
        float: Percentage of image area occupied by the bounding box (0-100)
    """
    # Parse bbox if it's a string
    if isinstance(bbox, str):
        bbox = [float(x) for x in bbox.split()]
    
    # Extract width and height (normalized values)
    width = bbox[3]
    height = bbox[4]
    
    # Calculate area percentage
    # Since width and height are normalized (0-1), 
    # their product directly gives us the percentage in decimal form
    area_percentage = width * height
    
    return area_percentage


def get_large_bbox_images(image_dir="test_labeled/image", label_dir="test_labeled/label_xml", min_occupancy=0.1):
    """
    Find images where all bounding boxes occupy at least the specified percentage of image area.
    
    Args:
        image_dir: Directory containing images
        label_dir: Directory containing label files
        min_occupancy: Minimum occupancy threshold (0-1)
    
    Returns:
        list: List of image names that meet the criteria
    """
    files = os.listdir(image_dir)
    pbar = tqdm(files)
    
    # Dictionary to store occupancy data per image
    image_occupancies = {}
    image_labels = {}
    
    for img_name in pbar:
        # pbar.set_description(f"Processing {img_name}")
        label_file = os.path.join(label_dir, img_name.replace('jpg', 'txt'))
        image_labels[img_name] = []
        # Skip if label file doesn't exist
        if not os.path.exists(label_file):
            continue
            
        with open(label_file) as f:
            labels = f.readlines()
        
        # Store all bbox occupancies for this image
        image_occupancies[img_name] = []
        
        for label in labels:
            info = [float(x) for x in label.strip().split()]
            image_labels[img_name].append(int(info[0]))
            occupancy = calculate_bbox_area_percentage(info)
            image_occupancies[img_name].append(occupancy)
    
    # Filter images where all bboxes meet the minimum occupancy threshold
    qualifying_images = [
        img_name for img_name, occupancies in image_occupancies.items()
        if occupancies and all(occ >= min_occupancy for occ in occupancies)
    ]
    
    return qualifying_images, image_labels




def build_caption_pairs(image_list, image_labels, img_path):
    pbar=tqdm(image_list)
    filtered_images=dict()
    count=0
    img_label_pairs=[]
    for img_name in pbar:
        captions=[]
        for label in image_labels[img_name]:
            try:
                caption = CAPTION_MAPPING[label]
                captions.append(caption)
            except KeyError:
                continue
    
        # img_label_pairs.append((img_name, "\n".join(captions))) if captions != [] else None
        if captions != []:
            filtered_images[count]={
                "image_path": os.path.join(img_path,img_name),
                "caption":captions
            }
            count+=1

        
    return filtered_images

def main(args):
    img_path=args.img_path
    label_path=args.label_path
    qualifying_images, image_labels=get_large_bbox_images(image_dir=img_path, \
        label_dir=label_path, min_occupancy=args.min_occ)
    
    filtered_images=build_caption_pairs(qualifying_images, image_labels, img_path)
    
    with open(f"filtered_images_{args.min_occ}_{img_path.split('/')[-2]}.json", "w") as f:
        json.dump(filtered_images, f, indent=4)

# Example usage
if __name__ == "__main__":

    # img_path="SVI2_Label/img"
    # label_path="SVI2_Label/xml_merged"
    # img_path="test_labeled/image"
    # label_path="test_labeled/label_xml"
    parser=ArgumentParser()
    parser.add_argument("--img_path", type=str, default="SVI2_Label/img")
    parser.add_argument("--label_path", type=str, default="SVI2_Label/xml_merged")
    parser.add_argument("--min_occ", type=float, default=0.04)
    args=parser.parse_args()
    main(args)
    
    
