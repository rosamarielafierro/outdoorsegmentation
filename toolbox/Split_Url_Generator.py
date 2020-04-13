import csv
import os


def main(split, root_dir):

images_dir = os.path.join(root_dir, 'leftImg8bit', split)
targets_dir = os.path.join(root_dir, 'gtFine', split)
images = ['image_urls']
targets = ['target_urls']

# CREATE URL LISTS
for city in os.listdir(images_dir):
    img_dir = os.path.join(images_dir, city)
    target_dir = os.path.join(targets_dir, city)
    for file_name in os.listdir(img_dir):
        img_id = file_name.split('_leftImg8bit')[0]
        target_name = '{}_{}_{}'.format(img_id,'gtFine', 'labelTrainIds.png')
        images.append(os.path.join(img_dir, file_name))
        targets.append(os.path.join(target_dir, target_name))
rows = zip(images,targets)

# SAVE SPLIT FILE
with open(root_dir+"/"+split+"_split_urls.csv", "w") as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)

"""
Los argumentos son son:
    ~ Split (-split)
        - train
        - val
        - test
    ~ root_dir (-path)

"""

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-split', '--split', default='train', help='Split to be generated')
    parser.add_argument('-path', '--path', default='', help='Path to locate the dataset')
    args = parser.parse_args()
    print('{}, {}'.format(args.split, args.path))

    main(split= args.split, root_dir=args.path)
