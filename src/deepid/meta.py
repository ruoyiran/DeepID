import json


class Meta(object):
    def __init__(self):
        self.num_train_examples = 0
        self.num_val_examples = 0
        self.num_test_examples = 0
        self.num_classes = 0
        self.image_width = 0
        self.image_height = 0
        self.num_channels = 0

    def save(self, path_to_json_file):
        with open(path_to_json_file, 'w') as f:
            content = {
                'num_examples': {
                    'train': self.num_train_examples,
                    'val': self.num_val_examples,
                    'test': self.num_test_examples,
                },
                'image_size': {
                    'width': self.image_width,
                    'height': self.image_height,
                    'channels': self.num_channels
                },
                'num_classes': int(self.num_classes)
            }
            json.dump(content, f, indent=4)

    def load(self, path_to_json_file):
        with open(path_to_json_file, 'r') as f:
            content = json.load(f)
            self.num_train_examples = content['num_examples']['train']
            self.num_val_examples = content['num_examples']['val']
            self.num_test_examples = content['num_examples']['test']


def get_meta_data(meta_file_path):
    meta = Meta()
    with open(meta_file_path, "r") as f:
        meta_data = json.load(f)
        image_width = meta_data['image_size']['width']
        image_height = meta_data['image_size']['height']
        num_channels = meta_data['image_size']['channels']
        num_classes = meta_data['num_classes']
        meta.num_train_examples = meta_data['num_examples']['train']
        meta.num_val_examples = meta_data['num_examples']['val']
        meta.num_test_examples = meta_data['num_examples']['test']
        meta.num_classes = num_classes
        meta.image_width = image_width
        meta.image_height = image_height
        meta.num_channels = num_channels
    return meta
