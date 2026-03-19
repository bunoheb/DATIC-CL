from .utils import Cutout
from .rvlcdip import get_rvlcdip_dataset

# TODO: other text  datasets like ptb, wikitext, etc.



all = [
    'Cutout',
    
    'get_rvlcdip_dataset'
]


data_dict = {
    'rvl': lambda data_dir, **kwargs: get_rvlcdip_dataset(data_dir, **kwargs)
}


def get_dataset(data_dir, data_name):

    assert data_name in data_dict, \
        'Assert Error: data_name should be in ' + str(list(data_dict.keys()))
    
    return data_dict[data_name](data_dir)


def get_dataset_with_noise(data_dir, data_name, use_huggingface=True):
    """Load Hugging Face or CSV dataset according to data_name"""

    if 'noise' in data_name:
        try:
            parts = data_name.split('-')
            data_name = parts[0] 
            noise_ratio = float(parts[-1]) 
        except:
            raise AssertionError('Assert Error: data_name should be [dataset]-noise-[ratio]')
        assert 0.0 <= noise_ratio <= 1.0, 'Assert Error: noise ratio should be in range [0.0, 1.0]'
    else:
        noise_ratio = 0.0

    # ✅ `use_huggingface` check
    print(f"🔥 use_huggingface in get_dataset_with_noise: {use_huggingface}")

    # dataset name validation
    assert data_name in data_dict, \
        'Assert Error: data_name should be in ' + str(list(data_dict.keys()))

    # dataset loader function
    return data_dict[data_name](data_dir, noise_ratio=noise_ratio, use_huggingface=use_huggingface)
