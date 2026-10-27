import yaml

def read_yaml(file : str):
    with open(file,"r") as conf_file:
        config = yaml.safe_load(conf_file)
    return config
