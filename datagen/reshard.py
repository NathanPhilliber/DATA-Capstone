from utils import *
import click


@click.command()
@click.option('--set-name', prompt='Name of dataset to modify.')
@click.option('--shard-size', prompt='New size of shard.')
def main(set_name, shard_size):
    data_dir = os.path.join(DATA_DIR, set_name)
    if not os.path.exists(data_dir):
        print(f"{data_dir} does not exist.")
    files = os.listdir(data_dir)

    if DATAGEN_CONFIG in files:
        files.remove(DATAGEN_CONFIG)

    print(files)

if __name__ == "__main__":
    main()
