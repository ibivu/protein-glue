import tensorflow as tf
import click

from dataset.seqs import create_dataset

@click.command()
@click.argument('input_files', nargs=-1)
@click.option('--batch-size', default=32)
@click.option('--reduced-target-alphabet/--no-reduced-target-alphabet', default=False)
def main(batch_size, reduced_target_alphabet, input_files):
    ds = create_dataset(input_files, batch_size=batch_size, target_reduced_alphabet=reduced_target_alphabet, add_start_end_tokens=False)
    for (inp, tar) in ds:
        print(inp, tar)
        break


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()