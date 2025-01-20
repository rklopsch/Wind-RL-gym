import torch
from tensordict import TensorDict
import sys


def recursively_shorten(td, new_td, length):
    for key in td.keys():
        if isinstance(td.get(key), torch.Tensor):
            new_td.set(key, td.get(key)[:length])
        else:
            recursively_shorten(td.get(key), new_td, length)
    return new_td


if __name__ == '__main__':
    # Check if an argument is provided
    if len(sys.argv) != 2:
        print("Usage: python shorten_replay_buffer.py <filename_of_buffer>")
        sys.exit(1)

    # Get the command-line argument
    filename = sys.argv[1]

    # load buffer
    buffer = torch.load(filename)

    # figure out the actual length
    key = ("_data", "action")
    for i in range(buffer.get(key).shape[0]):
        if not buffer.get(key)[i].any():
            actual_length = i
            break

    print(f"Stored length {buffer.get(key).shape[0]} | Actual length {actual_length}")

    shortened_buffer = TensorDict({}, batch_size=actual_length)
    shortened_buffer = recursively_shorten(buffer, shortened_buffer, actual_length)

    print("Old tensordict", buffer)

    print("New tensordict", shortened_buffer)

    torch.save(shortened_buffer, filename[:-3] + "_shortened.pt")
    print("Saved shortened buffer at " + filename[:-3] + "_shortened.pt")

    print("All done")
