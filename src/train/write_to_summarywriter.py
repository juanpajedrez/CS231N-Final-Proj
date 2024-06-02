from torch.utils.tensorboard import SummaryWriter

def write_to_summary_writer():
    # open logs from log file
    summary_writer = SummaryWriter(f"runs/vae-beta_1-10-epochs")

    log_file = 'src/train/summary_log.txt'
    with open(log_file, "r") as f:
        lines = f.readlines()

    # log_format: 2024-06-02_09-45-05 Epoch:  0 Iteration:  0 Loss:  1.5544651746749878
    # extract epoch, iteration, loss from log
    epoch = []
    iteration = []
    loss = []
    for line in lines:
        if "Epoch" in line:
            epoch.append(int(line.split()[2]))
            iteration.append(int(line.split()[4]))
            loss.append(float(line.split()[6]))

    # write to tensorboard
    for i in range(len(iteration)):
        summary_writer.add_scalar("loss", loss[i], epoch[i] * 180 + iteration[i])

if __name__ == "__main__":
    write_to_summary_writer()

    