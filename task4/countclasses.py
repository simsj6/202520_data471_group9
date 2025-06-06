with open("proj_data/task4_phoneme/dev.CT") as file:
    counter = 0
    for line in file:
        if line.strip() == '59':
            counter += 1

print(counter)